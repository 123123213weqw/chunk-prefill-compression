#!/usr/bin/env python3
"""Toy experiment: compress the detailed front span of one record by KV block means."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from run_prefill_experiment import (
    answer_matches,
    assert_cache_length,
    greedy_answer,
    kl_from_logprobs,
    prefill,
    score_gold_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--target-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    return parser.parse_args()


def clone_cache(cache: DynamicCache, config: Any) -> DynamicCache:
    data = [
        (layer.keys.detach().clone(), layer.values.detach().clone())
        for layer in cache.layers
    ]
    return DynamicCache(ddp_cache_data=data, config=config)


@torch.inference_mode()
def blockmean_t_in_place(
    cache: DynamicCache,
    s: int,
    t: int,
    r: int,
    memory_slots: int,
) -> DynamicCache:
    if memory_slots < 0 or memory_slots > t:
        raise ValueError(f"invalid memory_slots={memory_slots} for T={t}")
    if memory_slots and t % memory_slots:
        raise ValueError(f"T={t} is not divisible by memory_slots={memory_slots}")

    r_start = s + t
    r_end = r_start + r
    for layer in cache.layers:
        k_s = layer.keys[..., :s, :]
        v_s = layer.values[..., :s, :]
        k_r = layer.keys[..., r_start:r_end, :]
        v_r = layer.values[..., r_start:r_end, :]

        if memory_slots:
            b, h, _, d = layer.keys[..., s:r_start, :].shape
            group_size = t // memory_slots
            k_c = (
                layer.keys[..., s:r_start, :]
                .reshape(b, h, memory_slots, group_size, d)
                .mean(dim=3)
            )
            v_c = (
                layer.values[..., s:r_start, :]
                .reshape(b, h, memory_slots, group_size, d)
                .mean(dim=3)
            )
            layer.keys = torch.cat((k_s, k_c, k_r), dim=-2)
            layer.values = torch.cat((v_s, v_c, v_r), dim=-2)
        else:
            layer.keys = torch.cat((k_s, k_r), dim=-2)
            layer.values = torch.cat((v_s, v_r), dim=-2)
    return cache


def make_detailed_t(tokenizer: Any, target_tokens: int) -> tuple[list[int], str]:
    critical = (
        "这条详细记录的前半部分包含以下已经核验的事实：目标服务器 IPv4 地址为 "
        "10.73.18.204，SSH 监听端口为 28443，请求标识为 REQ-7M4K9P2X，部署区域为 "
        "cn-north-6。服务器上的 nginx 配置检查失败，错误文件为 "
        "/etc/nginx/conf.d/api-prod.conf，第 87 行缺少分号；证书加载成功，upstream 网络健康，"
        "所以它们都不是本次失败原因。"
    )
    filler = (
        "补充记录显示 CPU、内存、磁盘、调度器、审计日志和监控采样均处于正常范围，"
        "这些描述仅用于模拟一段很详细但大部分与问题无关的文字。"
    )
    ids = tokenizer.encode(critical, add_special_tokens=False)
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    if len(ids) > target_tokens:
        raise ValueError(f"critical text already has {len(ids)} tokens > target {target_tokens}")
    while len(ids) < target_tokens:
        take = min(len(filler_ids), target_tokens - len(ids))
        ids.extend(filler_ids[:take])
    return ids, tokenizer.decode(ids, skip_special_tokens=False)


def probe_specs() -> list[dict[str, Any]]:
    return [
        {
            "probe_id": "endpoint",
            "question": "前面详细记录中的服务器 IPv4 地址和 SSH 端口是什么？只输出实际值，格式示例：`192.0.2.1:22`。",
            "gold": {"type": "exact", "answer": "10.73.18.204:28443"},
        },
        {
            "probe_id": "request-region",
            "question": "前面记录中的 request_id 和 deployment_region 是什么？只输出 JSON。",
            "gold": {
                "type": "json_fields",
                "answer": {
                    "request_id": "REQ-7M4K9P2X",
                    "deployment_region": "cn-north-6",
                },
            },
        },
        {
            "probe_id": "root-cause",
            "question": "nginx 配置检查失败的直接根因是什么？只输出根因。",
            "gold": {
                "type": "semantic_exact",
                "answer": "缺少分号",
                "aliases": ["missing semicolon"],
            },
        },
    ]


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    s_text = (
        "<|im_start|>system\n你是严格的信息读取助手，必须记住记录中的精确字段并依据记录回答。"
        "<|im_end|>\n<|im_start|>user\n"
    )
    t_ids, t_text = make_detailed_t(tokenizer, args.target_tokens)
    r_text = (
        "这条记录的后半部分说明：记录已经读取完毕，后续回答只给出被询问的实际字段或结论，"
        "不要虚构未出现的值。<|im_end|>\n"
        "<|im_start|>assistant\n已读取并核验完整记录。<|im_end|>\n"
    )
    s_ids = tokenizer.encode(s_text, add_special_tokens=False)
    r_ids = tokenizer.encode(r_text, add_special_tokens=False)
    s, t, r = len(s_ids), len(t_ids), len(r_ids)
    logical_next = s + t + r

    full_cache = DynamicCache(config=model.config)
    prefill(model, full_cache, s_ids, 0, 256)
    prefill(model, full_cache, t_ids, s, 256)
    prefill(model, full_cache, r_ids, s + t, 256)
    assert_cache_length(full_cache, logical_next, "toy.full")

    slot_counts = [t, t // 2, t // 4, t // 8, t // 16, t // 32, 0]
    slot_counts = list(dict.fromkeys(x for x in slot_counts if x >= 0 and (x == 0 or t % x == 0)))
    all_results: dict[str, Any] = {}
    distributions: dict[str, dict[str, torch.Tensor]] = {}

    for memory_slots in slot_counts:
        label = "full_identity" if memory_slots == t else (
            "drop_all" if memory_slots == 0 else f"blockmean_m{memory_slots:03d}"
        )
        condition_cache = clone_cache(full_cache, model.config)
        blockmean_t_in_place(condition_cache, s, t, r, memory_slots)
        physical_length = s + memory_slots + r
        assert_cache_length(condition_cache, physical_length, label)

        if memory_slots == t:
            for original, identity in zip(full_cache.layers, condition_cache.layers):
                if not torch.equal(original.keys, identity.keys) or not torch.equal(
                    original.values, identity.values
                ):
                    raise AssertionError("m=T identity transform changed KV")

        probe_results = []
        distributions[label] = {}
        for probe in probe_specs():
            q_text = (
                f"<|im_start|>user\n{probe['question']}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            q_ids = tokenizer.encode(q_text, add_special_tokens=False)
            gold_score = score_gold_sequence(
                model,
                tokenizer,
                condition_cache,
                q_ids,
                probe["gold"],
                logical_next,
                physical_length,
            )
            decoded = greedy_answer(
                model,
                tokenizer,
                condition_cache,
                q_ids,
                logical_next,
                args.max_new_tokens,
                physical_length,
            )
            distributions[label][probe["probe_id"]] = decoded.first_token_logprobs
            probe_results.append(
                {
                    "probe_id": probe["probe_id"],
                    "gold": probe["gold"],
                    "answer": decoded.text,
                    "match": answer_matches(decoded.text, probe["gold"]),
                    "gold_mean_logprob": gold_score["mean_logprob"],
                    "gold_perplexity": gold_score["perplexity"],
                }
            )

        all_results[label] = {
            "memory_slots": memory_slots,
            "group_size": None if memory_slots == 0 else t // memory_slots,
            "compression_ratio": None if memory_slots == 0 else t / memory_slots,
            "physical_prefix_tokens": physical_length,
            "correct": sum(item["match"] for item in probe_results),
            "total": len(probe_results),
            "probes": probe_results,
        }

    full_dist = distributions["full_identity"]
    for label, condition in all_results.items():
        condition["first_token_kl_from_full"] = {
            probe_id: kl_from_logprobs(full_dist[probe_id], distributions[label][probe_id])
            for probe_id in full_dist
        }

    output = {
        "experiment": "KV-BlockMean-v0-toy",
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "transform": {
            "K_C[j]": "mean(K_T[j*g:(j+1)*g])",
            "V_C[j]": "mean(V_T[j*g:(j+1)*g])",
            "axis": "token sequence, independently per layer and KV head",
        },
        "segments": {
            "S_tokens": s,
            "T_tokens": t,
            "R_tokens": r,
            "logical_next_position": logical_next,
            "T_text": t_text,
            "R_text": r_text,
        },
        "results": all_results,
        "created_unix": time.time(),
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
