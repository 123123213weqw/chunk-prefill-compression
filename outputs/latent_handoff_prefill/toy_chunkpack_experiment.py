#!/usr/bin/env python3
"""Toy chunk-prefill and selective chunk-pack feasibility experiment."""

from __future__ import annotations

import argparse
import json
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
from toy_blockmean_experiment import make_detailed_t, probe_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--target-tokens", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--context-radius", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def find_all(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    return [
        i
        for i in range(len(haystack) - len(needle) + 1)
        if haystack[i : i + len(needle)] == needle
    ]


def build_protected_positions(
    tokenizer: Any,
    t_ids: list[int],
    chunk_size: int,
    radius: int,
) -> tuple[list[int], dict[str, Any]]:
    # Oracle/entity-protect upper bound. It never reads Q while running, but the
    # phrase list is hand-written for this toy record and is not a deployable selector.
    phrases = [
        "10.73.18.204",
        "28443",
        "REQ-7M4K9P2X",
        "cn-north-6",
        "/etc/nginx/conf.d/api-prod.conf",
        "第 87 行缺少分号",
        "证书加载成功",
        "upstream 网络健康",
    ]
    protected: set[int] = set()
    phrase_hits: dict[str, list[list[int]]] = {}
    for phrase in phrases:
        variants = [
            tokenizer.encode(phrase, add_special_tokens=False),
            tokenizer.encode(" " + phrase, add_special_tokens=False),
        ]
        spans: list[list[int]] = []
        seen_spans: set[tuple[int, int]] = set()
        for needle in variants:
            for start in find_all(t_ids, needle):
                span = (start, start + len(needle))
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                spans.append([span[0], span[1]])
                left = max(0, start - radius)
                right = min(len(t_ids), start + len(needle) + radius)
                protected.update(range(left, right))
        phrase_hits[phrase] = spans
        if not spans:
            raise AssertionError(f"protected phrase not found in T token IDs: {phrase!r}")

    # Preserve a very small boundary scaffold for every chunk, even pure filler.
    for chunk_start in range(0, len(t_ids), chunk_size):
        chunk_end = min(len(t_ids), chunk_start + chunk_size)
        protected.update(range(chunk_start, min(chunk_start + 2, chunk_end)))
        protected.update(range(max(chunk_start, chunk_end - 2), chunk_end))

    return sorted(protected), {
        "phrases": phrases,
        "phrase_token_spans": phrase_hits,
        "context_radius": radius,
        "boundary_tokens_per_side": 2,
    }


@torch.inference_mode()
def keep_tail_positions_in_place(
    cache: DynamicCache,
    physical_before: int,
    keep_local: list[int],
) -> None:
    for layer in cache.layers:
        device = layer.keys.device
        indices = torch.tensor(
            [physical_before + index for index in keep_local],
            dtype=torch.long,
            device=device,
        )
        k_tail = layer.keys.index_select(-2, indices) if keep_local else layer.keys[..., :0, :]
        v_tail = layer.values.index_select(-2, indices) if keep_local else layer.values[..., :0, :]
        layer.keys = torch.cat((layer.keys[..., :physical_before, :], k_tail), dim=-2)
        layer.values = torch.cat((layer.values[..., :physical_before, :], v_tail), dim=-2)


@torch.inference_mode()
def build_streaming_selective_cache(
    model: Any,
    s_ids: list[int],
    t_ids: list[int],
    r_ids: list[int],
    protected_positions: list[int],
    chunk_size: int,
) -> tuple[DynamicCache, list[dict[str, Any]], int]:
    cache = DynamicCache(config=model.config)
    s = len(s_ids)
    t = len(t_ids)
    prefill(model, cache, s_ids, 0, chunk_size)
    protected_set = set(protected_positions)
    chunks = []
    peak_physical = len(s_ids)

    for chunk_index, offset in enumerate(range(0, t, chunk_size)):
        chunk = t_ids[offset : offset + chunk_size]
        physical_before = int(cache.get_seq_length())
        prefill(model, cache, chunk, s + offset, chunk_size)
        peak_physical = max(peak_physical, int(cache.get_seq_length()))
        keep_local = [
            local
            for local in range(len(chunk))
            if offset + local in protected_set
        ]
        keep_tail_positions_in_place(cache, physical_before, keep_local)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "logical_T_range": [offset, offset + len(chunk)],
                "raw_tokens": len(chunk),
                "kept_tokens": len(keep_local),
                "kept_logical_T_positions": [offset + x for x in keep_local],
                "physical_after_pack": int(cache.get_seq_length()),
            }
        )

    prefill(model, cache, r_ids, s + t, chunk_size)
    return cache, chunks, peak_physical


def cache_max_difference(a: DynamicCache, b: DynamicCache) -> dict[str, float]:
    max_k = 0.0
    max_v = 0.0
    mean_k_sum = 0.0
    mean_v_sum = 0.0
    for left, right in zip(a.layers, b.layers):
        k_diff = (left.keys.float() - right.keys.float()).abs()
        v_diff = (left.values.float() - right.values.float()).abs()
        max_k = max(max_k, float(k_diff.max().item()))
        max_v = max(max_v, float(v_diff.max().item()))
        mean_k_sum += float(k_diff.mean().item())
        mean_v_sum += float(v_diff.mean().item())
    layers = len(a.layers)
    return {
        "max_abs_key": max_k,
        "max_abs_value": max_v,
        "mean_abs_key_across_layers": mean_k_sum / layers,
        "mean_abs_value_across_layers": mean_v_sum / layers,
    }


def run_condition(
    model: Any,
    tokenizer: Any,
    cache: DynamicCache,
    logical_next: int,
    physical_length: int,
    max_new_tokens: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    probes = []
    distributions = {}
    for probe in probe_specs():
        q_text = (
            f"<|im_start|>user\n{probe['question']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        q_ids = tokenizer.encode(q_text, add_special_tokens=False)
        score = score_gold_sequence(
            model,
            tokenizer,
            cache,
            q_ids,
            probe["gold"],
            logical_next,
            physical_length,
        )
        decoded = greedy_answer(
            model,
            tokenizer,
            cache,
            q_ids,
            logical_next,
            max_new_tokens,
            physical_length,
        )
        match = answer_matches(decoded.text, probe["gold"])
        probes.append(
            {
                "probe_id": probe["probe_id"],
                "gold": probe["gold"],
                "answer": decoded.text,
                "match": match,
                "gold_mean_logprob": score["mean_logprob"],
            }
        )
        distributions[probe["probe_id"]] = decoded.first_token_logprobs
    return {
        "correct": sum(x["match"] for x in probes),
        "total": len(probes),
        "probes": probes,
    }, distributions


def main() -> None:
    args = parse_args()
    if args.target_tokens % args.chunk_size:
        raise ValueError("target-tokens must be divisible by chunk-size for this toy")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
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

    # Reference: one forward containing the complete prefix.
    one_shot = DynamicCache(config=model.config)
    prefill(model, one_shot, s_ids + t_ids + r_ids, 0, logical_next)
    assert_cache_length(one_shot, logical_next, "one-shot")

    # Mathematically equivalent chunked prefill; all raw KV is retained.
    chunked_full = DynamicCache(config=model.config)
    prefill(model, chunked_full, s_ids, 0, args.chunk_size)
    prefill(model, chunked_full, t_ids, s, args.chunk_size)
    prefill(model, chunked_full, r_ids, s + t, args.chunk_size)
    assert_cache_length(chunked_full, logical_next, "chunked-full")

    protected, selector_metadata = build_protected_positions(
        tokenizer, t_ids, args.chunk_size, args.context_radius
    )
    selective, chunk_records, peak_physical = build_streaming_selective_cache(
        model,
        s_ids,
        t_ids,
        r_ids,
        protected,
        args.chunk_size,
    )
    selective_physical = s + len(protected) + r
    assert_cache_length(selective, selective_physical, "selective-chunkpack")

    # Zero-information baseline: omit T but preserve logical R/Q positions.
    drop_all = DynamicCache(config=model.config)
    prefill(model, drop_all, s_ids, 0, args.chunk_size)
    prefill(model, drop_all, r_ids, s + t, args.chunk_size)
    assert_cache_length(drop_all, s + r, "drop-all")

    caches = {
        "one_shot_full": (one_shot, logical_next),
        "chunked_full_b064": (chunked_full, logical_next),
        "selective_chunkpack_b064": (selective, selective_physical),
        "drop_all_T": (drop_all, s + r),
    }
    results = {}
    distributions = {}
    for name, (cache, physical_length) in caches.items():
        result, dist = run_condition(
            model,
            tokenizer,
            cache,
            logical_next,
            physical_length,
            args.max_new_tokens,
        )
        result["physical_prefix_tokens"] = physical_length
        results[name] = result
        distributions[name] = dist

    reference = distributions["one_shot_full"]
    for name, result in results.items():
        result["first_token_kl_from_one_shot"] = {
            probe_id: kl_from_logprobs(reference[probe_id], distributions[name][probe_id])
            for probe_id in reference
        }

    output = {
        "experiment": "ChunkPack-Selective-v1-toy",
        "status": "PASS",
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "segments": {
            "S_tokens": s,
            "T_tokens": t,
            "R_tokens": r,
            "T_text": t_text,
            "logical_next_position": logical_next,
        },
        "chunking": {
            "chunk_size": args.chunk_size,
            "num_chunks": t // args.chunk_size,
            "selector": "oracle_entity_span_plus_boundary_scaffold",
            "selector_metadata": selector_metadata,
            "protected_T_tokens": len(protected),
            "compression_ratio_T_only": t / len(protected),
            "peak_physical_tokens_during_streaming_T": peak_physical,
            "chunks": chunk_records,
        },
        "one_shot_vs_chunked_full_cache_difference": cache_max_difference(
            one_shot, chunked_full
        ),
        "results": results,
        "created_unix": time.time(),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
