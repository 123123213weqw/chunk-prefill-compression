#!/usr/bin/env python3
"""Three-arm KV-cache experiment for the latent-handoff dataset.

The experiment deliberately does not call ``model.generate``.  After the T
span is evicted, physical cache length and logical RoPE position are different,
so every forward pass supplies explicit position IDs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


CONDITIONS = ("full", "evict", "replay")
DERIVED_CANARY_FAMILIES = (
    "nginx_configuration_failure",
    "database_migration_failure",
    "python_test_failure",
    "kubernetes_rollout_failure",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local HF model directory")
    parser.add_argument("--dataset", required=True, help="latent_handoff_v1.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument(
        "--canary-grid",
        action="store_true",
        help="Select one sample from every (template,T,R) cell",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def choose_samples(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_id:
        wanted = set(args.sample_id)
        selected = [row for row in rows if row["sample_id"] in wanted]
        missing = sorted(wanted - {row["sample_id"] for row in selected})
        if missing:
            raise ValueError(f"unknown sample IDs: {missing}")
    elif args.canary_grid:
        rows_by_cell: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (
                row["template_type"],
                row["token_targets"]["T"],
                row["token_targets"]["R"],
            )
            rows_by_cell[key].append(row)

        selected = []
        derived_index = 0
        for key in sorted(rows_by_cell):
            candidates = rows_by_cell[key]
            if key[0] == "derived_workflow_state":
                target_family = DERIVED_CANARY_FAMILIES[
                    derived_index % len(DERIVED_CANARY_FAMILIES)
                ]
                matching = [
                    row for row in candidates if row.get("scenario_family") == target_family
                ]
                if not matching:
                    raise AssertionError(
                        f"canary cell {key} has no sample for family {target_family}"
                    )
                selected.append(matching[0])
                derived_index += 1
            else:
                selected.append(candidates[0])
    else:
        selected = rows

    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("no samples selected")
    return selected


def token_sha256(ids: list[int]) -> str:
    # This is exactly the representation used by the dataset generator.
    payload = ",".join(str(token_id) for token_id in ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def encode_and_validate(tokenizer: Any, segment: dict[str, Any], label: str) -> list[int]:
    ids = tokenizer.encode(segment["text"], add_special_tokens=False)
    if len(ids) != segment["token_count"]:
        raise AssertionError(
            f"{label}: token count changed: {len(ids)} != {segment['token_count']}"
        )
    actual_hash = token_sha256(ids)
    if actual_hash != segment["token_id_sha256"]:
        raise AssertionError(
            f"{label}: token hash changed: {actual_hash} != {segment['token_id_sha256']}"
        )
    return ids


def cache_length(cache: DynamicCache) -> int:
    return int(cache.get_seq_length())


def assert_cache_length(cache: DynamicCache, expected: int, label: str) -> None:
    actual = cache_length(cache)
    if actual != expected:
        raise AssertionError(f"{label}: cache length {actual} != {expected}")
    for layer_index, layer in enumerate(cache.layers):
        if not layer.is_initialized:
            raise AssertionError(f"{label}: layer {layer_index} is uninitialized")
        key_length = int(layer.keys.shape[-2])
        value_length = int(layer.values.shape[-2])
        if key_length != expected or value_length != expected:
            raise AssertionError(
                f"{label}: layer {layer_index} K/V lengths "
                f"{key_length}/{value_length} != {expected}"
            )


@torch.inference_mode()
def forward_block(
    model: Any,
    cache: DynamicCache,
    token_ids: list[int],
    logical_start: int,
) -> Any:
    if not token_ids:
        raise ValueError("forward_block received an empty token sequence")
    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    position_ids = torch.arange(
        logical_start,
        logical_start + len(token_ids),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    return model(
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )


@torch.inference_mode()
def prefill(
    model: Any,
    cache: DynamicCache,
    token_ids: list[int],
    logical_start: int,
    chunk_size: int,
) -> DynamicCache:
    for offset in range(0, len(token_ids), chunk_size):
        forward_block(
            model,
            cache,
            token_ids[offset : offset + chunk_size],
            logical_start + offset,
        )
    return cache


def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def span_hashes(cache: DynamicCache, start: int, end: int) -> list[dict[str, str]]:
    hashes = []
    for layer in cache.layers:
        hashes.append(
            {
                "k": tensor_sha256(layer.keys[..., start:end, :]),
                "v": tensor_sha256(layer.values[..., start:end, :]),
            }
        )
    return hashes


@torch.inference_mode()
def evict_t_in_place(cache: DynamicCache, s: int, t: int, r: int) -> DynamicCache:
    r_start = s + t
    r_end = r_start + r
    for layer in cache.layers:
        layer.keys = torch.cat(
            (layer.keys[..., :s, :], layer.keys[..., r_start:r_end, :]),
            dim=-2,
        )
        layer.values = torch.cat(
            (layer.values[..., :s, :], layer.values[..., r_start:r_end, :]),
            dim=-2,
        )
    return cache


def crop_cache(cache: DynamicCache, prefix_physical_length: int) -> None:
    cache.crop(prefix_physical_length)
    assert_cache_length(cache, prefix_physical_length, "post-probe crop")


def stop_token_ids(tokenizer: Any) -> set[int]:
    result: set[int] = set()
    eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        result.add(eos)
    elif eos:
        result.update(int(x) for x in eos)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        result.add(im_end)
    return result


@dataclass
class DecodeResult:
    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    first_token_logprobs: torch.Tensor


@torch.inference_mode()
def greedy_answer(
    model: Any,
    tokenizer: Any,
    cache: DynamicCache,
    q_ids: list[int],
    logical_next: int,
    max_new_tokens: int,
    prefix_physical_length: int,
) -> DecodeResult:
    stops = stop_token_ids(tokenizer)
    try:
        outputs = forward_block(model, cache, q_ids, logical_next)
        next_logits = outputs.logits[:, -1, :]
        first_token_logprobs = torch.log_softmax(next_logits.float(), dim=-1)[0].cpu()

        generated: list[int] = []
        generated_logprobs: list[float] = []
        answer_start = logical_next + len(q_ids)

        for step in range(max_new_tokens):
            log_probs = torch.log_softmax(next_logits.float(), dim=-1)
            next_id = int(torch.argmax(next_logits, dim=-1).item())
            generated_logprobs.append(float(log_probs[0, next_id].item()))

            if next_id in stops:
                break
            generated.append(next_id)

            outputs = forward_block(model, cache, [next_id], answer_start + step)
            next_logits = outputs.logits[:, -1, :]

        return DecodeResult(
            text=tokenizer.decode(generated, skip_special_tokens=True),
            token_ids=generated,
            token_logprobs=generated_logprobs,
            first_token_logprobs=first_token_logprobs,
        )
    finally:
        crop_cache(cache, prefix_physical_length)


def gold_surface(gold: dict[str, Any]) -> str:
    answer = gold["answer"]
    if isinstance(answer, dict):
        return json.dumps(answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(answer, bool):
        return "true" if answer else "false"
    return str(answer)


@torch.inference_mode()
def score_gold_sequence(
    model: Any,
    tokenizer: Any,
    cache: DynamicCache,
    q_ids: list[int],
    gold: dict[str, Any],
    logical_next: int,
    prefix_physical_length: int,
) -> dict[str, Any]:
    surface = gold_surface(gold)
    gold_ids = tokenizer.encode(surface, add_special_tokens=False)
    if not gold_ids:
        raise AssertionError("gold answer tokenized to an empty sequence")

    # logits at Q[-1] predict gold[0]; logits at gold[i-1] predict gold[i].
    teacher_input = q_ids + gold_ids[:-1]
    try:
        outputs = forward_block(model, cache, teacher_input, logical_next)
        first_logit_index = len(q_ids) - 1
        prediction_logits = outputs.logits[
            0, first_logit_index : first_logit_index + len(gold_ids), :
        ].float()
        targets = torch.tensor(gold_ids, dtype=torch.long, device=prediction_logits.device)
        selected = torch.log_softmax(prediction_logits, dim=-1).gather(
            1, targets.unsqueeze(1)
        )[:, 0]
        values = [float(x) for x in selected.cpu().tolist()]
        return {
            "surface": surface,
            "token_count": len(gold_ids),
            "token_logprobs": values,
            "sum_logprob": float(sum(values)),
            "mean_logprob": float(sum(values) / len(values)),
            "perplexity": float(math.exp(-sum(values) / len(values))),
        }
    finally:
        crop_cache(cache, prefix_physical_length)


def clean_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def compact(text: str) -> str:
    return re.sub(r"\s+", "", clean_answer(text)).strip("`\"'。.").lower()


def extract_json_object(text: str) -> Any | None:
    cleaned = clean_answer(text)
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


ACTION_TERMS = {
    "fix_configuration": (
        ("修复", "配置"),
        ("修正", "配置"),
        ("修复", "/etc/nginx"),
        ("fix", "config"),
    ),
    "repair_null_data": (
        ("修复", "空值"),
        ("处理", "空值"),
        ("清理", "空值"),
        ("repair", "null"),
    ),
    "fix_test_environment": (
        ("修复", "测试", "环境"),
        ("补充", "环境", "配置"),
        ("补全", "jwt_secret"),
        ("配置", "jwt_secret"),
        ("设置", "jwt_secret"),
        ("补充", "jwt_secret"),
        ("fix", "test", "environment"),
    ),
    "fix_health_check": (
        ("修复", "探针"),
        ("修正", "探针"),
        ("修复", "健康检查"),
        ("修正", "健康检查"),
        ("修改", "健康检查"),
        ("修改", "readiness"),
        ("修复", "readiness"),
        ("fix", "health"),
        ("fix", "readiness"),
    ),
}


def structured_match(text: str, expected: dict[str, Any]) -> bool:
    parsed = extract_json_object(text)
    if isinstance(parsed, dict) and all(parsed.get(k) == v for k, v in expected.items()):
        return True

    lowered = clean_answer(text).lower()
    false_expected = any(value is False for value in expected.values())
    negative = any(
        marker in lowered
        for marker in (
            "不能",
            "不可以",
            "不可",
            "不要",
            "不应",
            "不健康",
            "未健康",
            "未就绪",
            "尚未就绪",
            "失败",
            "否",
            "false",
            "not allowed",
        )
    )
    if false_expected and not negative:
        return False

    action = expected.get("first_action")
    groups = ACTION_TERMS.get(action, ())
    return bool(groups) and any(all(term in lowered for term in group) for group in groups)


def answer_matches(text: str, gold: dict[str, Any]) -> bool:
    kind = gold["type"]
    expected = gold["answer"]
    normalized = compact(text)

    if kind == "exact":
        return normalized == compact(str(expected))
    if kind == "binary":
        if expected == "否":
            return normalized in {"否", "不是", "不", "no", "false"}
        return normalized in {"是", "yes", "true"}
    if kind == "semantic_exact":
        candidates = [str(expected), *[str(x) for x in gold.get("aliases", [])]]
        if any(compact(candidate) in normalized for candidate in candidates):
            return True
        # Word order varies naturally in concise causal answers.  These checks
        # require all discriminative concepts instead of accepting a generic
        # keyword overlap.
        semantic_term_groups = {
            "jwt_secret": (
                ("jwt_secret", "缺失"),
                ("jwt_secret", "缺少"),
                ("jwt_secret", "未定义"),
                ("jwt_secret", "missing"),
            ),
            "not null": (
                ("null", "空值", "违反"),
                ("null", "existing", "violate"),
            ),
            "readiness": (
                ("readiness", "端口", "不一致"),
                ("探针", "端口", "不一致"),
                ("readiness", "port", "mismatch"),
            ),
            "分号": (
                ("分号", "缺少"),
                ("分号", "缺失"),
                ("semicolon", "missing"),
            ),
        }
        candidate_blob = " ".join(candidates).lower()
        for discriminator, groups in semantic_term_groups.items():
            if discriminator in candidate_blob:
                return any(all(term in normalized for term in group) for group in groups)
        return False
    if kind == "json_fields":
        parsed = extract_json_object(text)
        return isinstance(parsed, dict) and all(
            str(parsed.get(key)) == str(value) for key, value in expected.items()
        )
    if kind == "structured":
        return structured_match(text, expected)
    raise ValueError(f"unsupported gold type: {kind}")


def top_tokens(log_probs: torch.Tensor, tokenizer: Any, k: int = 10) -> list[dict[str, Any]]:
    values, indices = torch.topk(log_probs, k=k)
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "logprob": float(value),
        }
        for value, token_id in zip(values.tolist(), indices.tolist())
    ]


def kl_from_logprobs(p_log: torch.Tensor, q_log: torch.Tensor) -> float:
    # Both distributions are CPU float32 tensors over the full vocabulary.
    return float(torch.sum(torch.exp(p_log) * (p_log - q_log)).item())


def run_probes_for_condition(
    model: Any,
    tokenizer: Any,
    cache: DynamicCache,
    probes: list[dict[str, Any]],
    condition: str,
    logical_next: int,
    prefix_physical_length: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    records = []
    distributions: dict[str, torch.Tensor] = {}
    for probe in probes:
        probe_id = probe["probe_id"]
        q_ids = encode_and_validate(tokenizer, probe["Q"], f"{probe_id}.Q")
        gold_score = score_gold_sequence(
            model,
            tokenizer,
            cache,
            q_ids,
            probe["gold"],
            logical_next,
            prefix_physical_length,
        )
        decoded = greedy_answer(
            model,
            tokenizer,
            cache,
            q_ids,
            logical_next,
            max_new_tokens,
            prefix_physical_length,
        )
        distributions[probe_id] = decoded.first_token_logprobs
        records.append(
            {
                "probe_id": probe_id,
                "information_class": probe["information_class"],
                "gold": probe["gold"],
                "condition": condition,
                "answer": decoded.text,
                "answer_token_ids": decoded.token_ids,
                "answer_token_logprobs": decoded.token_logprobs,
                "match": answer_matches(decoded.text, probe["gold"]),
                "gold_score": gold_score,
                "first_token_top10": top_tokens(decoded.first_token_logprobs, tokenizer),
            }
        )
    return records, distributions


def release_cache(cache: DynamicCache | None) -> None:
    if cache is not None:
        del cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sample(
    model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_id = sample["sample_id"]
    S = encode_and_validate(tokenizer, sample["segments"]["S"], f"{sample_id}.S")
    T = encode_and_validate(tokenizer, sample["segments"]["T"], f"{sample_id}.T")
    R = encode_and_validate(tokenizer, sample["segments"]["R"], f"{sample_id}.R")
    s, t, r = len(S), len(T), len(R)
    logical_next = s + t + r

    if sample["layout"]["logical_next_position"] != logical_next:
        raise AssertionError(f"{sample_id}: dataset logical_next mismatch")

    result: dict[str, Any] = {
        "sample_id": sample_id,
        "template_type": sample["template_type"],
        "scenario_family": sample.get("scenario_family"),
        "token_lengths": {"S": s, "T": t, "R": r},
        "logical_next_position": logical_next,
        "conditions": {},
        "cache_invariants": {},
        "probe_divergences": {},
    }
    first_token_distributions: dict[str, dict[str, torch.Tensor]] = {}

    # FULL: S -> T -> R.  Keep compact hashes of retained spans for later checks.
    cache: DynamicCache | None = DynamicCache(config=model.config)
    prefill(model, cache, S, 0, args.chunk_size)
    prefill(model, cache, T, s, args.chunk_size)
    prefill(model, cache, R, s + t, args.chunk_size)
    assert_cache_length(cache, logical_next, f"{sample_id}.full")
    full_s_hashes = span_hashes(cache, 0, s)
    full_r_hashes = span_hashes(cache, s + t, logical_next)

    records, distributions = run_probes_for_condition(
        model,
        tokenizer,
        cache,
        sample["probes"],
        "full",
        logical_next,
        logical_next,
        args.max_new_tokens,
    )
    result["conditions"]["full"] = records
    first_token_distributions["full"] = distributions

    # EVICT: physically remove T without recomputing R.
    evict_t_in_place(cache, s, t, r)
    assert_cache_length(cache, s + r, f"{sample_id}.evict")
    evict_s_hashes = span_hashes(cache, 0, s)
    evict_r_hashes = span_hashes(cache, s, s + r)
    retained_exact = full_s_hashes == evict_s_hashes and full_r_hashes == evict_r_hashes
    if not retained_exact:
        raise AssertionError(f"{sample_id}: eviction changed retained S/R KV tensors")

    records, distributions = run_probes_for_condition(
        model,
        tokenizer,
        cache,
        sample["probes"],
        "evict",
        logical_next,
        s + r,
        args.max_new_tokens,
    )
    result["conditions"]["evict"] = records
    first_token_distributions["evict"] = distributions
    result["cache_invariants"]["evict_retained_kv_bitwise_equal"] = retained_exact
    result["cache_invariants"]["evict_physical_length"] = s + r

    release_cache(cache)
    cache = None

    # REPLAY: S -> R, with R retaining its original logical positions.
    replay_cache = DynamicCache(config=model.config)
    prefill(model, replay_cache, S, 0, args.chunk_size)
    prefill(model, replay_cache, R, s + t, args.chunk_size)
    assert_cache_length(replay_cache, s + r, f"{sample_id}.replay")
    replay_s_hashes = span_hashes(replay_cache, 0, s)
    replay_r_hashes = span_hashes(replay_cache, s, s + r)

    replay_s_exact = replay_s_hashes == full_s_hashes
    r_changed_layers = sum(
        replay_hash != full_hash
        for replay_hash, full_hash in zip(replay_r_hashes, full_r_hashes)
    )
    if not replay_s_exact:
        raise AssertionError(f"{sample_id}: replay changed S KV tensors")
    if r_changed_layers == 0:
        raise AssertionError(f"{sample_id}: replay R KV unexpectedly equals Full in every layer")

    records, distributions = run_probes_for_condition(
        model,
        tokenizer,
        replay_cache,
        sample["probes"],
        "replay",
        logical_next,
        s + r,
        args.max_new_tokens,
    )
    result["conditions"]["replay"] = records
    first_token_distributions["replay"] = distributions
    result["cache_invariants"].update(
        {
            "replay_s_kv_bitwise_equal_to_full": replay_s_exact,
            "replay_r_changed_layer_count": r_changed_layers,
            "num_layers": len(replay_cache.layers),
            "replay_physical_length": s + r,
            "all_logical_next_positions_equal": True,
        }
    )
    del replay_cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Distributional comparison for the first generated answer token.
    for probe in sample["probes"]:
        probe_id = probe["probe_id"]
        full_logp = first_token_distributions["full"][probe_id]
        evict_logp = first_token_distributions["evict"][probe_id]
        replay_logp = first_token_distributions["replay"][probe_id]
        full_evict = kl_from_logprobs(full_logp, evict_logp)
        full_replay = kl_from_logprobs(full_logp, replay_logp)
        result["probe_divergences"][probe_id] = {
            "metric_scope": "first_generated_answer_token",
            "kl_full_to_evict": full_evict,
            "kl_full_to_replay": full_replay,
            "inheritance_gain": full_replay - full_evict,
        }

    del first_token_distributions
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = Counter()
    correct = Counter()
    by_class_total = Counter()
    by_class_correct = Counter()
    gains: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        for condition, probes in row["conditions"].items():
            for probe in probes:
                total[condition] += 1
                by_class_total[(condition, probe["information_class"])] += 1
                if probe["match"]:
                    correct[condition] += 1
                    by_class_correct[(condition, probe["information_class"])] += 1
        for probe_id, metrics in row["probe_divergences"].items():
            gains[probe_id].append(metrics["inheritance_gain"])

    accuracy = {
        condition: correct[condition] / total[condition] if total[condition] else None
        for condition in CONDITIONS
    }
    accuracy_by_class: dict[str, dict[str, float]] = defaultdict(dict)
    for (condition, info_class), count in sorted(by_class_total.items()):
        accuracy_by_class[condition][info_class] = by_class_correct[(condition, info_class)] / count

    invariants = {
        "all_evict_retained_kv_bitwise_equal": all(
            row["cache_invariants"]["evict_retained_kv_bitwise_equal"] for row in rows
        ),
        "all_replay_s_kv_bitwise_equal_to_full": all(
            row["cache_invariants"]["replay_s_kv_bitwise_equal_to_full"] for row in rows
        ),
        "all_replay_r_changed": all(
            row["cache_invariants"]["replay_r_changed_layer_count"] > 0 for row in rows
        ),
    }
    return {
        "status": "PASS" if all(invariants.values()) else "FAIL",
        "num_samples": len(rows),
        "num_probe_runs": int(sum(total.values())),
        "accuracy": accuracy,
        "accuracy_by_information_class": dict(accuracy_by_class),
        "mean_first_token_inheritance_gain_by_probe": {
            probe_id: sum(values) / len(values) for probe_id, values in sorted(gains.items())
        },
        "cache_invariants": invariants,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "run_metadata.json"
    for path in (results_path, summary_path, metadata_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite")

    all_rows = read_jsonl(dataset_path)
    selected = choose_samples(all_rows, args)
    log(f"selected {len(selected)} of {len(all_rows)} samples")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    log(f"loading model={args.model} dtype={args.dtype} attn={args.attn_implementation}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    model.config.use_cache = True

    metadata = {
        "model": str(Path(args.model).resolve()),
        "dataset": str(dataset_path),
        "selected_sample_ids": [row["sample_id"] for row in selected],
        "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "chunk_size": args.chunk_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cuda_device": torch.cuda.get_device_name(torch.device(args.device))
        if str(args.device).startswith("cuda")
        else None,
        "started_unix": time.time(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    completed: list[dict[str, Any]] = []
    start = time.time()
    with results_path.open("w", encoding="utf-8") as output:
        for index, sample in enumerate(selected, 1):
            sample_start = time.time()
            log(
                f"[{index}/{len(selected)}] {sample['sample_id']} "
                f"T={sample['token_targets']['T']} R={sample['token_targets']['R']}"
            )
            row = run_sample(model, tokenizer, sample, args)
            row["elapsed_seconds"] = time.time() - sample_start
            completed.append(row)
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
            log(f"completed {sample['sample_id']} in {row['elapsed_seconds']:.2f}s")

    summary = aggregate(completed)
    summary["elapsed_seconds"] = time.time() - start
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FATAL {type(exc).__name__}: {exc}")
        raise
