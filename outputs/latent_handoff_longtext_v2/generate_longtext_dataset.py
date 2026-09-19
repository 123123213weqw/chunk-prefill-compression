#!/usr/bin/env python3
"""Generate the position-controlled long-context Latent Handoff Dataset v2.

The dataset is intentionally deterministic.  It contains four templates over a
full grid of T length, evidence position and random seed.  All serialized
segment lengths and hashes are computed with the model tokenizer actually used
by the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from transformers import AutoTokenizer


SCHEMA_VERSION = "latent-handoff-longtext/v2"
BASE_SEED = 20260902
T_TARGETS = (4096, 8192, 16384, 32768)
R_TARGET = 128
POSITION_TARGETS = {
    "begin": 0.05,
    "quarter": 0.25,
    "middle": 0.50,
    "three_quarter": 0.75,
    "end": 0.90,
}
SEED_INDICES = (0, 1, 2)
CHUNK_SIZE = 64


TEMPLATES = (
    "exact_endpoint",
    "exact_artifact",
    "math_inventory",
    "math_retry_budget",
)


def stable_seed(tag: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def token_hash(ids: list[int]) -> str:
    payload = ",".join(str(x) for x in ids).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def serialize_system_user(system: str, user: str, tool_call: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<tool_call>\n{tool_call}\n</tool_call><|im_end|>\n"
    )


def serialize_probe(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def segment_metadata(encode: Callable[[str], list[int]], text: str) -> dict[str, Any]:
    ids = encode(text)
    return {
        "text": text,
        "token_count": len(ids),
        "token_id_sha256": token_hash(ids),
    }


def fit_carrier(encode: Callable[[str], list[int]], content: str, target: int) -> str:
    prefix, suffix = "<|im_start|>assistant\n", "<|im_end|>\n"
    neutral = "\n审计记录保持有效，等待执行已登记的下一步动作。"
    body = content
    while len(encode(prefix + body + neutral + suffix)) <= target - 8:
        body += neutral
    for backtrack in range(8):
        candidate_body = body
        for _ in range(backtrack):
            if candidate_body.endswith(neutral):
                candidate_body = candidate_body[: -len(neutral)]
        for n in range(512):
            text = prefix + candidate_body + "\n附记:" + (" ok" * n) + suffix
            length = len(encode(text))
            if length == target:
                return text
            if length > target + 8:
                break
    raise RuntimeError(f"cannot fit R to {target} tokens")


def filler_line(sample_id: str, index: int) -> str:
    checksum = hashlib.sha256(f"{sample_id}:filler:{index}".encode()).hexdigest()[:16]
    components = ("scheduler", "telemetry", "storage-cache", "audit", "heartbeat")
    phases = ("scan", "collect", "normalize", "checkpoint")
    # Keep all filler numeric values above the range used by the math answers.
    metric = 100000 + ((stable_seed(f"{sample_id}:{index}:metric") % 800000))
    return (
        f"[info] record={index:05d} component={components[index % len(components)]} "
        f"phase={phases[index % len(phases)]} status=nominal metric_value={metric} "
        f"checksum={checksum}\n"
    )


def bridge_lines(sample_id: str, group: int, count: int = 3) -> str:
    lines = []
    for i in range(count):
        checksum = hashlib.sha256(f"{sample_id}:bridge:{group}:{i}".encode()).hexdigest()[:12]
        lines.append(
            f"[trace] correlation_group={group} step={i} unrelated_subsystem=telemetry "
            f"state=nominal checksum={checksum}\n"
        )
    return "".join(lines)


def build_evidence_bundle(sample_id: str, facts: list[str], candidate_status: str) -> str:
    pieces: list[str] = ["evidence_bundle_begin\n"]
    for index, fact in enumerate(facts):
        pieces.append(fact + "\n")
        if index != len(facts) - 1:
            # Three realistic trace records force adjacent required facts into
            # different 64-token chunks without adding another answer source.
            pieces.append(bridge_lines(sample_id, index))
    pieces.append(bridge_lines(sample_id, len(facts), count=2))
    pieces.append(f"candidate_status={candidate_status}\n")
    pieces.append("evidence_bundle_end\n")
    return "".join(pieces)


def fit_tool_at_position(
    encode: Callable[[str], list[int]],
    sample_id: str,
    target: int,
    evidence_bundle: str,
    position_fraction: float,
) -> str:
    prefix = (
        "<|im_start|>tool\n"
        f"audit_stream_id={sample_id}\n"
        "format_version=2\n"
        "records_are_chronological=true\n"
    )
    footer = "audit_stream_complete=true\nexit_code=0\n<|im_end|>\n"
    minimum_pad = "[padding]\n"

    sample_lines = "".join(filler_line(sample_id, i) for i in range(32))
    average_line_tokens = max(1.0, len(encode(sample_lines)) / 32.0)
    fixed_tokens = len(encode(prefix + evidence_bundle + footer + minimum_pad))
    line_count = max(0, int((target - fixed_tokens - 32) / average_line_tokens))
    desired_evidence_start = int(position_fraction * target)
    prefix_tokens = len(encode(prefix))
    before_count = max(0, int(round((desired_evidence_start - prefix_tokens) / average_line_tokens)))

    def construct(n_lines: int, pad: str) -> str:
        before = min(before_count, n_lines)
        lines = [filler_line(sample_id, i) for i in range(n_lines)]
        return prefix + "".join(lines[:before]) + evidence_bundle + "".join(lines[before:]) + pad + footer

    # Bring the deterministic filler count to the largest value that leaves
    # room for a padding record. Usually this adjusts the estimate by <3 lines.
    while line_count > 0 and len(encode(construct(line_count, minimum_pad))) > target:
        line_count -= 1
    while len(encode(construct(line_count + 1, minimum_pad))) <= target:
        line_count += 1

    # Boundary-aware exact fitting. Backtracking handles rare BPE merges at the
    # final filler/padding boundary.
    for backtrack in range(min(12, line_count) + 1):
        active = line_count - backtrack
        base_gap = target - len(encode(construct(active, "[padding]\n")))
        lo, hi = max(0, base_gap - 48), max(96, base_gap + 48)
        for n in range(lo, hi + 1):
            text = construct(active, "[padding]" + (" ok" * n) + "\n")
            length = len(encode(text))
            if length == target:
                return text
    raise RuntimeError(f"cannot fit T={target} for {sample_id}")


def find_all(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    return [
        i
        for i in range(len(haystack) - len(needle) + 1)
        if haystack[i : i + len(needle)] == needle
    ]


def locate_evidence(
    encode: Callable[[str], list[int]], t_text: str, evidence: list[str]
) -> dict[str, Any]:
    t_ids = encode(t_text)
    records: list[dict[str, Any]] = []
    all_chunks: set[int] = set()
    for item in evidence:
        spans: set[tuple[int, int]] = set()
        for variant in (item, " " + item, "\n" + item):
            ids = encode(variant)
            for start in find_all(t_ids, ids):
                spans.add((start, start + len(ids)))
        if not spans:
            raise RuntimeError(f"token span not found for evidence: {item}")
        # Variants can produce overlapping matches. Keep the shortest earliest
        # span, which is sufficient for chunk membership and oracle auditing.
        start, end = sorted(spans, key=lambda pair: (pair[1] - pair[0], pair[0]))[0]
        chunk_ids = list(range(start // CHUNK_SIZE, (end - 1) // CHUNK_SIZE + 1))
        all_chunks.update(chunk_ids)
        records.append(
            {
                "text": item,
                "token_span": [start, end],
                "chunk_ids_b064": chunk_ids,
                "normalized_start": round(start / len(t_ids), 8),
            }
        )
    all_chunk_ids = set(range(math.ceil(len(t_ids) / CHUNK_SIZE)))
    return {
        "items": records,
        "evidence_chunk_ids_b064": sorted(all_chunks),
        "distractor_chunk_ids_b064": sorted(all_chunk_ids - all_chunks),
        "anchor_normalized_position": records[0]["normalized_start"],
    }


def add_layout(sample: dict[str, Any]) -> None:
    ns = sample["segments"]["S"]["token_count"]
    nt = sample["segments"]["T"]["token_count"]
    nr = sample["segments"]["R"]["token_count"]
    sample["layout"] = {
        "S": [0, ns],
        "T": [ns, ns + nt],
        "R": [ns + nt, ns + nt + nr],
        "full_prefix_tokens": ns + nt + nr,
        "physical_tokens_after_eviction": ns + nr,
        "logical_next_position": ns + nt + nr,
        "retained_kv_ratio": round((ns + nr) / (ns + nt + nr), 8),
        "canonical_construction": "encode each serialized segment with add_special_tokens=false, then concatenate token IDs",
    }


def template_state(template: str, rng: random.Random, seed_index: int) -> tuple[dict[str, Any], list[str], dict[str, Any], str]:
    if template == "exact_endpoint":
        ip = f"10.{rng.randint(20, 239)}.{rng.randint(20, 239)}.{rng.randint(20, 239)}"
        port = rng.randint(20000, 60999)
        state = {"server_ip": ip, "ssh_port": port}
        facts = [f"server_ip={ip}", f"ssh_port={port}"]
        gold = {"type": "exact", "answer": f"{ip}:{port}"}
        question = "审计日志中当前生效的服务器 IP 和 SSH 端口是什么？只输出 `IP:端口`。"
    elif template == "exact_artifact":
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        request_id = "REQ-" + "".join(rng.choice(alphabet) for _ in range(14))
        regions = ("cn-east-3", "cn-north-6", "ap-southeast-4", "eu-central-2", "us-west-5")
        region = regions[rng.randrange(len(regions))]
        state = {"request_id": request_id, "deployment_region": region}
        facts = [f"request_id={request_id}", f"deployment_region={region}"]
        gold = {
            "type": "json_fields",
            "answer": {"request_id": request_id, "deployment_region": region},
        }
        question = "审计日志中当前 request_id 和 deployment_region 是什么？只输出 JSON。"
    elif template == "math_inventory":
        shards = rng.randint(7, 19)
        per_shard = rng.randint(101, 499)
        rejected = rng.randint(3, min(79, shards * per_shard - 1))
        accepted = shards * per_shard - rejected
        state = {
            "shard_count": shards,
            "items_per_shard": per_shard,
            "rejected_items": rejected,
            "accepted_items": accepted,
        }
        facts = [
            f"shard_count={shards}",
            f"items_per_shard={per_shard}",
            f"rejected_items={rejected}",
        ]
        gold = {"type": "exact", "answer": str(accepted)}
        question = "按 shard_count × items_per_shard − rejected_items 计算有效条目总数。只输出整数。"
    elif template == "math_retry_budget":
        base = rng.randint(120, 850)
        retries = rng.randint(2, 9)
        penalty = rng.randint(11, 73)
        discount = rng.randint(5, 49)
        total = base + retries * penalty - discount
        state = {
            "base_latency_ms": base,
            "retry_count": retries,
            "retry_penalty_ms": penalty,
            "parallel_discount_ms": discount,
            "effective_latency_ms": total,
        }
        facts = [
            f"base_latency_ms={base}",
            f"retry_count={retries}",
            f"retry_penalty_ms={penalty}",
            f"parallel_discount_ms={discount}",
        ]
        gold = {"type": "exact", "answer": str(total)}
        question = "按 base_latency_ms + retry_count × retry_penalty_ms − parallel_discount_ms 计算有效延迟。只输出整数。"
    else:
        raise ValueError(template)
    return state, facts, gold, question


def derivation_program(template: str) -> dict[str, Any]:
    if template == "exact_endpoint":
        return {"op": "format", "expression": "server_ip + ':' + str(ssh_port)"}
    if template == "exact_artifact":
        return {"op": "project_json", "fields": ["request_id", "deployment_region"]}
    if template == "math_inventory":
        return {"op": "integer_arithmetic", "expression": "shard_count * items_per_shard - rejected_items"}
    if template == "math_retry_budget":
        return {"op": "integer_arithmetic", "expression": "base_latency_ms + retry_count * retry_penalty_ms - parallel_discount_ms"}
    raise ValueError(template)


def recompute_primary_answer(template: str, state: dict[str, Any]) -> Any:
    if template == "exact_endpoint":
        return f"{state['server_ip']}:{state['ssh_port']}"
    if template == "exact_artifact":
        return {"request_id": state["request_id"], "deployment_region": state["deployment_region"]}
    if template == "math_inventory":
        return str(state["shard_count"] * state["items_per_shard"] - state["rejected_items"])
    if template == "math_retry_budget":
        return str(
            state["base_latency_ms"]
            + state["retry_count"] * state["retry_penalty_ms"]
            - state["parallel_discount_ms"]
        )
    raise ValueError(template)


def make_sample(
    encode: Callable[[str], list[int]],
    template: str,
    template_index: int,
    t_target: int,
    length_index: int,
    position_label: str,
    position_index: int,
    seed_index: int,
) -> dict[str, Any]:
    sid = f"long-{template}-t{t_target:05d}-p{position_label}-s{seed_index}"
    rng = random.Random(stable_seed(sid))
    state, primary_facts, primary_gold, primary_question = template_state(template, rng, seed_index)

    candidate_approved = (template_index + length_index + position_index + seed_index) % 2 == 0
    candidate_status = "approved" if candidate_approved else "rejected"
    state["candidate_approved"] = candidate_approved
    state["next_action"] = "verify-checkpoint"

    evidence_bundle = build_evidence_bundle(sid, primary_facts, candidate_status)
    t_text = fit_tool_at_position(
        encode, sid, t_target, evidence_bundle, POSITION_TARGETS[position_label]
    )
    r_text = fit_carrier(
        encode,
        "长日志已完成审计。下一步动作标识为 verify-checkpoint；完成验证前不要提交。",
        R_TARGET,
    )
    system = "你是严格的长日志审计助手。只能依据当前上下文中的权威记录回答，不得使用常识猜测数值。"
    user = "读取完整审计日志，记录当前事实、计算所需状态以及候选状态；回复只概括审计已完成，不复述具体值。"
    call = json.dumps(
        {"name": "read_audit_stream", "arguments": {"stream_id": sid}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    s_text = serialize_system_user(system, user, call)

    primary_class = "T_only_exact" if template.startswith("exact_") else "T_only_derived"
    all_t_evidence = primary_facts + [f"candidate_status={candidate_status}"]
    probes = [
        {
            "probe_id": "primary",
            "Q": segment_metadata(encode, serialize_probe(primary_question)),
            "gold": primary_gold,
            "evidence": primary_facts,
            "information_class": primary_class,
        },
        {
            "probe_id": "counterfactual-status",
            "Q": segment_metadata(
                encode,
                serialize_probe("日志中的备用候选是否已获批准，可用于正式执行？只回答是或否。"),
            ),
            "gold": {"type": "binary", "answer": "是" if candidate_approved else "否"},
            "evidence": [f"candidate_status={candidate_status}"],
            "information_class": "T_negative_evidence",
        },
        {
            "probe_id": "explicit-control",
            "Q": segment_metadata(
                encode,
                serialize_probe("审计完成后登记的下一步动作标识是什么？只输出动作标识。"),
            ),
            "gold": {"type": "exact", "answer": "verify-checkpoint"},
            "evidence": ["下一步动作标识为 verify-checkpoint"],
            "information_class": "R_explicit_control",
        },
    ]

    evidence_map = locate_evidence(encode, t_text, all_t_evidence)
    primary_map = locate_evidence(encode, t_text, primary_facts)
    split = "development" if seed_index == 0 else "test"
    sample: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sid,
        "split": split,
        "template_type": template,
        "task_family": "exact_retrieval" if template.startswith("exact_") else "cross_chunk_math",
        "seed": stable_seed(sid),
        "seed_index": seed_index,
        "token_targets": {"T": t_target, "R": R_TARGET},
        "evidence_position": {
            "label": position_label,
            "target_fraction": POSITION_TARGETS[position_label],
            "actual_anchor_fraction": evidence_map["anchor_normalized_position"],
        },
        "chunking": {
            "chunk_size": CHUNK_SIZE,
            "num_t_chunks": math.ceil(t_target / CHUNK_SIZE),
            "primary_evidence_chunk_ids": primary_map["evidence_chunk_ids_b064"],
            "all_evidence_chunk_ids": evidence_map["evidence_chunk_ids_b064"],
            "distractor_chunk_ids": evidence_map["distractor_chunk_ids_b064"],
        },
        "evidence_spans": evidence_map["items"],
        "required_facts": {key: value for key, value in state.items() if key not in {"next_action"}},
        "derivation_program": derivation_program(template),
        "counterfactual": {
            "group_id": f"{template}-t{t_target}-p{position_label}",
            "candidate_approved": candidate_approved,
        },
        "segments": {
            "S": segment_metadata(encode, s_text),
            "T": segment_metadata(encode, t_text),
            "R": segment_metadata(encode, r_text),
        },
        "latent_state": state,
        "forbidden_in_R": [str(value) for key, value in state.items() if key != "next_action"],
        "probes": probes,
        "cache_conditions": ["full", "chunked_full", "drop_all_t", "compressed"],
    }
    add_layout(sample)
    return sample


def validate(rows: list[dict[str, Any]], encode: Callable[[str], list[int]]) -> dict[str, Any]:
    errors: list[str] = []
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("sample IDs are not unique")

    grid = Counter()
    templates = Counter()
    splits = Counter()
    classes = Counter()
    counterfactuals = Counter()
    position_errors: list[float] = []

    for row in rows:
        sid = row["sample_id"]
        template = row["template_type"]
        t_target = row["token_targets"]["T"]
        position = row["evidence_position"]["label"]
        seed_index = row["seed_index"]
        grid[(template, t_target, position, seed_index)] += 1
        templates[template] += 1
        splits[row["split"]] += 1
        counterfactuals[str(row["counterfactual"]["candidate_approved"])] += 1

        for segment_name in ("S", "T", "R"):
            segment = row["segments"][segment_name]
            actual = encode(segment["text"])
            if len(actual) != segment["token_count"]:
                errors.append(f"{sid}:{segment_name}: token count mismatch")
            if token_hash(actual) != segment["token_id_sha256"]:
                errors.append(f"{sid}:{segment_name}: token hash mismatch")
        if row["segments"]["T"]["token_count"] != t_target:
            errors.append(f"{sid}: T target mismatch")
        if row["segments"]["R"]["token_count"] != R_TARGET:
            errors.append(f"{sid}: R target mismatch")

        t_text = row["segments"]["T"]["text"]
        r_text = row["segments"]["R"]["text"]
        for forbidden in row["forbidden_in_R"]:
            # Ignore one-character and boolean strings; they are too generic to
            # constitute answer leakage.
            if len(forbidden) >= 2 and forbidden.lower() in r_text.lower():
                errors.append(f"{sid}: forbidden value leaked into R: {forbidden}")
        for probe in row["probes"]:
            classes[probe["information_class"]] += 1
            q = probe["Q"]
            q_ids = encode(q["text"])
            if len(q_ids) != q["token_count"] or token_hash(q_ids) != q["token_id_sha256"]:
                errors.append(f"{sid}:{probe['probe_id']}: Q metadata mismatch")
            evidence_text = r_text if probe["information_class"] == "R_explicit_control" else t_text
            for item in probe["evidence"]:
                if evidence_text.count(item) != 1:
                    errors.append(f"{sid}:{probe['probe_id']}: evidence count != 1: {item}")

        recomputed = recompute_primary_answer(template, row["latent_state"])
        stored = row["probes"][0]["gold"]["answer"]
        if recomputed != stored:
            errors.append(f"{sid}: primary gold failed executable recomputation")
        expected_binary = "是" if row["latent_state"]["candidate_approved"] else "否"
        if row["probes"][1]["gold"]["answer"] != expected_binary:
            errors.append(f"{sid}: counterfactual gold mismatch")

        position_error = abs(
            row["evidence_position"]["actual_anchor_fraction"]
            - row["evidence_position"]["target_fraction"]
        )
        position_errors.append(position_error)
        if position_error > 0.035:
            errors.append(f"{sid}: evidence position error {position_error:.5f} > 0.035")
        if template.startswith("math_") and len(row["chunking"]["primary_evidence_chunk_ids"]) < 2:
            errors.append(f"{sid}: math evidence does not cross chunks")

        ns = row["segments"]["S"]["token_count"]
        nt = row["segments"]["T"]["token_count"]
        nr = row["segments"]["R"]["token_count"]
        if row["layout"]["logical_next_position"] != ns + nt + nr:
            errors.append(f"{sid}: invalid logical position")
        if row["layout"]["physical_tokens_after_eviction"] != ns + nr:
            errors.append(f"{sid}: invalid physical cache length")

    for template in TEMPLATES:
        for target in T_TARGETS:
            for position in POSITION_TARGETS:
                for seed_index in SEED_INDICES:
                    if grid[(template, target, position, seed_index)] != 1:
                        errors.append(f"grid mismatch: {template} T={target} p={position} s={seed_index}")

    if counterfactuals["True"] != counterfactuals["False"]:
        errors.append("counterfactual yes/no classes are not balanced")

    return {
        "status": "PASS" if not errors else "FAIL",
        "num_samples": len(rows),
        "num_probes": sum(classes.values()),
        "template_counts": dict(sorted(templates.items())),
        "split_counts": dict(sorted(splits.items())),
        "probe_class_counts": dict(sorted(classes.items())),
        "counterfactual_counts": dict(sorted(counterfactuals.items())),
        "grid_cells": len(grid),
        "max_evidence_position_absolute_error": max(position_errors, default=0.0),
        "checks": {
            "unique_sample_ids": len(ids) == len(set(ids)),
            "full_factorial_grid": not any("grid mismatch" in error for error in errors),
            "exact_token_budgets": not any("target mismatch" in error for error in errors),
            "token_hashes_recomputed": not any("token" in error and "mismatch" in error for error in errors),
            "gold_recomputed": not any("gold" in error for error in errors),
            "evidence_unique": not any("evidence count" in error for error in errors),
            "evidence_position_controlled": not any("position error" in error for error in errors),
            "math_evidence_crosses_chunks": not any("does not cross chunks" in error for error in errors),
            "R_answer_leakage_absent": not any("leaked into R" in error for error in errors),
            "counterfactual_balanced": counterfactuals["True"] == counterfactuals["False"],
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local tokenizer/model directory")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)

    rows: list[dict[str, Any]] = []
    for template_index, template in enumerate(TEMPLATES):
        for length_index, target in enumerate(T_TARGETS):
            for position_index, position in enumerate(POSITION_TARGETS):
                for seed_index in SEED_INDICES:
                    rows.append(
                        make_sample(
                            encode,
                            template,
                            template_index,
                            target,
                            length_index,
                            position,
                            position_index,
                            seed_index,
                        )
                    )

    report = validate(rows, encode)
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))

    artifacts: list[Path] = []
    combined = output / "longtext_v2.jsonl"
    dump_jsonl(combined, rows)
    artifacts.append(combined)
    for template in TEMPLATES:
        path = output / f"{template}_v2.jsonl"
        dump_jsonl(path, [row for row in rows if row["template_type"] == template])
        artifacts.append(path)
    dev = output / "longtext_v2_development.jsonl"
    test = output / "longtext_v2_test.jsonl"
    smoke = output / "longtext_v2_smoke16.jsonl"
    dump_jsonl(dev, [row for row in rows if row["split"] == "development"])
    dump_jsonl(test, [row for row in rows if row["split"] == "test"])
    dump_jsonl(
        smoke,
        [
            row
            for row in rows
            if row["seed_index"] == 0 and row["evidence_position"]["label"] == "middle"
        ],
    )
    artifacts.extend((dev, test, smoke))

    report["tokenizer"] = {
        "source": str(Path(args.model).resolve()),
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "encoding": "add_special_tokens=false per serialized segment",
    }
    report["configuration"] = {
        "T_targets": list(T_TARGETS),
        "R_target": R_TARGET,
        "position_targets": POSITION_TARGETS,
        "seed_indices": list(SEED_INDICES),
        "chunk_size": CHUNK_SIZE,
    }
    report["artifacts"] = {
        path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
        for path in artifacts
    }
    report_path = output / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
