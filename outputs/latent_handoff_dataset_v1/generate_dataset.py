#!/usr/bin/env python3
"""Generate and validate Latent Handoff Dataset v1.

The generator is deterministic. It uses the Qwen tokenizer to make the serialized
tool block T and carrier block R exactly match their requested token budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from transformers import AutoTokenizer


TOOL_TARGETS = (512, 2048, 8192, 16384)
CARRIER_TARGETS = (32, 128, 512)
VARIANTS = 10
BASE_SEED = 20260901


def stable_seed(tag: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def token_hash(ids: list[int]) -> str:
    payload = ",".join(str(x) for x in ids).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def serialize_system_user(system: str, user: str, tool_call: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<tool_call>\n{tool_call}\n</tool_call><|im_end|>\n"
    )


def serialize_tool(content: str) -> str:
    return f"<|im_start|>tool\n{content}<|im_end|>\n"


def serialize_assistant(content: str) -> str:
    return f"<|im_start|>assistant\n{content}<|im_end|>\n"


def serialize_probe(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def fit_with_logs(
    encode: Callable[[str], list[int]],
    prefix: str,
    suffix: str,
    target: int,
    filler_line: Callable[[int], str],
) -> str:
    """Fit a serialized tool block to an exact token count."""
    lines: list[str] = []
    i = 0
    # Leave room for the final deterministic padding record.
    while True:
        line = filler_line(i)
        candidate = prefix + "".join(lines) + line + suffix
        if len(encode(candidate)) > target - 24:
            break
        lines.append(line)
        i += 1

    # Boundary-aware brute force. Backtracking handles rare BPE boundary merges.
    for backtrack in range(min(8, len(lines)) + 1):
        active = lines[: len(lines) - backtrack] if backtrack else lines
        body = prefix + "".join(active)
        for n in range(0, 512):
            pad = "[padding]" + (" ok" * n) + "\n"
            candidate = body + pad + suffix
            length = len(encode(candidate))
            if length == target:
                return candidate
            if length > target + 8:
                break
    raise RuntimeError(f"Unable to fit tool segment to {target} tokens")


def fit_carrier(
    encode: Callable[[str], list[int]], base_content: str, target: int
) -> str:
    """Fit an assistant carrier block to an exact token count."""
    wrapper_prefix = "<|im_start|>assistant\n"
    wrapper_suffix = "<|im_end|>\n"
    neutral = "\n状态记录：上下文分析已经完成，当前结论保持不变。"
    content = base_content
    while len(encode(wrapper_prefix + content + neutral + wrapper_suffix)) <= target - 8:
        content += neutral

    for backtrack in range(0, 8):
        current = content
        for _ in range(backtrack):
            if current.endswith(neutral):
                current = current[: -len(neutral)]
        for n in range(0, 256):
            pad = "\n附记:" + (" ok" * n)
            candidate = wrapper_prefix + current + pad + wrapper_suffix
            length = len(encode(candidate))
            if length == target:
                return candidate
            if length > target + 8:
                break
    raise RuntimeError(f"Unable to fit carrier segment to {target} tokens")


def segment_metadata(encode: Callable[[str], list[int]], text: str) -> dict[str, Any]:
    ids = encode(text)
    return {
        "text": text,
        "token_count": len(ids),
        "token_id_sha256": token_hash(ids),
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


def exact_fact_sample(
    encode: Callable[[str], list[int]], tool_target: int, carrier_target: int, variant: int
) -> dict[str, Any]:
    sid = f"exact-t{tool_target:05d}-r{carrier_target:03d}-v{variant:02d}"
    rng = random.Random(stable_seed(sid))
    ip = f"10.{rng.randint(11, 239)}.{rng.randint(11, 239)}.{rng.randint(11, 239)}"
    port = str(rng.randint(20000, 60999))
    request_id = "REQ-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(12))
    regions = ("cn-east-3", "cn-north-6", "ap-southeast-4", "eu-central-2", "us-west-5")
    region = regions[rng.randrange(len(regions))]

    system = "你是服务器运维助手。必须依据工具结果更新任务状态。"
    user = "连接预先登记的目标服务器，确认状态并给出下一步建议。回复中不要复述连接参数。"
    tool_call = '{"name":"ssh_exec","arguments":{"host_alias":"primary-nginx"}}'
    s_text = serialize_system_user(system, user, tool_call)

    critical = (
        "tool_name=ssh_exec\n"
        "ssh connection established\n"
        f"server_ip={ip}\n"
        f"ssh_port={port}\n"
        f"request_id={request_id}\n"
        f"deployment_region={region}\n"
        "service=nginx\n"
        "status=connected\n"
    )
    suffix = "exit_code=0\n"

    def filler(i: int) -> str:
        checksum = hashlib.sha256(f"{sid}:log:{i}".encode()).hexdigest()[:16]
        phases = ("handshake", "package-scan", "service-scan", "permission-check")
        return f"[info] event={i:05d} phase={phases[i % len(phases)]} status=ok checksum={checksum}\n"

    t_text = fit_with_logs(
        encode,
        "<|im_start|>tool\n" + critical,
        "" + suffix + "<|im_end|>\n",
        tool_target,
        filler,
    )
    r_text = fit_carrier(encode, "连接成功。下一步检查 nginx 配置。", carrier_target)

    probes_raw = [
        {
            "probe_id": "exact-ip-port",
            "question": "刚才连接成功的服务器 IP 和 SSH 端口分别是什么？只输出实际值，不要输出字段名；格式示例：`192.0.2.1:22`。",
            "gold": {"type": "exact", "answer": f"{ip}:{port}"},
            "evidence": [f"server_ip={ip}", f"ssh_port={port}"],
            "information_class": "T_only_exact",
        },
        {
            "probe_id": "exact-request-region",
            "question": "刚才工具结果中的 request_id 和 deployment_region 是什么？只输出 JSON。",
            "gold": {
                "type": "json_fields",
                "answer": {"request_id": request_id, "deployment_region": region},
            },
            "evidence": [f"request_id={request_id}", f"deployment_region={region}"],
            "information_class": "T_only_exact",
        },
        {
            "probe_id": "explicit-control",
            "question": "接下来准备检查哪个服务？只输出服务名。",
            "gold": {"type": "exact", "answer": "nginx"},
            "evidence": ["下一步检查 nginx 配置"],
            "information_class": "R_explicit_control",
        },
    ]
    probes = []
    for p in probes_raw:
        q_text = serialize_probe(p.pop("question"))
        probes.append({**p, "Q": segment_metadata(encode, q_text)})

    sample = {
        "schema_version": "latent-handoff/v1",
        "sample_id": sid,
        "template_type": "exact_fact",
        "seed": stable_seed(sid),
        "token_targets": {"T": tool_target, "R": carrier_target},
        "segments": {
            "S": segment_metadata(encode, s_text),
            "T": segment_metadata(encode, t_text),
            "R": segment_metadata(encode, r_text),
        },
        "latent_state": {
            "server_ip": ip,
            "ssh_port": port,
            "request_id": request_id,
            "deployment_region": region,
            "service": "nginx",
            "connection_status": "connected",
        },
        "forbidden_in_R": [ip, port, request_id, region],
        "probes": probes,
        "cache_conditions": ["full", "evict", "replay"],
    }
    add_layout(sample)
    return sample


DERIVED_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "name": "nginx_configuration_failure",
        "critical": lambda v: (
            f"[error] nginx configuration test failed file=/etc/nginx/conf.d/site-{v}.conf\n"
            f"[location] line={70 + v}\n"
            "[parser] unexpected end of file, expecting semicolon\n"
            "[certificate] status=loaded\n"
            "[upstream_network] health=passed\n"
            f"[rollback] active_revision=nginx-r{300 + v}\n"
            "command=nginx -t\n"
        ),
        "state": lambda v: {
            "root_cause": "missing_semicolon",
            "detail_answer": f"/etc/nginx/conf.d/site-{v}.conf:{70 + v}",
            "active_revision": f"nginx-r{300 + v}",
            "failed_stage": "configuration_validation",
        },
        "carrier": "验证失败，旧版本仍运行。先修复配置并复验，成功前不要 reload。",
        "probes": lambda v: [
            ("derived-root-cause", "配置验证失败的直接根因是什么？只输出根因。", {"type": "semantic_exact", "answer": "缺少分号", "aliases": ["missing semicolon"]}, ["expecting semicolon"], "T_only_derived"),
            ("exact-evidence-location", "错误位于哪个配置文件的哪一行？只输出实际值，不要输出字段名；格式示例：`/etc/nginx/conf.d/example.conf:42`。", {"type": "exact", "answer": f"/etc/nginx/conf.d/site-{v}.conf:{70 + v}"}, [f"file=/etc/nginx/conf.d/site-{v}.conf", f"line={70 + v}"], "T_only_exact"),
            ("workflow-state", "现在是否可以 reload nginx？下一步首先做什么？", {"type": "structured", "answer": {"reload_allowed": False, "first_action": "fix_configuration"}}, ["成功前不要 reload", "先修复配置"], "R_explicit_control"),
            ("negative-causal-control", "证书加载或 upstream 网络是否是本次失败根因？只回答是或否。", {"type": "binary", "answer": "否"}, ["[certificate] status=loaded", "[upstream_network] health=passed"], "T_negative_evidence"),
        ],
        "forbidden": lambda v: ["missing_semicolon", "缺少分号", f"site-{v}.conf", str(70 + v), "certificate", "upstream"],
    },
    {
        "name": "database_migration_failure",
        "critical": lambda v: (
            f"[migration] file=db/migrations/20260901_{v:02d}_users_email.sql\n"
            f"[validation] null_rows={13 + v}\n"
            "[constraint] ALTER COLUMN email SET NOT NULL rejected\n"
            "[database_connection] health=passed\n"
            "[backup] status=completed\n"
            f"[rollback] active_schema_version={180 + v}\n"
            "command=apply_migration\n"
        ),
        "state": lambda v: {
            "root_cause": "existing_nulls_violate_not_null",
            "detail_answer": f"db/migrations/20260901_{v:02d}_users_email.sql:{13 + v}",
            "active_revision": str(180 + v),
            "failed_stage": "migration_validation",
        },
        "carrier": "迁移未完成，旧 schema 仍生效。先修复数据再重跑迁移，成功前不要切换应用。",
        "probes": lambda v: [
            ("derived-root-cause", "数据库迁移失败的直接根因是什么？", {"type": "semantic_exact", "answer": "现有空值违反 NOT NULL 约束", "aliases": ["existing nulls violate NOT NULL"]}, ["null_rows=", "SET NOT NULL rejected"], "T_only_derived"),
            ("exact-evidence-location", "迁移文件是什么，有多少条空值记录？只输出实际值，不要输出字段名；格式示例：`db/migrations/example.sql:3`。", {"type": "exact", "answer": f"db/migrations/20260901_{v:02d}_users_email.sql:{13 + v}"}, [f"file=db/migrations/20260901_{v:02d}_users_email.sql", f"null_rows={13 + v}"], "T_only_exact"),
            ("workflow-state", "现在是否可以切换应用？下一步首先做什么？", {"type": "structured", "answer": {"switch_allowed": False, "first_action": "repair_null_data"}}, ["成功前不要切换应用", "先修复数据"], "R_explicit_control"),
            ("negative-causal-control", "数据库连接或备份是否是本次失败根因？只回答是或否。", {"type": "binary", "answer": "否"}, ["[database_connection] health=passed", "[backup] status=completed"], "T_negative_evidence"),
        ],
        "forbidden": lambda v: ["NOT NULL", "空值", f"20260901_{v:02d}", str(13 + v), "database_connection", "backup"],
    },
    {
        "name": "python_test_failure",
        "critical": lambda v: (
            f"[pytest] file=tests/test_auth_{v}.py\n"
            f"[pytest] case=test_refresh_token_{v}\n"
            "[exception] KeyError: JWT_SECRET\n"
            "[environment] required variable JWT_SECRET is absent\n"
            "[database_fixture] status=passed\n"
            "[network_mock] status=passed\n"
            "command=pytest -q\n"
        ),
        "state": lambda v: {
            "root_cause": "missing_JWT_SECRET_environment_variable",
            "detail_answer": f"tests/test_auth_{v}.py:test_refresh_token_{v}",
            "active_revision": "working_tree_unchanged",
            "failed_stage": "test_execution",
        },
        "carrier": "测试未通过，当前改动不能合并。先修复测试环境配置并重跑测试，通过后再继续。",
        "probes": lambda v: [
            ("derived-root-cause", "测试失败的直接根因是什么？", {"type": "semantic_exact", "answer": "缺少 JWT_SECRET 环境变量", "aliases": ["missing JWT_SECRET environment variable"]}, ["KeyError: JWT_SECRET", "JWT_SECRET is absent"], "T_only_derived"),
            ("exact-evidence-location", "失败的测试文件和测试用例是什么？只输出实际值，不要输出字段名；格式示例：`tests/test_example.py:test_case`。", {"type": "exact", "answer": f"tests/test_auth_{v}.py:test_refresh_token_{v}"}, [f"file=tests/test_auth_{v}.py", f"case=test_refresh_token_{v}"], "T_only_exact"),
            ("workflow-state", "现在是否可以合并代码？下一步首先做什么？", {"type": "structured", "answer": {"merge_allowed": False, "first_action": "fix_test_environment"}}, ["不能合并", "先修复测试环境配置"], "R_explicit_control"),
            ("negative-causal-control", "数据库 fixture 或网络 mock 是否是本次失败根因？只回答是或否。", {"type": "binary", "answer": "否"}, ["[database_fixture] status=passed", "[network_mock] status=passed"], "T_negative_evidence"),
        ],
        "forbidden": lambda v: ["JWT_SECRET", f"test_auth_{v}.py", f"test_refresh_token_{v}", "database_fixture", "network_mock"],
    },
    {
        "name": "kubernetes_rollout_failure",
        "critical": lambda v: (
            f"[deployment] name=payments-api-{v}\n"
            f"[container] listening_port={8080 + v}\n"
            f"[readiness_probe] configured_port={8180 + v}\n"
            "[readiness_probe] result=connection_refused\n"
            "[image_pull] status=completed\n"
            "[configmap_mount] status=completed\n"
            f"[rollback] active_revision=payments-r{90 + v}\n"
            "command=kubectl rollout status\n"
        ),
        "state": lambda v: {
            "root_cause": "readiness_probe_port_mismatch",
            "detail_answer": f"{8180 + v}:{8080 + v}",
            "active_revision": f"payments-r{90 + v}",
            "failed_stage": "readiness_validation",
        },
        "carrier": "新版本尚未就绪，旧版本仍在服务。先修复健康检查配置再重新 rollout。",
        "probes": lambda v: [
            ("derived-root-cause", "rollout 失败的直接根因是什么？", {"type": "semantic_exact", "answer": "readiness probe 端口与容器监听端口不一致", "aliases": ["readiness probe port mismatch"]}, ["configured_port=", "listening_port=", "connection_refused"], "T_only_derived"),
            ("exact-evidence-location", "探针配置端口和容器监听端口分别是什么？只输出两个实际端口，不要输出字段名；格式示例：`8080:8081`。", {"type": "exact", "answer": f"{8180 + v}:{8080 + v}"}, [f"configured_port={8180 + v}", f"listening_port={8080 + v}"], "T_only_exact"),
            ("workflow-state", "新版本现在是否健康？下一步首先做什么？", {"type": "structured", "answer": {"new_revision_healthy": False, "first_action": "fix_health_check"}}, ["尚未就绪", "先修复健康检查配置"], "R_explicit_control"),
            ("negative-causal-control", "镜像拉取或 ConfigMap 挂载是否是本次失败根因？只回答是或否。", {"type": "binary", "answer": "否"}, ["[image_pull] status=completed", "[configmap_mount] status=completed"], "T_negative_evidence"),
        ],
        "forbidden": lambda v: [str(8180 + v), str(8080 + v), "connection_refused", "image_pull", "configmap_mount"],
    },
)


def derived_sample(
    encode: Callable[[str], list[int]], tool_target: int, carrier_target: int, variant: int
) -> dict[str, Any]:
    # Rotate continuously across the full factorial grid. There are 120 derived
    # samples, so this produces exactly 30 instances for each of the four families.
    grid_index = TOOL_TARGETS.index(tool_target) * len(CARRIER_TARGETS) + CARRIER_TARGETS.index(carrier_target)
    family = DERIVED_FAMILIES[(grid_index * VARIANTS + variant) % len(DERIVED_FAMILIES)]
    sid = f"derived-{family['name']}-t{tool_target:05d}-r{carrier_target:03d}-v{variant:02d}"
    system = "你是故障诊断 Agent。必须依据工具日志更新状态，只报告结论和下一步，不逐行复述证据。"
    user = "执行当前检查，判断失败阶段、当前状态和下一步动作。"
    tool_call = json.dumps(
        {"name": "diagnostic_command", "arguments": {"scenario": family["name"]}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    s_text = serialize_system_user(system, user, tool_call)
    critical = f"scenario={family['name']}\n" + family["critical"](variant)
    suffix = "exit_code=1\n"

    def filler(i: int) -> str:
        checksum = hashlib.sha256(f"{sid}:telemetry:{i}".encode()).hexdigest()[:16]
        components = ("scheduler", "metrics", "audit", "storage-cache")
        return f"[info] event={i:05d} component={components[i % len(components)]} observation=nominal checksum={checksum}\n"

    t_text = fit_with_logs(
        encode,
        "<|im_start|>tool\n" + critical,
        suffix + "<|im_end|>\n",
        tool_target,
        filler,
    )
    r_text = fit_carrier(encode, family["carrier"], carrier_target)

    probes = []
    for probe_id, question, gold, evidence, info_class in family["probes"](variant):
        probes.append(
            {
                "probe_id": probe_id,
                "Q": segment_metadata(encode, serialize_probe(question)),
                "gold": gold,
                "evidence": evidence,
                "information_class": info_class,
            }
        )

    sample = {
        "schema_version": "latent-handoff/v1",
        "sample_id": sid,
        "template_type": "derived_workflow_state",
        "scenario_family": family["name"],
        "seed": stable_seed(sid),
        "token_targets": {"T": tool_target, "R": carrier_target},
        "segments": {
            "S": segment_metadata(encode, s_text),
            "T": segment_metadata(encode, t_text),
            "R": segment_metadata(encode, r_text),
        },
        "latent_state": family["state"](variant),
        "forbidden_in_R": family["forbidden"](variant),
        "probes": probes,
        "cache_conditions": ["full", "evict", "replay"],
    }
    add_layout(sample)
    return sample


def validate(rows: list[dict[str, Any]], encode: Callable[[str], list[int]]) -> dict[str, Any]:
    errors: list[str] = []
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("sample_id values are not unique")

    grid = Counter()
    template_counts = Counter()
    family_counts = Counter()
    exact_fact_tuples: set[tuple[str, str, str]] = set()

    for row in rows:
        sid = row["sample_id"]
        template_counts[row["template_type"]] += 1
        if row.get("scenario_family"):
            family_counts[row["scenario_family"]] += 1
        tt, rt = row["token_targets"]["T"], row["token_targets"]["R"]
        grid[(row["template_type"], tt, rt)] += 1

        for name in ("S", "T", "R"):
            seg = row["segments"][name]
            actual_ids = encode(seg["text"])
            if len(actual_ids) != seg["token_count"]:
                errors.append(f"{sid}:{name}: stored token count mismatch")
            if token_hash(actual_ids) != seg["token_id_sha256"]:
                errors.append(f"{sid}:{name}: token hash mismatch")

        if row["segments"]["T"]["token_count"] != tt:
            errors.append(f"{sid}: T target mismatch")
        if row["segments"]["R"]["token_count"] != rt:
            errors.append(f"{sid}: R target mismatch")

        r_text = row["segments"]["R"]["text"]
        for forbidden in row["forbidden_in_R"]:
            if forbidden and forbidden.lower() in r_text.lower():
                errors.append(f"{sid}: forbidden value leaked into R: {forbidden}")

        t_text = row["segments"]["T"]["text"]
        for probe in row["probes"]:
            q_ids = encode(probe["Q"]["text"])
            if len(q_ids) != probe["Q"]["token_count"] or token_hash(q_ids) != probe["Q"]["token_id_sha256"]:
                errors.append(f"{sid}:{probe['probe_id']}: Q token metadata mismatch")
            evidence_text = r_text if probe["information_class"] == "R_explicit_control" else t_text
            for evidence in probe["evidence"]:
                if evidence not in evidence_text:
                    errors.append(f"{sid}:{probe['probe_id']}: missing evidence: {evidence}")
                elif probe["information_class"] != "R_explicit_control" and t_text.count(evidence) != 1:
                    errors.append(f"{sid}:{probe['probe_id']}: T evidence is not unique: {evidence}")

        ns = row["segments"]["S"]["token_count"]
        nt = row["segments"]["T"]["token_count"]
        nr = row["segments"]["R"]["token_count"]
        layout = row["layout"]
        expected_layout = ([0, ns], [ns, ns + nt], [ns + nt, ns + nt + nr])
        if (layout["S"], layout["T"], layout["R"]) != expected_layout:
            errors.append(f"{sid}: invalid token layout")
        if layout["logical_next_position"] != ns + nt + nr:
            errors.append(f"{sid}: invalid logical next position")
        if layout["physical_tokens_after_eviction"] != ns + nr:
            errors.append(f"{sid}: invalid physical length after eviction")

        if row["template_type"] == "exact_fact":
            state = row["latent_state"]
            fact_tuple = (state["server_ip"], state["ssh_port"], state["request_id"])
            if fact_tuple in exact_fact_tuples:
                errors.append(f"{sid}: duplicated randomized exact-fact tuple")
            exact_fact_tuples.add(fact_tuple)

    for template in ("exact_fact", "derived_workflow_state"):
        for tt in TOOL_TARGETS:
            for rt in CARRIER_TARGETS:
                if grid[(template, tt, rt)] != VARIANTS:
                    errors.append(f"grid mismatch: {template} T={tt} R={rt}")

    expected_per_family = (len(TOOL_TARGETS) * len(CARRIER_TARGETS) * VARIANTS) // len(DERIVED_FAMILIES)
    for family in DERIVED_FAMILIES:
        if family_counts[family["name"]] != expected_per_family:
            errors.append(f"scenario family imbalance: {family['name']}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "num_samples": len(rows),
        "template_counts": dict(sorted(template_counts.items())),
        "scenario_family_counts": dict(sorted(family_counts.items())),
        "grid_cells": len(grid),
        "expected_samples_per_grid_cell": VARIANTS,
        "checks": {
            "unique_sample_ids": len(ids) == len(set(ids)),
            "exact_T_token_budgets": not any("T target mismatch" in e for e in errors),
            "exact_R_token_budgets": not any("R target mismatch" in e for e in errors),
            "token_hashes_recomputed": not any("token" in e and "mismatch" in e for e in errors),
            "forbidden_R_leakage_absent": not any("leaked into R" in e for e in errors),
            "evidence_present_and_unique": not any("evidence" in e for e in errors),
            "logical_physical_layout_valid": not any("layout" in e or "position" in e or "physical" in e for e in errors),
            "full_factorial_grid_complete": not any("grid mismatch" in e for e in errors),
            "scenario_families_balanced": not any("scenario family imbalance" in e for e in errors),
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local Qwen tokenizer/model directory")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)

    exact_rows: list[dict[str, Any]] = []
    derived_rows: list[dict[str, Any]] = []
    for tool_target in TOOL_TARGETS:
        for carrier_target in CARRIER_TARGETS:
            for variant in range(VARIANTS):
                exact_rows.append(exact_fact_sample(encode, tool_target, carrier_target, variant))
                derived_rows.append(derived_sample(encode, tool_target, carrier_target, variant))

    all_rows = exact_rows + derived_rows
    report = validate(all_rows, encode)
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))

    exact_path = out / "exact_fact_v1.jsonl"
    derived_path = out / "derived_workflow_v1.jsonl"
    combined_path = out / "latent_handoff_v1.jsonl"
    dump_jsonl(exact_path, exact_rows)
    dump_jsonl(derived_path, derived_rows)
    dump_jsonl(combined_path, all_rows)

    report["tokenizer"] = {
        "source": args.model,
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "model_max_length": tokenizer.model_max_length,
        "encoding": "add_special_tokens=false per serialized segment",
    }
    report["artifacts"] = {
        p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
        for p in (exact_path, derived_path, combined_path)
    }
    (out / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
