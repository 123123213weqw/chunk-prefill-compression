#!/usr/bin/env python3
"""Recompute deterministic answer matches and summary from saved raw answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_prefill_experiment import aggregate, answer_matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    changes = 0
    for row in rows:
        for probes in row["conditions"].values():
            for probe in probes:
                updated = answer_matches(probe["answer"], probe["gold"])
                if updated != probe["match"]:
                    changes += 1
                probe["match"] = updated

    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    old_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary = aggregate(rows)
    if "elapsed_seconds" in old_summary:
        summary["elapsed_seconds"] = old_summary["elapsed_seconds"]
    summary["rescored_match_changes"] = changes
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
