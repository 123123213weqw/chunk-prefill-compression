# Query-blind adaptive chunk compression experiment

Read PROTOCOL.md before interpreting results. Entry: pipeline.py --out <new directory>.
GPU-only; sequential Qwen3-4B-Instruct-2507 and Qwen3-0.6B. New output directories
required; finite jobs, identity gates, shape audits, calibration-only thresholds,
strict JSON scoring, randomized measured timing order and rebuildable reports.
No Goal/automation created and no other GPU task killed.

Files: data.py/dataset.jsonl (64 synthetic paired docs); engine.py (fixed and
adaptive routing); base_engine.py (cache mechanics copied from v2); test_engine.py
(8 CPU implementation tests); run.py (CUDA runner); report.py (coverage/integrity
checks and statistics); pipeline.py (source snapshot and sequential supervisor).

Primary question: whether cheap within-chunk reconstruction error predicts
compressibility beyond a fixed depth/budget. Negative findings are valid.
The exact matched quantity is T linear-projection/MLP work, NOT total attention
FLOPs. Actual wall time includes mandatory routing/merging overhead. This is an
exploratory test on synthetic tasks; it cannot establish general robustness.
