# v5: candidate replication + clean timing

Read PROTOCOL.md before interpreting results. Same Qwen3-4B model, new256
synthetic documents (seeds20260926..29). Three frozen compressed candidates:
d27_mean32, d27_endpoint32, d30_mean16. Full/manual identity baselines. All four
conditions tested at1024 and4096 prefill block sizes. No new parameter tuning.

3200 atomic units:2048 correctness (Full includes manual identity),1152 timing.
Clean timings:16 predefined documents ×4methods ×2blocks ×(2warmup+7measured).
No feature logging/scoring/hooks in measured prefix; mandatory merge/indexing
and mask/position management remain. Report each block and best-of-two separately.

Entry: launch.py --out <new-directory> --hours12. Frozen source/data snapshot,
17 CPU tests, sequential one-GPU worker;12h safety budget, finish early if done.
Check results/status.json or progress.json; report/summary every128 units and
at finish. Partial/budget exhaustion is not completion. New units atomically
saved in results/units/. Explicit resume via frozen_source/run.py --resume
--out <same results> --hours <budget>. Resumed budget is per invocation.

GPU preflight smoke.py uses exposed v4 diagnostics, NOT new validation answers.
Do not use smoke or correctness runtime as production speed estimates.
No Goal, recurring automation, other-process killing, or public upload.
