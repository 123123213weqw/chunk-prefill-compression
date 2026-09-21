# Long verification on RTX 4080

See PROTOCOL.md. Frozen fresh dataset:256 documents, four new seeds, same four
synthetic task templates. 14 compression variants plus Full/identity gates.
Exposed v3 diagnostics:16 selected documents,2368 chunks,9472 isolated-merge or
single-chunk-restoration trials. Total queue:13360 atomic units.

Entry: launch.py --out <new-directory> --hours 24. It snapshots source/data,
runs14 CPU implementation tests, and supervises one GPU worker. Maximum24h
worker budget; finish early if queue completes. Worker exits at unit boundaries.
Runs do not kill foreign GPU processes. No scheduler/Goal/public upload.

Artifacts: frozen_source/, tests.log, worker.log, results/status.json,
results/progress.json, results/units/*.json, results/summary.json and REPORT.md.
Worker checkpoints each successful unit. A report is rebuilt every256 units
and at completion/budget expiry. Failed jobs retain their previous checkpoints.

Explicit restart only: use the same frozen_source/run.py with --resume,
--out <same results directory> and --hours <remaining budget>. Source/data
hashes must match. A resumed invocation receives its explicitly supplied budget;
it is NOT automatically capped by the first invocation's original deadline.

Timing is instrumented diagnostic runtime, not production acceleration. Gold
likelihood is for mechanism analysis only, never routing or threshold fitting.
New-seed validation does not establish new-domain or cross-architecture transfer.
