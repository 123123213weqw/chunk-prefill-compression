# Publication snapshot contract

This repository is a versioned publication, not a live mirror of the GPU server.
Completed v1-v5 results are exported with raw JSON records. The canonical v5
run `4080-20260921-v1` contains all3200 atomic units,256 paired documents and
independent clean timing. The earlier v5 directory
ending in `snapshot-507` is frozen at507 completed atomic units out of3200.
Its62 common paired documents are a subset of the256 planned documents.
`SNAPSHOT_IN_PROGRESS` refers to publication coverage, not a process running
inside this repository. No final timing result is included for this snapshot.

Original synthetic text, gold answers and measurement values are preserved.
Only private paths/host labels/GPU UUIDs are redacted, and source hashes that
refer to changed public files are rebased explicitly. See the provenance map
for original and public hashes. Report audit passes establish consistency,
not research success, losslessness, deployment readiness or universal validity.

The original SHA256 fingerprints are evidence identifiers; reproducing private
path strings from them is neither necessary nor part of the audit. CPU audits
operate on the public hashes and public data. Audit scripts may regenerate
report/summary files deterministically. New hardware experiments must write
new output directories and supply an actual model path rather than placeholders.

The completed v5 run was published separately on2026-09-22, retaining the partial
snapshot as historical evidence. PUBLIC_PROVENANCE_20260922.json records this
new export; PUBLIC_PROVENANCE_20260921.json remains unchanged. Final public and
private statistical summaries were checked for equality after report regeneration.

The failure_diagnosis_v5 extension (2026-09-22) publishes all108 diagnostic units
for3 previously exposed loss cases and3 correct controls. It is a post-hoc oracle
intervention study, not independent evaluation. Runtime-frozen sources are separate
from later written interpretation. Its provenance is in
PUBLIC_PROVENANCE_DIAGNOSIS_20260922.json. Existing completed/partial v5 runs
and their provenance files remain unchanged.

The v6 inverse failure-causality run (2026-09-23) publishes all36 new atomic
records, matched same-budget placebo interventions, frozen source and independent
CPU audit. It reuses exposed v5 cases, so is mechanistic and post-hoc, not a new
validation cohort. Its path/source fingerprints are mapped in
PUBLIC_PROVENANCE_CAUSALITY_20260923.json. Previous releases are unchanged.
