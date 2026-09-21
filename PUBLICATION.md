# Publication snapshot contract

This repository is a versioned publication, not a live mirror of the GPU server.
Completed v1-v4 results are exported with raw JSON records. The v5 directory
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

Future v5 completion should be published as a separate completed snapshot or
canonical completed run, retaining this partial snapshot as historical evidence.
Do not silently replace a partial report with a claim that all3200 units ran.
