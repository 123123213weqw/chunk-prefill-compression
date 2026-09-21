# v5: frozen-candidate replication and clean prefill timing

Frozen on2026-09-21 before new sample outcomes. Model Qwen3-4B-Instruct-2507,
FP16 SDPA batch1, same raw ChatML prompts and greedy <=96 tokens as v3/v4.
No training, no new adaptive criterion. No claim of cross-architecture transfer.

## New evaluation set and candidates
256 new documents from seeds20260926..20260929, disjoint IDs from v3/v4.
Four existing synthetic task families,4K/16K,4instances and2paired members each.
Gold independently reparsed; original pair changes do not preserve exact token
positions. New seeds are independent of candidate selection, but templates and
model remain the same. No parameter changes after these outcomes are inspected.
Full baseline and three frozen candidates:
- d27_mean32: prior reference, depth27 then64->32 contiguous mean;
- d27_endpoint32: depth27 then64->32 retaining each pair's final hidden vector;
- d30_mean16: depth30 then64->16 contiguous four-token mean.
All three save12.5% of T projection/MLP layer-token work, not equal total FLOPs.

Every document is evaluated at prefill block1024 AND4096. Full/manual identity
must pass before lossy runs for that document/block: KV/logit relative RMS<=.005,
first-token KL<=.001, exact greedy token IDs equal. Cross-block differences are
reported, not hidden; they need not be zero in FP16. Cache/input-shape audits
run during correctness only. Retain all Full errors; native cell accuracy<75%
marks that task/length cell inadequate for robust preservation claims.
Report exact task match, control, losses/gains relative to same-block Full,
conditional correct-answer retention and pair-cluster bootstrap confidence
intervals. Full total-score equality is not evidence of losslessness.
Do not promote a candidate just because its total accuracy is higher; v4-based
candidate selection requires this frozen new-seed replication.

## Clean timing, predeclared independently of answers
16 documents: seeds20260926/27 ×4tasks ×2lengths ×instance0/member0.
4methods (native+3 candidates) ×2blocks ×(2warmups+7measured repeats) =1152 trials.
Shuffle all method/block combinations each repeat with a fixed seed. Each
trial re-prefills S+T+R from empty KV. No decode, task scoring, teacher forcing,
shape hooks, feature logging, GPU telemetry calls or output writing inside the
timed interval. Mandatory merging/indexing, mask/position handling and CPU/GPU
copies remain included. Use synchronized wall time. Native timing directly
calls HF model.model, avoiding extra experimental cache-position bookkeeping.
Record per-trial peak allocated/reserved CUDA bytes and boundary GPU process,
clock,temperature,power telemetry. Stop if foreign GPU processes appear; do not
kill them. Warmups are saved but excluded from estimates. No outcome-selected
timing subset. Prior instrumented v4 runtimes are NOT production baselines.

Report per-document median time and per-length reductions with bootstrap CIs
resampling documents, not treating seven repeated timings as independent tasks.
Two comparisons: same block, and best observed native-block median versus
best observed candidate-block median (explicitly labelled two-point tuning on
the timing set). Also show every block separately; do not imply universally
optimal kernels or batching. Timing results apply only to this GPU/software.

## Finite execution and artifacts
2048 atomic correctness units (256docs ×2blocks ×4units; full unit includes an
identity gate), then1152 clean timing units, total3200 units. Source/data hashes,
config/tokenizer hashes and software versions saved. New immutable run directory.
Successful units atomically persisted; explicit --resume skips matching units.
12h safety deadline per invocation; finish earlier if queue completes. At the
deadline finish current unit, mark BUDGET_EXHAUSTED and report exact coverage.
OOM/failed gates report FAILED; external GPU use reports INTERRUPTED. No daemon,
Goal, recurring task or public upload. Summaries every128 units and at finish.
