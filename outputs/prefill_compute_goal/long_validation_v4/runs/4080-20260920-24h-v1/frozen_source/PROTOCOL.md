# Long validation v4 — predeclared, single RTX 4080, maximum 24 h active budget

Purpose: test fixed-depth merging on fresh data and diagnose the earlier v3
failures. No training, no Goal, no recurring scheduler, no public upload.
Model: local Qwen3-4B-Instruct-2507, FP16 SDPA, block1024, greedy <=96 tokens.
0.6B is excluded because its v3 native task baseline was inadequate.

## Frozen fresh validation
256 documents, four new generator seeds 20260921..20260924; each seed gives
4 task families × 4K/16K × 4 instances × 2 paired members. All are validation,
not threshold fitting. Exact text gold is reparsed independently. Pair changes
are factual but evidence token positions are not exactly matched; do not claim
pure positional causal effects from the paired text alone. v3 document IDs
must be disjoint. These are new seeds of the same templates, NOT new domains.
All baseline errors remain in denominators; cells with native accuracy <75%
are flagged inadequate. Thresholds are copied unchanged from v3 calibration.

Fixed candidates: depth12/18/24/27/30 with keep32; depth18/27 keep48;
depth24 keep40; depth30 keep16; depth18 keep48 floor phase;
endpoint instead of mean at depth18/48 and depth27/32; adaptive_equal and
random_equal with frozen v3 settings. No selection on fresh outcomes.
Full/manual identity gates must pass per document BEFORE lossy trials on that
document: KV/logit relative RMS <=.005, first-token KL<=.001, greedy IDs equal.
This study does not claim all candidates have equal compute. Only the v3
primary early18/48 vs late27/32 and relevant variants match T linear work.

## Mechanism intervention on already-exposed v3 validation (NOT held-out)
Select up to16 v3 Full-correct documents deterministically: prioritize late27/32
losses, then adaptive losses, then early18/48 losses; supplement with successful
controls spread across task/length cells. Record selection and prior outcomes.
For every selected document and every64-token chunk, test four interventions:
1 only this chunk compressed at18/48, all other chunks uncompressed;
2 all chunks compressed at18/48 except this chunk restored;
3 only this chunk compressed at27/32;
4 all chunks compressed at27/32 except this chunk restored.
Recompute the entire prefix from original input every trial. This avoids
pretending a single KV splice reverses downstream consequences. Chunk-wise
interventions are a causal test of this operation, not an assertion that one
chunk contains all necessary information. Interactions can defeat single-chunk
restoration. Failed/no-effect interventions are retained.
Record exact task answer, uncompressed-control answer, canonical correct-vs-
paired-counterfactual answer sequence log likelihood, retained KV/cache shape,
query-blind local merge score (diagnostic only) and instrumented runtime.
Canonical sequence log likelihood is not semantic correctness probability;
length/format can affect it. Gold is used for diagnostics, never router input.

## Budget and robustness
Fresh validation first, then complete diagnostic baselines for all16 documents,
then chunk interventions round-robin across documents and four conditions so
an early deadline does not cover only the first document. Deadline is checked
between units. Each completed unit is atomically written and restart-skippable.
A 24h deadline yields BUDGET_EXHAUSTED with coverage, not COMPLETED. A unit may
finish slightly after deadline (bounded by one prefill+short probe unit).
No sleeping to fill24h; finish early if the declared queue finishes.
CUDA OOM or failed identity/shape gates stop with explicit FAILED status.
Foreign GPU processes at boundaries stop as INTERRUPTED; never kill others.
Source/data/threshold hashes checked on resume. A killed/partial JSON file is
not accepted as a completed unit. Resume requires explicit --resume; no daemon.
Timing here is instrumented diagnostics, not a new production-speed claim.
A source snapshot, progress, JSON units, and summary/report are written. Reports
state the completed denominators and distinguish exploratory diagnostics from
fresh-validation results. No parameter changes after fresh results are seen.
