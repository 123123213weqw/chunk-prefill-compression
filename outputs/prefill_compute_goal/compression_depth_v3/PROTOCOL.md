# Compression depth v3 — exploratory, frozen before GPU results

Date: 2026-09-20. Goal: first-prefill savings in an untrained, dense Qwen3 model;
not prefix-cache reuse, not a new linear-attention architecture. No training.

## Data and information boundary
64 fresh synthetic documents = 4 tasks × 2 lengths (4096,16384) × 4 instances ×
2 counterfactual members. Tasks: fields, narrative, code branch, event updates.
Instances 0/1 are calibration (32); 2/3 are validation (32); pairs never cross
splits. Gold is independently reparsed from text. Same raw documents for both
models; tokenizer-specific full 64-token T chunks, remainder retained in R.
Counterfactual pairs change the relevant fact; exact evidence token positions
are NOT held constant, so pair differences cannot establish a positional causal
mechanism. This is small exploratory validation, NOT a locked final benchmark.

Routing consumes only the hidden states of a completed chunk of 64 tokens.
No Q, gold, evidence annotations, or future chunks enter its API. Chunk-wide
routing is not token-level autoregressive likelihood preservation. Lower-layer
KV remains full; this is not an independently reusable constant-size memory.

## Models and gates
Qwen3-4B-Instruct-2507 primary (36 layers); Qwen3-0.6B secondary (28), conditional
on download availability and valid native baseline. Same family / different
scale, NOT cross-architecture generalization. FP16, SDPA, batch=1, block=1024.
No concurrent foreign GPU process allowed at measured boundaries. Native HF
Full and manual identity must match: relative KV/logit RMS <=0.005,
first-token KL <=0.001, greedy generated IDs identical. Fail closed before
compressed validation. Cache lengths and seven projection/MLP input shapes
are audited on every compressed correctness prefill. No accuracy-based sample
exclusion. Full accuracy <75% in a task/length cell marks it uninformative for
compression robustness; all results remain reported.

## Fixed scan and equal-budget definition
Depths round(L × {1/3,1/2,2/3,3/4,5/6}), keep ∈ {32,48} per 64-token chunk.
Same contiguous equal-bin arithmetic-mean family, retained RoPE at bin ends.
A floor-vs-ceil bin-phase control is added for early/48.
Primary matched pair: early=L/2, keep48; late=3L/4, keep32.
Both save EXACTLY 1/8 of T linear-projection/MLP layer-token work:
(L-L/2)×16 = (L-3L/4)×32. Additional integral matched-budget keep16/40 points
are included where possible. This is NOT exact equality of total FLOPs:
attention is quadratic, and routing/merging add work. Report actual hooked
linear FLOPs, dense rectangular QK+AV FLOP estimates, cache bytes and wall time
separately. Never label this pair equal total compute. A strict end-to-end
compute-matched claim requires a later latency-budget frontier/interpolation.

## Query-blind criterion and frozen adaptive policies
score = ||H - unpool(mean_pool(H))||_F / max(||H||_F, epsilon).
This is a testable surrogate, NOT an attention/logit error bound. Identity
teacher collects per-chunk scores at scan depths on calibration ONLY.
Thresholds: early/48 median and 25th percentile; late/32 25th percentile,
pooled over calibration chunks (long documents receive more weight).
Freeze thresholds before any validation compression results.
- adaptive_equal: score <= early median → early/48; otherwise late/32.
- random_equal: deterministic query-blind pseudo-random early/late routing.
- adaptive_safe: early score <= q25 → early/48; otherwise late score <= q25
  → late/32; otherwise leave64. Its compute budget is variable, not matched.
No answer-dependent threshold selection, retraining or post-validation tuning.
Teacher scores are unavailable to router during validation. Diagnostic
recording is off in timings; mandatory scoring/routing/merging remains on.

## Measurements and interpretation
Correctness: strict JSON exact task match plus an uncompressed R control.
Report all errors, Full-correct conditional retention, paired loss/gain counts,
per-task/length, counterfactual pair consistency, pair-cluster bootstrap CIs.
Calibration runs Full, identity teacher and the two primary fixed baselines;
validation runs Full, identity and every fixed/adaptive/random method.
For indicator diagnostics, compare document aggregate mean/p95 teacher scores
with Full-correct task loss *within each fixed method*, never pool depths to
manufacture a correlation. AUROC requires both lost and retained examples;
otherwise report not identifiable. This is document association, NOT evidence
of which individual chunk caused failure.
Timings: fixed validation instance2/member0, 8 documents/model. Methods Full
at block1024/4096, two primary fixed, adaptive_equal/random_equal/adaptive_safe.
One warmup and five randomized-order measured repeats. Record each trial's
prefill S+T+R wall time, peak CUDA allocated/reserved bytes and GPU boundary
telemetry; no decode or shape hooks. Do not include data tokenization/model
loading; include all on-path router/merge work. Abort timings on interference.
Source, data, model config/tokenizer hashes, software versions and phases are
saved. Finite pipeline ends with report or explicit FAILED status. CPU tests
are implementation checks, not real-model task evidence. No public upload.
