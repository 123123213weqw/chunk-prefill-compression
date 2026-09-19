# Real Qwen KV representation reconstruction — development pilot

RTX 4080, FP16 model; CPU FP32 attention reconstruction. 4 development samples / 2 paired documents; not held-out evidence.
Layers 0/9/18/27/35; all 8 KV and 32 query heads; T chunks 7/31/55, each 64 tokens.
Each condition compresses ONE chunk, leaving all other visible KV raw. Future Q used only for scoring, never clustering.
All sampled queries are after the completed chunk; full causal context includes each query token itself.
Percentages below are attention vector errors, NOT answer error rates. Query/head observations are correlated; no independence CI claimed.

| Method | 64→m | Mixed mean relative L2 | Mixed p95 | Chunk-only mean relative L2 | Chunk logZ RMSE |
|---|---:|---:|---:|---:|---:|
| mean_no_mass | 32 | 1.93% | 9.68% | 39.34% | 1.5056 |
| mean_mass | 32 | 1.35% | 5.80% | 39.34% | 0.9539 |
| cluster_mass | 32 | 0.80% | 3.36% | 16.29% | 0.3528 |
| cluster_mass_fp16 | 32 | 0.80% | 3.36% | 16.29% | 0.3528 |
| mean_no_mass | 16 | 2.55% | 12.71% | 52.10% | 2.6501 |
| mean_mass | 16 | 1.86% | 8.58% | 52.10% | 1.4672 |
| cluster_mass | 16 | 1.55% | 6.75% | 31.14% | 0.7648 |
| cluster_mass_fp16 | 16 | 1.55% | 6.75% | 31.14% | 0.7648 |

Reference capture check: max FP32-reconstruction / actual-FP16-output relative RMS = 0.000255; threshold 0.005.

## Scope limits
- No model weights, future queries, answers, or target evidence used to choose clusters.
- cluster_mass_fp16 rounds prototype K/V to FP16 before FP32 scoring; bias stays FP32. Other methods retain FP32 prototypes.
- Physical n→m compression applies only to the tested chunk, not the full 4K memory.
- A small mixed error may only mean this chunk received little original attention; inspect chunk-local errors and masses.
- CPU cluster timings are reference implementation cost, not optimized GPU compression overhead.
- No full-model compressed decoding, readout accuracy, multilayer error propagation, or speedup has been tested.
- No training, learned encoder, all-query robustness, or lossless-general-compression claim.
