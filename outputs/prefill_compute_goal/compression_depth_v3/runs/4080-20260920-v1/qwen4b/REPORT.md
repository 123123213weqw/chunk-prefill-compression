# Compression depth v3: Qwen3-4B-Instruct-2507

Exploratory synthetic validation: 32 documents / 16 pairs. Same-family results do not establish generality.

| Method | Exact task | Lost / Full-correct | Control |
|---|---:|---:|---:|
| full | 26/32 | 0/26 | 32/32 |
| identity | 26/32 | 0/26 | 32/32 |
| d12_k32 | 11/32 | 15/26 | 32/32 |
| d12_k48 | 21/32 | 6/26 | 32/32 |
| d18_k32 | 15/32 | 12/26 | 32/32 |
| d18_k48 | 19/32 | 8/26 | 32/32 |
| d24_k32 | 25/32 | 2/26 | 32/32 |
| d24_k48 | 25/32 | 2/26 | 32/32 |
| d27_k32 | 26/32 | 1/26 | 32/32 |
| d27_k48 | 25/32 | 1/26 | 32/32 |
| d30_k32 | 25/32 | 1/26 | 32/32 |
| d30_k48 | 25/32 | 1/26 | 32/32 |
| d18_k48_floor | 22/32 | 4/26 | 32/32 |
| d30_k16 | 25/32 | 1/26 | 32/32 |
| d24_k40 | 26/32 | 1/26 | 32/32 |
| adaptive_equal | 23/32 | 4/26 | 32/32 |
| random_equal | 26/32 | 3/26 | 32/32 |
| adaptive_safe | 23/32 | 3/26 | 32/32 |

## Prefill timings
Includes router and merge overhead; excludes loading/tokenization and decode.

| Method | Median per-document time reduction vs native b1024 |
|---|---:|
| native_b1024 | 0.00% |
| native_b4096 | -7.46% |
| d18_k48 | 14.87% |
| d27_k32 | 13.34% |
| adaptive_equal | 13.54% |
| random_equal | 15.17% |
| adaptive_safe | 4.59% |

## Limits
Only linear projection/MLP work is exactly budget-matched, not total FLOPs or wall time.
Full accuracy below 75% in a task/length cell invalidates robustness claims there; see summary.json.
Indicator AUROC=null means losses or retained cases are absent, not a successful predictor.
No method was tuned on validation. This small study cannot establish robust universal compression.

Audit PASS: 128 identity gates, 576 shape audits, 336 timing trials.
