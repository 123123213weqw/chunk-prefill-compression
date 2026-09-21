# Compression depth v3: Qwen3-0.6B

Exploratory synthetic validation: 32 documents / 16 pairs. Same-family results do not establish generality.

| Method | Exact task | Lost / Full-correct | Control |
|---|---:|---:|---:|
| full | 9/32 | 0/9 | 32/32 |
| identity | 9/32 | 0/9 | 32/32 |
| d9_k32 | 8/32 | 2/9 | 32/32 |
| d9_k48 | 9/32 | 3/9 | 32/32 |
| d14_k32 | 8/32 | 2/9 | 32/32 |
| d14_k48 | 8/32 | 3/9 | 32/32 |
| d19_k32 | 8/32 | 2/9 | 32/32 |
| d19_k48 | 9/32 | 3/9 | 32/32 |
| d21_k32 | 12/32 | 0/9 | 32/32 |
| d21_k48 | 10/32 | 0/9 | 32/32 |
| d23_k32 | 11/32 | 0/9 | 32/32 |
| d23_k48 | 10/32 | 0/9 | 32/32 |
| d14_k48_floor | 9/32 | 1/9 | 32/32 |
| adaptive_equal | 8/32 | 2/9 | 32/32 |
| random_equal | 9/32 | 1/9 | 32/32 |
| adaptive_safe | 10/32 | 0/9 | 32/32 |

## Prefill timings
Includes router and merge overhead; excludes loading/tokenization and decode.

| Method | Median per-document time reduction vs native b1024 |
|---|---:|
| native_b1024 | 0.00% |
| native_b4096 | -1.31% |
| d14_k48 | 14.70% |
| d21_k32 | 13.20% |
| adaptive_equal | 13.07% |
| random_equal | 13.62% |
| adaptive_safe | 3.36% |

## Limits
Only linear projection/MLP work is exactly budget-matched, not total FLOPs or wall time.
Full accuracy below 75% in a task/length cell invalidates robustness claims there; see summary.json.
Indicator AUROC=null means losses or retained cases are absent, not a successful predictor.
No method was tuned on validation. This small study cannot establish robust universal compression.

Audit PASS: 128 identity gates, 512 shape audits, 336 timing trials.
