# Long validation v4
Status: COMPLETE. Units: 13360/13360.
Fresh shared paired coverage: 256/256 documents. Diagnostic set is previously exposed v3 data.

| Method | Correct | Full-correct lost | Control |
|---|---:|---:|---:|
| full | 202/256 | 0/202 | 256/256 |
| d12_k32 | 91/256 | 116/202 | 256/256 |
| d18_k32 | 126/256 | 87/202 | 256/256 |
| d24_k32 | 191/256 | 23/202 | 256/256 |
| d27_k32 | 199/256 | 9/202 | 256/256 |
| d30_k32 | 202/256 | 2/202 | 256/256 |
| d18_k48 | 182/256 | 29/202 | 256/256 |
| d27_k48 | 198/256 | 8/202 | 256/256 |
| d24_k40 | 194/256 | 17/202 | 256/256 |
| d30_k16 | 205/256 | 1/202 | 256/256 |
| d18_k48_floor | 158/256 | 50/202 | 256/256 |
| d18_k48_endpoint | 152/256 | 59/202 | 256/256 |
| d27_k32_endpoint | 203/256 | 2/202 | 256/256 |
| adaptive_equal | 190/256 | 23/202 | 256/256 |
| random_equal | 193/256 | 18/202 | 256/256 |

## Interpretation limits
Fresh seeds use the same synthetic templates: not new domains or cross-architecture generalization.
Native baseline <75% makes that cell inadequate for a compression-robustness claim. See summary.json.
Intervention likelihood compares exact canonical JSON sequences, not semantic correctness probabilities.
Runtime is instrumented diagnostic cost, not production prefill speed. Single-chunk effects can interact.
A partial report is not a completed 24-hour validation. Thresholds and candidate rules remain frozen.
