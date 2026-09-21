# v5 frozen-candidate replication
Status: PARTIAL; units 507/3200; shared paired correctness coverage 62/256.

| Method/block | Task correct | Full-correct lost | Conditional retention |
|---|---:|---:|---:|
| full/b1024 | 52/62 | 0/52 | 100.00% |
| d27_mean32/b1024 | 50/62 | 2/52 | 96.15% |
| d27_endpoint32/b1024 | 46/62 | 6/52 | 88.46% |
| d30_mean16/b1024 | 50/62 | 2/52 | 96.15% |
| full/b4096 | 52/62 | 0/52 | 100.00% |
| d27_mean32/b4096 | 49/62 | 3/52 | 94.23% |
| d27_endpoint32/b4096 | 46/62 | 6/52 | 88.46% |
| d30_mean16/b4096 | 50/62 | 2/52 | 96.15% |

## Clean prefill time reduction
No decode/scoring/hooks in timing; required merge/indexing/mask/position costs included.
| Method/block | Complete timing docs | 4K reduction | 16K reduction |
|---|---:|---:|---:|
| full/b1024 | 0/16 | NA | NA |
| full/b4096 | 0/16 | NA | NA |
| full/best_of_two_blocks | 0/16 | NA | NA |
| d27_mean32/b1024 | 0/16 | NA | NA |
| d27_mean32/b4096 | 0/16 | NA | NA |
| d27_mean32/best_of_two_blocks | 0/16 | NA | NA |
| d27_endpoint32/b1024 | 0/16 | NA | NA |
| d27_endpoint32/b4096 | 0/16 | NA | NA |
| d27_endpoint32/best_of_two_blocks | 0/16 | NA | NA |
| d30_mean16/b1024 | 0/16 | NA | NA |
| d30_mean16/b4096 | 0/16 | NA | NA |
| d30_mean16/best_of_two_blocks | 0/16 | NA | NA |

## Limits
New seeds, same four synthetic templates and one model/GPU. Not new-domain or architecture generalization.
Native accuracy below75% makes a cell inadequate for strong preservation claims. All cells remain reported.
Equal total task counts do not imply identical correct answers or losslessness. Confidence intervals and losses/gains are in summary.json.
Best-of-two timing is explicitly tuning on the timing set. Cross-block output differences are reported in summary.json.
A partial report is not a completed replication.
