# v5 frozen-candidate replication
Status: COMPLETE; units 3200/3200; shared paired correctness coverage 256/256.

| Method/block | Task correct | Full-correct lost | Conditional retention |
|---|---:|---:|---:|
| full/b1024 | 213/256 | 0/213 | 100.00% |
| d27_mean32/b1024 | 208/256 | 8/213 | 96.24% |
| d27_endpoint32/b1024 | 201/256 | 14/213 | 93.43% |
| d30_mean16/b1024 | 213/256 | 3/213 | 98.59% |
| full/b4096 | 213/256 | 0/213 | 100.00% |
| d27_mean32/b4096 | 207/256 | 9/213 | 95.77% |
| d27_endpoint32/b4096 | 201/256 | 14/213 | 93.43% |
| d30_mean16/b4096 | 214/256 | 3/213 | 98.59% |

## Clean prefill time reduction
No decode/scoring/hooks in timing; required merge/indexing/mask/position costs included.
| Method/block | Complete timing docs | 4K reduction | 16K reduction |
|---|---:|---:|---:|
| full/b1024 | 16/16 | 0.00% | 0.00% |
| full/b4096 | 16/16 | 0.00% | 0.00% |
| full/best_of_two_blocks | 16/16 | 0.00% | 0.00% |
| d27_mean32/b1024 | 16/16 | 12.40% | 14.32% |
| d27_mean32/b4096 | 16/16 | 13.90% | 15.87% |
| d27_mean32/best_of_two_blocks | 16/16 | 12.40% | 14.32% |
| d27_endpoint32/b1024 | 16/16 | 12.50% | 14.43% |
| d27_endpoint32/b4096 | 16/16 | 13.85% | 15.91% |
| d27_endpoint32/best_of_two_blocks | 16/16 | 12.50% | 14.43% |
| d30_mean16/b1024 | 16/16 | 11.58% | 13.03% |
| d30_mean16/b4096 | 16/16 | 12.46% | 13.98% |
| d30_mean16/best_of_two_blocks | 16/16 | 11.58% | 13.03% |

## Limits
New seeds, same four synthetic templates and one model/GPU. Not new-domain or architecture generalization.
Native accuracy below75% makes a cell inadequate for strong preservation claims. All cells remain reported.
Equal total task counts do not imply identical correct answers or losslessness. Confidence intervals and losses/gains are in summary.json.
Best-of-two timing is explicitly tuning on the timing set. Cross-block output differences are reported in summary.json.
A partial report is not a completed replication.
