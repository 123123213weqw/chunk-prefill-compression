# Answer-blind chunk-count sweep; frozen before running

Mechanism study on the previously exposed three v5 Full-correct/d30_mean16-wrong narrative16K examples and three selected correct controls. This is not a new evaluation cohort or a deployable compression selector.

Fixed model/runner: Qwen3-4B-Instruct-2507, RTX4080, FP16 SDPA, batch1, 1024-token prefill blocks, 256 T chunks of64 tokens, merge at layer30 to16 for selected chunks; other T chunks64; S/R/Q/decode untouched; RoPE retains original logical group-end positions. Full baseline and all-compressed endpoint must exactly reproduce historical greedy token IDs, including control probe. Identity gate and old engine equivalence must pass or run stops.

Counts: 0,16,32,64,96,128,160,192,224,240,256. For each intermediate count, four nested, answer-blind chunk orders:
1. prefix first: chunk indices0..255;
2. suffix first:255..0;
3. random-A: Python random.Random int(SHA256('v7:A:'+sample_id)[:16],16), shuffle0..255;
4. random-B: same with'v7:B:' instead of'A:'.

Both random permutations fixed before outcomes, with independent seeds for each sample. Order selection does not inspect question, gold or answer; it is a measurement control, not a proposed optimal route. The 0 and256 endpoint results are shared across all four paths (not re-counted as independent trials). Queue has6×(2+9×4)=228 atomic units. No adaptively added counts in this run. No early stop on errors; all cases and orders reported.

Primary descriptive outputs: first failure count (if any), full sequence of correctness over counts, number/direction of correctness transitions, and whether trajectories are non-monotone. Secondary: generated IDs, actual route keeps, saved T layer-tokens, control-probe correctness, per-case task output. `first failure count` is only a grid point, not a statistically estimated true threshold; a later recovery does not erase a prior failure. Random orders are not replicates of independent documents. No new clean timing claim; instrumented seconds are diagnostic only.

Exclusive GPU required; reject foreign GPU processes before each unit; atomic files; maximum 2h wall time. The previously stopped llama-server must not be restarted by this experiment. No other service modified.
