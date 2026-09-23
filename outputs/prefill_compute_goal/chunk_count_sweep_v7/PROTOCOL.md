# v5 failure-localization diagnostic

This is a post-hoc, answer-aware diagnostic of an exposed dataset, NOT a new independent evaluation or deployable adaptive algorithm. Do not overwrite the original v5 run.

## Frozen choices before intervention outcomes
- All three Full-correct/d30_mean16-wrong b1024 documents, plus three unique correct controls chosen by select_cases.py. There are only two same-label Amber controls; the third has Dune gold. Matching is descriptive, not randomized inference.
- Exact original model/tokenizer, FP16 SDPA, 64-token T chunks, depth30, default keep16, block1024. S/R/Q/decode unchanged.
- Locate whole warehouse/box evidence sentences using tokenizer offset mapping. Include every intersecting chunk; publish boundaries, token group offsets and original logical RoPE anchors.
- Intervene on each evidence chunk separately, warehouse-sentence union, and all-evidence union: keep32 or keep64. keep64 performs identity pooling, original positions unchanged. Other chunks remain keep16.
- Each intervention receives a same-count non-evidence placebo (nearest preceding eligible chunks; deterministic fallback forward), with identical layer-token budget. Duplicate chunk sets across named groups are retained transparently, not counted as independent evidence.
- Re-run native/manual Full gate per document and require exact historical greedy-ID reproduction for both Full and compressed task/control probes. Verify compressed baseline with the frozen legacy engine too. Halt rather than interpret a nonreproducing baseline.
- Record per-layer cache shape, 7 projection/MLP input shapes, actual route keeps, saved T layer-tokens, generated IDs, parsed gold scoring and canonical correct vs wrong-warehouse Dune (or Amber for Dune control) likelihood margin.
- No clean performance measurements: instrumented prefill seconds are diagnostic only. No causal claim that isolates RoPE versus mean pooling from a keep intervention alone.

## Stop / resource rules
Do not kill foreign GPU workloads without user approval. Before every unit reject foreign GPU processes. GPU jobs require an exclusive available device. Maximum2hours; atomic per-unit records and failure status. No automatic waiting scheduler or automatic takeover.

## Interpretation
Report all cases and placebo outcomes. Recovery of failed cases is not accuracy gain on an independent set. A local rescue supports local intervention sensitivity, not a claim of mathematically lossless compression. No rescue may indicate distributed effects, weak base-model evidence retrieval, or insufficient intervention; it does not prove information is absent.
