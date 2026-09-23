# Inverse localization protocol (frozen before new outcome inspection)

This run is post-hoc, answer-aware and confined to the exact 3 failed / 3 correct controls from completed v5 diagnosis. It is a within-case mechanism intervention, not fresh validation or a deployable selector.

- Same Qwen3-4B-Instruct-2507, RTX4080, FP16 SDPA, 1024-token block, layer30, 64-token T chunks, greedy decoding and strict JSON scoring.
- Reproduce native/manual Full and original 64→16 all-T-compressed baselines per document, including identity gate, generated IDs and old engine equivalence.
- Invert previous retention test: **all T chunks stay64 except selected group(s) compressed to16**. S/R/Q unchanged; original logical positions preserved as in engine.
- For each document test warehouse evidence sentence (one chunk) and warehouse+boxes evidence (two chunks). Each has a same-number neighboring non-evidence chunk placebo. Selection is frozen in copied prepared.json. 4 conditions × 6 documents =24 intervention units, plus12 baseline units =36 total.
- Equal *compressed* layer-token budget for evidence vs placebo within each group. Previous study tested equal *preserved* budget with most chunks compressed. The pair of studies isolates local-only versus background-plus-local interactions descriptively, but not cleanly disambiguates content, RoPE anchoring, attention normalization or floating-point effects.
- Atomic unit records and frozen code. Two-hour wall-clock cap, exclusive GPU required. No clean timing claim.
- Predefined interpretation: if evidence-only compression fails while placebo-only compression remains correct, local evidence is sufficient to induce that error in this context; if both correct yet global compression fails, distributed/context interactions are required. If placebo-only fails too, the loss is not evidence-specific. These are per-case mechanistic descriptions, not statistical generalization.
