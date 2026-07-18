# qk_mdl results

**Start here:** [EXPLAINER.md](EXPLAINER.md) (objects, shapes, reductions, the methods that
worked — with math) · [GLOSSARY.md](GLOSSARY.md) (term definitions) ·
[EXAMPLES.md](EXAMPLES.md) (qualitative examples with specifics).

One file per experiment, each with: how the model was compressed, the tables, inline
figures, examples from the decomposition, and caveats. All numbers were produced with the
Tier-0 exactness gate passing (folding reproduces the live model to ~1e-13–1e-15), under
the frozen conventions of `../mdl_accounting.py` (ΔCE = binding audit, per Logan).

0. [The methods: code + intuition + one comparison graph](00_methods.md)
1. [Gates + ground-truth battery](01_tier0_gates_battery.md)
2. [Tiny models, layer-0 MDL](02_tiny_layer0.md)
3. [The conjunction test (induction circuit)](03_conjunction.md) — the program headline
4. [546M bilinear-attention model, layer-0](04_tier2_546m.md)
5. [162M squared-attention model, layer-0](05_tier2_162m.md)
6. [Tier-3 path-folded lookups (negative)](06_tier3_pathfold.md)
7. [OV circuit + bilinear-MLP blocks](07_ov_blocks.md)
8. [Attention patterns from the compressed model](08_pattern_display.md)
9. [Grand combined: the fully codebooked layer 0 (flagship) + sqrd12 contrast](09_grand_combined.md)
10. [Layers 1-17: conditional-mean codebooks, the depth sweep, and the menu](10_layer1_condmean.md)
11. [The wall, and how windowed code propagation cracked it](11_windowed_codes.md) — current flagship
12. [Inside the window: naming the contextual core (H5 = match, H7 = transport)](12_within_window.md)
13. [The top MLPs: diffuse input, low-rank contextual output](13_top_mlp.md)
14. [Method E: backward MDL — a careful null](14_backward_mdl.md)
15. [The edge map: every module→read connection, causally priced](15_edge_heatmaps.md)
16. [Compressing the tables: three methods and a champion config](16_table_mdl.md)
17. [The context-order ladder: why token-static fails where it fails](17_context_ladder.md)

**Headlines (ΔCE at T=512, untrained unless noted):** layer-0 grand codebook (trained)
**−0.019** · full-stack score tables (trained) +0.757 = the wall · windowed code
propagation: bilin18 qk+v+mlp(L1-12) W=6 **+0.059**, sqrd12 ALL reads W=6 **+0.030**
(compressibility ranking inverts between decomposition families) · contextual core:
H5 = induction match (noisy identity payload, WW-7), H7 = rank-1 structure gain,
mlp16 = ~4-16 fast structural gains (TM-4).

Chronological detail: `../LOG.md`. Spec: `../qk_mdl_spec.md`.
