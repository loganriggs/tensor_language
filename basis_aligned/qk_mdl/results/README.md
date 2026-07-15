# qk_mdl results

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

Chronological detail: `../LOG.md`. Spec: `../qk_mdl_spec.md`.
