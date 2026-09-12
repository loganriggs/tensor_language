# Frozen regional producer group: held-out test

Group attention8, attention9, attention13 selected once from the32 original
regional prompts. The saved joint CPU audit gives43.04/55.90%whole-child transfer.
No group outcomes from the48 geographic/spelling holdout rows were inspected.
Whole-block and child-role outcomes on these rows were inspected previously;
this is held-out producer-role validation, not historically untouched text.

Reuse exact residual unroll, native positional gates and frozen source factors.
Each single numerator port swap keeps recipient RMS; the group is the sum of
its three write deltas. Recompute the nonlinear native suffix after combining.
A: projected residual, child, native attention and full-child donor replay all
<=1e-5 relative. B: A plus whole-child transfer>=10%native cue gap and>=4/6
positive pairs in EACH of four families. C: A/B plus group transfer>=40%whole-child
transfer,>=4/6positive pairs, unrelated meanabs<=.5regional meanabs EACH family.
Null: discovery-template group does not generalize. Preserve failures; all other
producer scores remain descriptive. Existing embedding/MLP16 dominance misses
remain unchanged. This is conditional numerator mediation, not full-network
module ablation or extraction without the native background.

Price:10 body forwards,48 sequences of6–14tokens,batches<=8,40 suffix arms,
180sec managed GPU cap,no optimization. Native same-row references bound.
Reuse projected hooks; save writes/margins only (~18MB), no full vocab arrays.
