# Rung 413 preregistration — original-native signed gate for the physical BF16 repair

## Decision

Rung 412 physically repaired the 43-certificate program: every generated MLP0/4 tensor is BF16, every QK factor is
fp16, the source-native tensors use their frozen BF16 treatment, and the complete object occupies 1,023,517,292
bytes. Prediction, certificates, shifted text, and fresh windows passed. This rung asks whether that exact stored
artifact preserves the direction and relative magnitude of a named causal intervention.

## Frozen intervention and predictions

Replace the output of attention layer 16 by its native mean write. Measure the per-position signed effect as
cross-entropy after knockout minus cross-entropy without knockout. Compare the compiled effect vector with the
original-native effect vector measured before global rounding.

- pred_a: reproduce rung 412 at census damage at most .015, at least 43 certificates, shifted maximum at most .120,
  and fresh maximum at most .030; assert all generated MLP tensors BF16, all QK factors fp16, exact source treatment,
  shapes/counts, and 1,023,517,292-byte price.
- pred_b: signed-effect cosine at least .98, normalized vector error at most .30, and norm ratio in [.90,1.15].
- pred_c: collateral circuit-effect Spearman correlation at least .98 and the median magnitude ratio on
  attention-16-owned circuits in [.90,1.15].

Strong null: cosine below .70, collateral correlation below .75, or any physical identity failure.

All three predictions with the null false restore formal adoption of the literal all-two-byte 43-certificate tier.
Failure retains rung 412 as a predictive/OOD artifact only. No dtype, rank, layer, intervention, or threshold tuning.
