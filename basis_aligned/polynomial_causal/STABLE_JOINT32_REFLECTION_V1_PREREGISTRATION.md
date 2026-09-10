# Two unsupervised products: conditional reflection screen

Prior comparison: STABLE_JOINT32_PRIOR_V1_AUDIT.json. The two products are the only
mutual matches passing the frozen joint32 rule: native indices 0/9 and random 29/4.
They were discovered without token labels or activations. Pronoun interpretation
was assigned afterward; this screen evaluates that post-discovery hypothesis.
Existing gender-axis work is prior art, not a new discovery claimed here.

Rows: first 96 rows, first 257 tokens of census_state_diverse.pt, frozen separately.
Score next-token positions 64..255. All rows retained. These historically opened
corpus rows are not fresh/OOD, though no activation was used to fit these products.
Pronoun targets are the six-token he/she/they/He/She/They class used by old work.

At normalized MLP17 input x, v is the frozen smallest eigenvector of the old
pronoun-class form reconstructed in FP64. Reflect about ZERO, Jx=x-2v(v^T x).
Keep the incoming residual and all upstream computation unchanged.

Reference: replace the MLP17 output with native B(Jx). Candidate: replace only the
two-product component P(x) by P(Jx), retaining native B(x)-P(x). Evaluate native-
initialized and random-initialized stable pairs separately. Removal subtracts P(x)
with all native background retained. Final normalization and tanh remain live.

- pred_a_instrument: exactly 13 model forwards/104 sequences; at least 20 pronoun
  target positions; independent full-hook versus final-state reflection replay
  max absolute logit error <=1e-3 and relative L2 <=1e-5; exact FP64 product delta
  identity relative error <=1e-10; finite metrics and parameters. Independent
  class reflection must have mean absolute target-CE effect >=.05 nats to be live.
- pred_b_conditional_sufficiency: for BOTH fixed pairs, centered full-valid-vocab
  reflection-response relative L2 error <=.20 at pronoun targets AND relative
  per-position target-CE response error <=.20. No sign cancellation in errors.
- pred_c_selective_removal: for BOTH pairs, mean absolute target-CE change under
  removal >=.05 on pronoun targets and <=.01 on all other target positions.

Null: weight-stable products correlate with known pronoun readout but do not
reproduce its conditional causal operation or support selective removal.
Failure closes the two-product sufficient-operation claim at these fixed bars;
do not refit rank/initialization or filter rows to rescue it.

Price: 12 baseline forwards of 8x256 inputs plus one full-reflection-hook replay
on the first batch, 13 forwards/104 sequences total; 12 extra direct MLP evaluations,
four local readout variants per batch (two reflection, two removal), and reference
readout. All native weights remain. No independent input producer or extraction
claim. Store row-level sums and counts, compact frozen program and token rows;
no large logit caches. Managed lane1 only, 900s hard bound.
