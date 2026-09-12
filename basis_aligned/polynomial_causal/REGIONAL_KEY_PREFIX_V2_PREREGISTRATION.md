V2 corrects a pre-native count assertion: the fresh two prefixes overlap the original panel. All112 rows and original numerical bars remain unchanged. V1 failed before model load.

# Causal prefix replay and contextual joint-key update families

Use all 112 rows in corrected source V2, geographic OOD V1 and fresh controls V2.
Twelve distinct prefixes end at their sole changed cue token. No new labels,
fit, behavioral selection or altered rows. Execute each prefix independently,
and compare both normalized/rotated key factors jointly with full-prompt keys
at the same absolute position, for attention layers 8, 9 and 13.

For each prefix and layer, expand the raw incoming residual exactly into the
zero-input background, actual embedding re-entry, and scaled attention/MLP
update differences from the zero-input trajectory. Keep both QK factors together.
Compare anchor alone, anchor plus all attention differences, anchor plus all MLP
differences, and the exact sum. Include the shared native input RMS and both key
normalizers; no denominator freezing. Joint-key error is Frobenius error in the
outer product of the two key vectors, aggregated over heads within a prefix.

- pred_a: maximum full/prefix joint-key error, raw residual replay error and
  reconstructed joint-key error are all <= 2e-5.
- pred_b: pred_a and maximum attention-only restored joint-key error <= 0.1.
- pred_c: pred_a and maximum MLP-only restored joint-key error <= 0.1.

The last two are opposing sufficiency screens; neither is assumed to pass.
Null: both update families are needed. A miss concerns this declared interface,
not general weight-only factorization. Numerical failure of pred_a invalidates
family interpretation. Preserve every prefix and report worst-case errors.
Counterfactual updates are not propagated recursively: this is attribution of
the key-input expression, not whole-model removal or downstream behavioral proof.

Price: <=40 native batches through14blocks, 180second managed cap, no logits,
no optimizer, <10MB artifact. Save the scaled residual update parts to support
subsequent reader folding. Frozen helper/checkpoint/row hashes bind execution.
