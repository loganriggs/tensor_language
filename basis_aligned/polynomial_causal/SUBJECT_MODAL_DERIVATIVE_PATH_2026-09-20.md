# A smaller modal reader path

The conditional output predictor's modal branch can omit attention derivatives13–17 on the tested source interventions. It retains all six MLP derivatives and attention12. This changes the predictor, not the native model: no primal attention module is ablated.

At a fixed native baseline define M_l=(I+J_MLP,l)^T and A_l=J_attention,l^T. The full backward reader is the ordered product of lambda_l*(I+A_l)*M_l applied to the final RMS/softcap reader. Expanding by the number of attention derivative factors gives an all-MLP term, six terms with exactly one A_l, and terms containing two or more A_l. This degree counts derivative branches, not polynomial degree in the original input. All normalization derivatives stay explicit.

`single_attention_terms` computes the first two groups with shared intermediate contractions. CPU tests compare each individual term plus the zero-attention term against an independently selected single-attention path, within1e-10.

## Negative and positive screens

On the opened v687 panel, v689's carry-only, MLP17-only, last-two-MLP, all-MLP and attention-only readers all fail the5% modal prediction bar. All-MLP is closest: A3.80%,B5.31%. This negative remains valid.

v690's sum of zero/one-attention terms passes (A0.506%,B1.052%). Individual attention12,15,17 corrections also pass. Attention12 is frozen for prospective testing because it is the earliest passing layer: every MLP pullback can remain at the readout token until the attention12 pullback spreads the reader across input positions. No uniqueness claim follows from several passing paths.

## Prospective validation

v691 freezes48longer sentence combinations, reusing nouns but changing the relative-clause/preposition combinations. Hash c38db7a51d9f958e5020d5c921af04b0fe14a3f4a8be6a2ff88116e9bce8c62a. The number branch is the uncorrected sparse nonlinear predictor; the modal branch uses only attention12 and all six MLP derivatives.

| Source | Maximum number error | Maximum modal prediction error | Maximum native modal collateral |
|---|---:|---:|---:|
| A | 6.57% | 3.36% | 6.59% |
| B | 5.66% | 2.48% | 5.50% |
| sum | 5.24% | 2.67% | 4.82% |

Native capability is100% in all cells. Registered A/B gates pass. The sum also meets these numerical thresholds. Pure full-gradient tangent prediction passes on this particular panel, although it fails the earlier source-B number tests; this panel alone does not establish a need for the nonlinear branch.

The optimized reader matches the generic selected-path implementation on native contexts to3.80e-19. CPU tests additionally alter unused later attention inputs, non-readout MLP inputs, and non-readout final states; the optimized output remains unchanged within1e-10. The tests also retain full-reader versus autograd and finite-change closure checks.

## What became cheaper

For the modal reader generator, six full attention pullbacks become one, and six full-sequence MLP pullbacks become six token-local MLP pullbacks. Three modal readers replace the four-reader corrected hybrid; the number correction was removed before this prospective test. This is an algebraic reduction in the required derivative computations, not a measured end-to-end speedup.

Required baseline ports for this reader are the final readout state, six pre-MLP states at the readout token, the raw input to attention12 at all positions, and cached first-layer values. Ignoring rotary/scalar metadata, these contain (7+2T)Bd values, versus (1+13T)Bd for the minimally retained full modal-gradient generator. This is a dependency count; the screen scripts retain extra data for controls.

All six native MLP weight sets and the attention12 weights remain required by this generator. More importantly, the **number branch still needs its six native-prepared attention/MLP contexts**. Thus this does not remove those dependencies from the whole predictor or eliminate native baseline generation. The selected reader has not yet been exported as a replacement for the earlier full-reader artifact v688.

The result is a prospective conditional output-path simplification, not a standalone token-input model, a unique semantic circuit, or a verified residual-state replacement. Next tests should establish amplitude/composition robustness and portable execution of this smaller path before further pruning or claims of intervention adoption.

Primary receipts: `subject_attention_freeze_v689_result.json`, v690, v691. Implementation: `selected_suffix_readers.py`; CPU checks: `test_full_suffix_readers.py`. Earlier exported conditional predictor: `SUBJECT_HYBRID_RESPONSE_2026-09-20.md`.
