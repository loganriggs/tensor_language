# Conditional nonlinear number / linear modal predictor

The fixed eight-feature state reconstruction fails source reuse, but the failure is not uniform across outputs. The v686 baseline derivative predicts modal effects accurately while missing source B's nonlinear number effect. This suggests separating the observable computations rather than insisting that one compressed residual state represent them all.

For post11 source response delta, frozen encoder W and decoder P, let q be the four gradients of the final softcapped contrasts at the unchanged native context. The tested hybrid is

- number effect: sparse nonlinear prediction minus direction * q_number dot (delta - P W^T delta);
- modal effects: -q_modal dot delta, for can/will, may/might, should/could.

The number reader is folded once as q_number * (I-P W^T); runtime shares the resulting fields across A, B and their sum. These are gradients at the **baseline only**: no edited endpoints or labels are used to generate them. Their native computation and storage remain charged. The nonlinear branch retains six full conditional attention stages and sparse bilinear MLP cores, with RMS and softcap explicit.

## Baselines and falsifiers

v686 uses the opened48source rows. Central finite-change validation of the analytic baseline gradient passes (relative4.89e-4 against1% bar). Pure tangent number error reaches25.4% for B, failing10%; modal errors are0.20% A and1.06% B, passing5%. Correcting the sparse predictor in all four outputs still fails modal gates. Combining corrected nonlinear number with pure tangent modal branches passes only as a **post-hoc candidate** at this point.

The discarded initial-response effects are approximately additive across A/B: lost(sum) versus lost(A)+lost(B) has pooled relative errors13.54% for number and3.16%,2.37%,1.38% for modals. This is a response-pattern observation, not discovery of a unique feature or computation.

## Frozen fresh-combination test

v687 freezes48new rows before model evaluation, with guard/judge/clerk/coach/chef/cook subjects and new object-relative/fronted-preposition combinations. Exact rows are disjoint from v667. Historical and calibration vocabulary overlap is not excluded. SHA256: c0b8214f736ffc4aa77016a9364ac740b80c2a07f7f1010427a484577264e6e0.

| Source | Maximum number prediction error | Maximum modal prediction error | Maximum native modal collateral |
|---|---:|---:|---:|
| A | 4.66% | 0.223% | 9.73% |
| B | 7.31% | 0.789% | 9.44% |
| post11 sum | 4.95% | 0.942% | 7.27% |

All cells have100% native number capability. Registered A/B gates pass: target10%,modal prediction5%,native collateral10%,capability90%. Sum results are reported separately and also meet these numerical thresholds. Native collateral means the underlying tested source intervention changes those controls by this amount; the output predictor itself is not an installed model intervention. This is one fresh combination panel, not universal OOD evidence.

Metadata clarification: inherited v687/v688 scope prose mentions older opened rows/templates. The frozen row file/hash and experiment-specific predicates above are authoritative. v688 repeats the now-opened v687 panel for export, not another fresh test.

## Extraction and literal cost

v688 exports the first singular/plural row per template, eight texts and24source cases. A/B/sum share prepared context and readers. `check_hybrid_response_export.py` replays on CPU without checkpoint/CUDA: maximum difference2.08e-15; zero-response floor2.94e-6 from native float32 baseline-margin preparation.

- Artifact:28,750,173bytes, `subject_hybrid_v688.pt`.
- Sparse runtime:1,934values (including pair indices); initial encoder:9,216values.
- Prepared cases:3,553,136tensor values, including294,912reader values,221,184full source-input values,96reference values.
- Native context, source-port and gradient generators remain external. The artifact is an extracted conditional **output predictor**, not a token-input model or a coherent replacement residual state.

Independent reader generation needs four batched reverse passes over the six native suffix blocks for the four template batches (24attention pullbacks/48reference attention forwards), in addition to baseline generation. Small runtime size alone is not evidence of model compression.

## Simplicity ablation and next step

A post-hoc audit removes only the number kernel correction and keeps sparse nonlinear number plus three linear modal readers. It still passes the output thresholds on both opened v686 and now-opened v687 results. On v687, uncorrected number errors are6.31% A,7.98% B,5.11% sum. This removes one of four reader fields and its gradient; it needs prospective confirmation before replacing the frozen tested formula.

Next useful folding test: prune actual backward attention/MLP derivative branches for the modal readers and compare their predictions against full gradients and native effects. This targets fewer required computations and context ports. Changing coordinates or reporting only the1934runtime values would not remove the dominant dependencies.

Primary receipts: `subject_attention_freeze_v686_result.json`, v687, v688; CPU audits `DISCARDED_RESPONSE_COMPOSITION_2026-09-20.json`, `HYBRID_RESPONSE_CPU_REPLAY_2026-09-20.json`, `HYBRID_NUMBER_CORRECTION_ABLATION_2026-09-20.json`. Previous failed state-reuse line remains in `SOURCE_REUSE_AND_FULL_READERS_2026-09-20.md`.
