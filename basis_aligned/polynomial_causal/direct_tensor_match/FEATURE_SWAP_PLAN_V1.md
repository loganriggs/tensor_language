# Context-dependent feature interchange using the frozen scalar dictionary

2026-09-20 22:25 UTC. Removals test replacement by a fixed calibration mean.
Now test differences between actual feature values; fixed constants cancel.
The candidate, four canonical directions and residual writers remain frozen.

Use previous confirmation panels(context256), with new token-only donor maps
built by build_feature_swap_donors.py before model outputs. Recipients and donors
have positions>=16. Two families: next document at same position, and nearest
position carrying the SAME token ID in a different document. Same-token swaps
hold current-token identity fixed and probe context-dependent feature changes.
Donor selection never sees feature values or output effects.

For native projected amplitudes a and extracted scalars b, edit the original
recipient final residual by V_g*(a_donor-a_recipient)/s(h_recipient)^2, versus
V_g*(b_donor-b_recipient)/s(h_recipient)^2. Compare native final-normalized and
softcapped logit changes, token CE changes, magnitudes, all four modes and joint.
Denominator/background are recipient-native and held fixed. This is a component
interchange, not replacement of the entire upstream state or semantic variable.

Registered before swap outcomes:
- pred_a_instrument: self-donor is exact zero edit, same-token IDs equal and
  documents differ, hash checks pass; each domain's same-token coverage>=.20.
- pred_b_major: same-token modes0/1 effect cosine>.90 and error<.40 in BOTH
  domains, with native effectRMS>1e-3 (not inert).
- pred_c_all: all four same-token modes cosine>.80 and error<.65 in BOTH
  domains. Aligned-family and joint metrics also reported, never hidden.
The zero-effect/static-token predictor has relative error1 for nonzero true
same-token effects; compare to it without fitting a lookup model.

All prediction masks and effects are scored only at eligible recipient sites.
Local code sources remain related. No claim of task-specific selectivity or
monosemanticity follows from accurate operational interchange alone.
Price:shared13916scalar coefficients/10products +4608residual writer coefficients;
donor/recipient state caches are instrumentation, not learned parameters.
