# Shared causal hop-state compiler v1 — 2026-09-09 14:45 UTC

The fixed-position history-field experiment missed its strong payload/retargeting
bars and found no head with one factor per field. Preserve that null. Its anchored
score diagnostic nevertheless nominates q1/k1 in final heads 0 and 3: changing hop
has relative RMS effect about 1.05 while changing entity has effect below .082,
confirmed on short cycles. This is a lead, not a semantic identification.

Test one fixed replacement: token role (binding key/value, query marker/entity/hop/
answer) and most recently visible hop marker, including a no-marker state. Compute
one causal 30-state code sequence and reuse it for q1/k1 in both nominated heads.
Fit unrotated 32-vector means for each state/head/projection on 64 cycle documents,
seed6909, with no output fitting. Freeze/save/hash those means before held-out text.
Retain heads1/2 projection rows, all q2/k2/V, native RMS normalization, cached RoPE,
and prefix computation. Remove the replaced weight rows physically. Unseen states
are an instrument failure, not silently zeroed predictions. Charge all30 states,
including unreachable entries, both books and the whole opaque background.

Held panels:16 cycle docs seed6910;16 short-cycle permutation docs seed6911;
16 cycle docs seed6912 with only16 query blocks (112 tokens including final label).
Score every input position and all query positions separately. Four arms: full,
final head0 removed, head3 removed, both removed, implemented by zeroing those q1
coordinates identically in native and compiled execution. No example filtering.

- A instrument: causal parser/scatter/direct-vs-folded synthetic controls; original
  native/folded full-logit replay at atol=rtol=1e-9; every evaluated state seen in fit.
- B prediction: full29-way native-to-candidate KL mean<=1e-3 and p99<=1e-2 in every
  population, arm and position subset. Native model correctness is not an admission filter.
- C manipulation/composition: native-corresponding centered full-logit removal
  vectors relative RMS error<=.01, or absolute RMS<=1e-8 if native RMS<1e-6,
  for each single and joint arm in every panel. Both native singleton effects live
  (RMS>1e-6); report joint additivity separately, which final affine readout permits.
- D structural reduction: charged arbitrary constants below the prior folded
  program387968, with selected original rows absent. Expected375424: remove16384
  weights and add3840 constants. Parser has one integer state/token shared by4 readers.

Failure closes this fixed semantic code; no state-feature, head, rank or threshold
sweep. Passing licenses portable export and fresh field counterfactual confirmation,
not the full bilin18 claim. All computations begin with tokens and charged weights;
no teacher activation cache at evaluation. GPU through managed bqrunner only,
batch4 FP64, 1800s hard guard, each analysis tensor<256MiB. Reuse existing scoring
and source-readout implementation. This is the small 400640-parameter checkpoint.
