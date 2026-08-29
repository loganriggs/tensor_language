# MLP1 bilinear CMR discovery: preserved no-run decision

Date: 2026-08-29 UTC

The protocol in `MLP1_BILINEAR_CMR_DISCOVERY_PREREGISTRATION.md` was frozen and
pushed before model outcomes were opened.  It was then stopped **without a GPU
run** after a source and prior-result audit found substantial duplication with the
completed MLP1 global-gate response assay.  The preregistration remains unchanged
as a record; it is not an outcome and must not be queued.

## Why it was stopped

The completed MLP1 assay had already compared a trajectory-complete downstream
response selector with response energy, an activation-times-`Down` selector,
deranged-factor controls, and random controls at equal native-channel budgets.  It
found stable gate identities but `no_admitted_support`: the proposed native-gate
sets did not reliably beat all controls across validation halves and execution
modes.

The planned diagonal score was

$$
s_j=\operatorname{Var}(a_j)\lVert D_{:j}\rVert_2^2.
$$

The spent assay's activation/Down score was

$$
r_j=\sqrt{\mathbb E[a_j^2]}\lVert D_{:j}\rVert_2.
$$

Therefore

$$
r_j^2=s_j+\mathbb E[a_j]^2\lVert D_{:j}\rVert_2^2.
$$

The new rule centers the activation, but it remains the same broad local
immediate-write family.  Running another MLP1 native-channel screen would not be a
genuinely new mathematical move unless the centering term first predicts a
materially different, consequence-superior support.

## Terminology correction

$s_j$ is invariant to reciprocal rescaling of one fixed native bilinear channel,
and its joint form measures immediate MLP write distortion.  It is **not** final
logit distortion through the transformer suffix.  RMSNorm, residual additions,
attention, and later MLPs make that suffix nonlinear, state-dependent, and
cross-position coupled.

Consequently, the CMR margin certificate must use the actual joint squared
post-softcap-logit difference under the finite compiled intervention.  Local
`Down`-space error can select a candidate or diagnose covariance, but cannot be
inserted into that certificate.  The invariance claim is also limited to native
channel rescaling/permutation; it does not quotient arbitrary CP
splitting/merging or alternate factorizations.

## CPU action and result

`audit_mlp1_cmr_duplication.py` opens only the hash-pinned spent MLP1 bundle and
result.  It reports support overlaps and score-rank correlations between the
activation/Down control and the prior response, Fisher/leverage, deranged, and
random selectors.  Its receipt is `mlp1_cmr_duplication_audit_results.json`.

## Revised highest-return experiment

The smallest nonduplicated site is MLP2, for which no equivalent global-gate assay
exists.  The first version should be narrower than the rejected MLP1 protocol:

1. use the native upstream trajectory;
2. freeze one retained budget and constant mean-folding only;
3. compare a full-suffix joint/Fisher selector with centered immediate-write mass,
   uncentered activation/Down mass, invariant weight mass, random, and a
   document/probe-deranged null;
4. evaluate actual finite final logits, CE, KL, top-1 agreement, collateral cells,
   and signed small-edit calibration;
5. compute the margin certificate from actual finite final-logit distortion and
   native margins, never from the local score;
6. if it passes on native rows, cross the same frozen program with C512 before
   claiming a modular MLP0-to-MLP2 interface.

This MLP2 experiment still requires fresh source-document-disjoint row roles and a
frozen simultaneous-inference contract.  Existing MLP1, C512-interchange, and copy
rows are spent for promotion.

