# Candidate: letter-continuation group 2

Status: **behavioral candidate; not adopted or fully identified**. Updated2026-09-21.

## Exact executable boundary

Use `half0`, component2 of `MIDPOINT_STABLE_GROUP_REMOVAL_PROGRAMS_V1.pt` (SHA256 recorded in the frozen native plans). Let $h$ be the original last-MLP input, $m_0$ the preceding MLP polynomial output with last-block residual scaling, and $s(h)$ the original recipient RMS denominator. Define

$$
n=(h-m_0/2)/s(h),\qquad m=m_0/s(h).
$$

The extracted contribution is

$$
u=a^\top n-\alpha,\qquad v=b^\top m-\beta,\qquad
\widehat y=wuv.
$$

Here $a,b,w\in\mathbb R^{1152}$ and $\alpha,\beta$ are frozen calibration centering constants. Output $\widehat y$ uses the exact QR-reduced vocabulary frame. If $U=QR_U$, the residual write is $R_U^{-1}\widehat y$. Removal subtracts this write from the original final residual, followed by native final RMSNorm, unembedding and softcap.

Local specification: one variable product, two dense readers and one dense writer (3,456 weights plus two centering constants), with the readout-frame mapping shared. This price excludes upstream computation, normalization and the native background; it is not a whole-model extraction or acceleration claim.

The term arose from a six-product local approximation to eight terms of an earlier512-product folded-path approximation. Therefore, it is not automatically a unique exact summand of the original native operator. Its closeness to a corresponding original-weight observable needs direct testing.

## Evidence ledger

| Requirement | Evidence | Remaining gap |
|---|---|---|
| Computational specification | Frozen scalar readers/product/output writer above; legal residual intervention | Upstream production still supplied by the model |
| Weight-derived discovery | Folded source-dependent path; output-sharing factors, joint local refactor | Multiple approximation stages; native-observable grounding needed |
| Stable identification | Two converged covariance-half fits have 1.47% removal-effect disagreement on16 unused documents | Teacher and initial selection used all calibration rows; broader split/gauge/OOD robustness absent |
| Held-out prediction | Frozen continuation hypothesis passes32 new FineWeb rows | Same corpus family; no broad OOD result |
| Selective manipulation | Removal harms alphabetic-continuation examples while spaced-word average CE stays near zero; norm-matched writer control weaker | Conditions use observed next tokens; broader collateral tasks and controlled minimal pairs absent |
| Extraction/sufficiency | Standalone executable contribution given native inputs | No independent recovery of upstream inputs; no unique native-unit sufficiency claim |
| Composition/reuse | A scalar feature and write can be evaluated independently | Joint deployment with other extracted circuits untested |

## Operational behavioral hypothesis

Among prefixes ending in an alphabetic character with no pending UTF-8 bytes, the group supports a following token that begins with an ASCII letter without leading whitespace. Contrast: a following token that begins with whitespace followed by an ASCII letter. These labels are evaluated using the realized next token; they are not available as an input to the candidate.

On32 new FineWeb rows, removal CE added is0.12555 nats/token on514 continuation sites and−0.00021 on4787 spaced-word sites. A control uses the same scalar activation with another component's writer, matched in vocabulary-centered norm; it causes0.05249 continuation CE damage. The real direction decreases bare-versus-spaced-letter logodds by2.31797, versus1.16971 for the control.

A same-current-token stratified comparison covers34 token types,51 continuation sites and257 spaced sites. The min-count-weighted effect contrast is1.59844 logodds, with paired-document bootstrap interval[0.85836,2.02112]. This is evidence of context dependence beyond current token identity, not control of all context or difficulty confounds.

Do not label this monosemantic, a general word-boundary detector, or a UTF-8 circuit. Those broader claims are not established. Static vocabulary writer energy is diffuse (9.81% in its top256 coordinates), so a handful of top tokens cannot define its meaning.

## Next decisive checks

1. Compare the extracted feature with an original-weight observable along its output direction, rather than only against its approximate teacher.
2. Test a frozen prediction on code or controlled tokenization examples, keeping current-token identity and perturbation magnitude explicit.
3. Fold the scalar readers backward through the preceding MLP and test whether their production can be extracted more simply without losing the behavioral effect.
4. Check composition, interventions on unrelated behaviors and robustness across discovery splits before adoption.

Evidence: `MIDPOINT_STABLE_GROUP_REMOVAL_NATIVE_V1.json`, `MIDPOINT_STABLE_GROUP_PROFILE_V1.json`, `MIDPOINT_STABLE_GROUP_BOUNDARY_AUDIT_V1.json`, `MIDPOINT_CONTINUATION_GROUP_NATIVE_V1.json`, and `MIDPOINT_CONTINUATION_GROUP_AUDIT_V1.json` in the parent directory. [Timed explanation](../../explanations/for_logan/research_update_2026-09-21_0346_continuation_candidate.md).
