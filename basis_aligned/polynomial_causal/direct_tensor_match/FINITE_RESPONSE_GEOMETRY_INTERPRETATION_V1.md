**The response-transfer gap already exists in the polynomial fitting metric.**

The native pre-final audit completed in2.377s with exact model-state replay. All registered predictions passed. For the learned finite-response program, the same centered-quadratic response metric used during fitting changes as follows:

| Inputs and donor rule | Relative response error |
|---|---:|
|Historical calibration, roll257donors|3.26%|
|Opened FineWeb, token/cohort-matched continuation donors|11.96%|
|Opened FineWeb, token/cohort-matched spaced-word donors|15.03%|
|Opened code, token/cohort-matched continuation donors|7.45%|
|Opened code, token/cohort-matched spaced-word donors|10.24%|

All four native values exceed twice the fitting error, and all four improve slightly over the fixed-reader alternative (12.23/15.47/7.79/10.44%). These are exactly evaluated differences of the exported polynomial including its affine repair, measured in the uncentered-vocabulary QR metric, before final RMSNorm and softcap. Thus downstream nonlinearities cannot be the sole source of the observed transfer gap.

Changing only the denominator changes the numerical error: learned full-residual response errors are10.83/7.20/9.11/6.55%, while learned MLP-only response errors are14.39/8.66/6.25/5.61%. Do not compare these percentages as if they estimated the same quantity. The later centered-logit behavior measure also includes nonlinearities and vocabulary centering; its error ratio is not an isolated amplification factor.

This comparison changes states, corpus/context composition and donor selection together. It does not establish which change causes the gap. The CPU successor therefore holds all historical states fixed and varies source donor pairings (roll64/509/1021/1537versus fitted257), with no refitting. Large deterioration would demonstrate pair-specific overfit even before changing the state distribution; little deterioration would demote that explanation. No claim of independent textual validation follows from recombinations of old states.

[Full metric audit](FINITE_RESPONSE_GEOMETRY_AUDIT_V1.json) · [Previous finite fit and native results](FULL_QUADRATIC_FINITE_RESPONSE_INTERPRETATION_V1.md).

**Successor completed: donor-pair overfit is real but does not explain the whole native gap.** The independent CPU recombination audit ran in17.0s. Learned error rises from3.264%on fitted pairs to5.217–5.254%on four unfitted pairings of the same states (about1.60times). All registered predictions pass. Fixed-reader errors rise from4.546%to5.628–5.669%; the previous mixed tangent-fit program scores6.243–6.349%on the unfitted pairings. Thus the learned finite-response program retains an advantage on recombinations, but its apparent training advantage is overstated. The remaining7.45–15.03%native errors cannot be explained quantitatively by these four recombinations alone. No one-factor causal decomposition of the full gap is established. The actual comparison has three frozen programs (finite-fixed, finite-learned, prior mixed), correcting the board's initial four-program wording. [Recombination receipt](HISTORICAL_PAIR_RECOMBINATION_V1.json).
