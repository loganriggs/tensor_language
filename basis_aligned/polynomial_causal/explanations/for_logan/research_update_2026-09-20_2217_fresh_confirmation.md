# Fresh confirmation: better fidelity, incomplete feature transfer

2026-09-20 22:17 UTC

The preregistered50/50 Gaussian/empirical moment blend was frozen and tested on
32 unused FineWeb documents and16 new local code files, all atcontext256.
All comparisons retained the real normalization, residual background and softcap.
The Gaussian-only readout has the same21,948coefficient/10product cost.

| Domain and program | CE added above native | KL from native |
|---|---:|---:|
| FineWeb smaller base | .01399 | .01702 |
| FineWeb Gaussian-only skip | .00822 | .01098 |
| FineWeb frozen50/50 skip | .00486 | .00720 |
| Code smaller base | .03002 | .06343 |
| Code Gaussian-only skip | .01993 | .04770 |
| Code frozen50/50 skip | .01901 | .03143 |

The instrument passes. The feature-confirmation bar FAILS: code feature1
removal-effect error is40.002529%, above the strict40% cutoff. Its95%
document-bootstrap interval is[33.45%,46.48%], so this tiny threshold miss
should not be overstated as a sharp qualitative difference. The strong
preservation bar also FAILS: codeKL remains above.02, with interval
[.02896,.03407]. Failures are preserved without changing tolerances.

The improvement relative to the Gaussian-only control is more robust than
absolute preservation: paired codeKL difference has interval[-.01927,-.01355].
CodeCE difference includes zero; do not claim a reliable CE advantage there.
FineWebCE difference is[-.00517,-.00153]. Code sources remain related, limiting
corpus-level inference despite their exclusion from earlier panels.

All-mode removal errors for the frozen primary are:

| Mode | FineWeb | Code |
|---|---:|---:|
| 0 | 10.3% | 18.7% |
| 1 | 29.0% | 40.0025% |
| 2 | 35.4% | 31.0% |
| 3 | 39.5% | 42.0% |
| Joint | 15.4% | 21.3% |

A follow-on CPU screen asks whether position alone explains the strongest
mode. On a reused256-token panel, its native position-mean variance fraction
is3.7%, versus3.1% under independent document circular shifts; correlation
with log position is.072. The first8 positions carry only2.0% of centered
energy. This does not suggest a dominant simple position explanation, but the
exploratory null is not definitive and gives no semantic label.

The current result is a better extracted polynomial branch with partially
predictable removals, not a complete selective/reusable semantic circuit.
The full goal remains active.

[Confirmation](../../direct_tensor_match/BLEND_CONFIRMATION_V1.json),
[uncertainty](../../direct_tensor_match/BLEND_CONFIRMATION_UNCERTAINTY_V1.json),
[position screen](../../direct_tensor_match/MODE_POSITION_AUDIT_V1.json).
