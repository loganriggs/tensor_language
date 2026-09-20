# Learned-direction responses transfer better than arbitrary responses

2026-09-20 20:43 UTC

Frozen candidates were tested on 128 fresh input rows. Perturbations followed
eight learned quadratic readers or eight seeded random directions, with matched
pre-normalization lengths. Every perturbed input was returned to native radius.
Three amplitudes and both signs were tested; there was no fitting.

| Direction family | Quadratic response error | Quartic response error |
|---|---:|---:|
| Learned | 23.96–24.44% | 18.68–20.36% |
| Matched random | 47.25–49.63% | 48.45–51.77% |

The metric compares changes in native and candidate outputs, so constant bias
cancels. Learned-direction response cosines exceed .969 for the quadratic and
.979 for the quartic. All registered bars passed, including response precision
below 1.2e-4. Random directions reveal a substantial limitation: neither program
is a faithful local replacement in arbitrary directions. Learned directions
were drawn from the quadratic candidate itself, so this advantage is not an
unbiased test of general response fidelity. Radius normalization also changes
realized perturbation lengths. These are folded-function input counterfactuals,
not semantic or whole-model causal interventions.

The next structural step returns to joint tensor decomposition: the quartic
program computes four quadratic bank features using 16 primitive products before
its ten root products. An exact 4×32×32 bank tensor has now been extracted and
replayed to 3.2e-15. Root-weighted input-mode energy concentrates strongly, but
this does not prove a low CP rank or preserve the composed quartic automatically.
The registered next fit shares 8 or 12 primitive products across all four bank
features and checks the complete quartic afterward. This directly tests whether
cross-feature reuse can reduce the program, rather than only its output storage.

[Response receipt](../../direct_tensor_match/DIRECTIONAL_RESPONSE_V1.json),
[exact bank core](../../direct_tensor_match/QUARTIC_BANK_CORE_V1.json),
[next refactor](../../direct_tensor_match/QUARTIC_BANK_REFACTOR_PLAN_V1.md).
