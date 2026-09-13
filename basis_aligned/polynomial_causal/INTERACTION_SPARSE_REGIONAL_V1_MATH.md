# Does the sparse mixed interaction preserve its circuit contribution?

13 September 2026, 06:22 UTC. Six weight-only sparse candidates passed the registered conditional regional screen. A deletion control shows why the result needs a narrower interpretation: this mixed term supplies only a small part of the compact predictor's effect.

## Computation and boundary

Let $z$ be the cached MLP-only pre-MLP17 state and let $Wa$ be the compact head write. Recover $a\in\mathbb R^{128}$ from the difference of the cached compact and MLP-only linear states, with fixed $W\in\mathbb R^{1152\times128}$. Native rounding leaves a relative write reconstruction residual of 0.0489%. The test uses the projected write consistently for the exact and approximate mixed operators.

The selected-output mixed numerator is

$$
M_o(z,a)=\sum_{ia}T_{oia}z_i a_a
=\left[UD\big((Lz)\odot(RWa)+(Rz)\odot(LWa)\big)\right]_o.
$$

The overloaded coordinate index in the contraction is simply the head-coordinate index; the implementation uses `einsum('oia,ni,na->no', T, z, a)`. There are twelve output coordinates, 1152 residual coordinates and 128 head coordinates. Direct tensor and L/R formulas agree to relative error $2.23\times10^{-15}$.

Replace only this numerator with its sparse approximation. If $z'=z+Wa$, the change to the raw selected logits is

$$
\delta\ell_o=\frac{(\widehat T-T)_o(z,a)}{R(z')\,\rho(h)},\qquad
R(z')=\operatorname{mean}(z'^2)+\epsilon.
$$

$R$ is the squared input RMS denominator, because the bilinear term has two normalized inputs. $\rho(h)$ is the final RMS denominator. Apply the native $30\tanh(\ell/30)$ afterwards. In execution, $z'$ and $h$ are the cached native compact states, retaining their native rounding. The small projected-write discrepancy is reported rather than hidden.

Everything except this selected-output mixed numerator stays supplied: linear and head/head terms, background states, both normalization factors and the other output readers. This is conditional operator validation, not a compressed full state generator. Control readers are unchanged by construction and do not establish discovered selectivity. Native cached margin replay differs by at most $6.06\times10^{-6}$.

## Results and executed red-team

All six candidates (output-only or output/head frames, each at 2%, 5%, 10% coefficient error) add less than 0.6% error to the compact predictor effect and preserve all registered material signs on 120 historical regional prefixes. No new text was fitted or evaluated.

The zero-operator countercheck produces group errors of 1.72%, 4.42%, 2.88%, 4.98%, and 5.86%. It fails the registered all-groups 5% bar, but confirms that a large part of the initial apparent accuracy comes from leaving other contributions untouched.

We therefore also report error relative to the mixed term's own signed effect, defined by the difference between exact and zero mixed-operator predictions with the same supplied normalization. This diagnostic was added after the initial screen, not substituted for its registered criterion.

| Frozen candidate | Dense tensor storage saved | Error relative to its own mixed contribution, across groups |
|---|---:|---:|
| Output-only frame, 2% coefficient error | 8.26% | 0.92–4.68% |
| Output/head frames, 5% coefficient error | 18.29% | 1.75–7.78% |
| Output/head frames, 10% coefficient error | 30.81% | 2.72–8.12% |

This supports a modest, interaction-specific compression result. It does not establish fresh/OOD transfer, complete extraction, shared computation across behaviors, or whole-model savings. The dense folded tensor is the price baseline; the original factored program and runtime still need comparison. The 10% coefficient candidate is descriptive evidence, not a new post-selected passing registration.

The next promotion should preserve this explicit boundary and compare an executable sparse contraction with the native factored operator, then extend to normalization dependencies before claiming a reusable extracted component.

Receipts: [registered screen](INTERACTION_SPARSE_REGIONAL_V1_RESULT.json), [zero control](INTERACTION_SPARSE_REGIONAL_ZERO_V1_RESULT.json), [own-contribution diagnostic](INTERACTION_SPARSE_REGIONAL_OWN_EFFECT_V1_RESULT.json). Sources: [shared scorer](interaction_sparse_regional_v1.py), [zero-control entry point](interaction_sparse_regional_zero_v1.py). The original screen took 1.25 CPU seconds internally; no body forwards or fitting steps.
