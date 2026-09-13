# Five products for a same-amplitude response: an exact conic relation

13 September 2026. A further simplification exists for the **diagonal/self-product** of the three-vector MLP9 response. It reduces its per-context output bank from six vectors to five. It does not replace the six-vector bank for arbitrary independently chosen edit amplitudes.

## Derivation and scope

Use the established response basis $v_0,v_1,v_2\in\mathbb R^{1152}$ and coefficients

$$
\delta(a)=\sum_{i=0}^2u_i(a)v_i,\qquad
u(a)=\left(-a,-\frac{a}{\rho_a},\frac{a^2}{2\rho_a}\right),
\qquad \rho_a=\rho_0-2\beta a+\gamma a^2.
$$

Here $\rho_0$ is the pristine MLP9 mean-square norm plus native epsilon, $\beta=\langle z,w\rangle/1152$, and $\gamma=\|w\|^2/1152$. Learned block10 re-entry scaling is included in the basis vectors. The stable perpendicular-component denominator remains used in execution.

The coefficient curve obeys a homogeneous quadratic relation—a conic:

$$
\boxed{u_0u_1=\rho_0u_1^2+4\beta u_1u_2+4\gamma u_2^2.}
$$

Multiplying through by $\rho_a^2$ reduces both sides to $a^2\rho_a$. For $P_{ij}=K(v_i,v_j)$, where $K$ is the symmetric MLP10 mixed bilinear operator, the ordinary self-product uses six vectors. Substitute the relation into its $2u_0u_1P_{01}$ term. Store

$$
\begin{aligned}
Q_0&=P_{00},& Q_1&=P_{02},\\
Q_2&=P_{11}+2\rho_0P_{01},&
Q_3&=P_{12}+4\beta P_{01},\\
Q_4&=P_{22}+8\gamma P_{01}.
\end{aligned}
$$

Then exactly

$$
K(\delta(a),\delta(a))=
u_0^2Q_0+2u_0u_2Q_1+u_1^2Q_2+2u_1u_2Q_3+u_2^2Q_4.
$$

The ordinary bilinear numerator $B(\delta(a))$ is half this quantity because $K(x,x)=2B(x)$. Do not lose this factor when using the bank in a full branch expansion.

Five is the generic dimension of these scalar coefficient functions: after multiplying by $\rho_a^2$ they span monomials of degrees2through6 when $\gamma>0$. This is a statement about the diagonal coefficient family, not a minimum arithmetic-circuit size or independence guarantee for the actual output vectors.

For two independent amplitudes $a,b$, the corresponding polarized conic expression need not vanish. Therefore the old six-product cross kernel remains necessary in its declared unrestricted two-amplitude coefficient representation. We explicitly test a nonzero counterexample rather than treating this new identity as a replacement for that kernel.

## How this can serve an interaction

A joint finite difference evaluates diagonal branch contributions at $a$, $b$, and $a+b$. Each may have its own native downstream denominator. The five-bank identity holds pointwise before each division, so it also preserves the signed sum of these diagonal contributions. It does not by itself account for linear background terms, residual/attention cross terms, changed attention, downstream normalization or the suffix. In particular, it must not silently substitute for $K(\delta(a),\delta(b))$ in the older residual/residual interaction approximation.

## Actual-weight controls and preparation

[Implementation](response_diagonal_conic_v1.py), [signed-strength control](check_response_diagonal_conic_v1.py), [receipt](RESPONSE_DIAGONAL_CONIC_V1_CONTROL.json). Four synthetic pristine contexts use actual checkpoint weights. Ten strengths include zero, signs, values as small as $10^{-8}$, and amplitudes up to2:

- Five-bank versus six-bank diagonal relative error at most $1.43\times10^{-16}$.
- FP32 rounded-bank/coefficient combination error at most $9.13\times10^{-8}$; zero edit remains exactly zero.
- An independent-amplitude counterexample gives conic residual0.0486, confirming the scope restriction.

Initially deriving five vectors from six saves only retained working state. `prepare_direct` instead combines the hidden product vectors **before** the Down projection. It projects the three basis vectors through Left/Right, forms the shared hidden product $h_{01}$ once, adds its weighted versions to three other hidden products, and applies Down to five vectors instead of six.

[Direct-preparation control](RESPONSE_DIAGONAL_CONIC_DIRECT_V1_CONTROL.json): agreement with the converted six-bank is $3.79\times10^{-15}$; an unequal-positive-denominator finite-difference check agrees within $3.62\times10^{-15}$. Eleven interleaved two-thread FP64 CPU timings on four contexts give7.86ms for six-product preparation and7.33ms for direct five-product preparation, about1.07times faster. This small local timing is not a whole-model or GPU speedup.

Retained bank size falls from6912to5760scalars per context, **16.7% less than the six-bank**. It does not remove native weights or context preparation. No native full-circuit adoption or fresh/OOD validation is claimed yet. The useful structural lesson is that the producer's rational coefficient curve constrains its diagonal products even when its arbitrary two-input cross-products remain independent.

## Complete normalized branch and native validation, 13 September 09:04

The [full branch implementation](diagonal_mlp10_branch_v1.py) now retains the omitted terms explicitly. For the actual pre-MLP10 state $z$, let $r=\delta(a)$ be the generated residual response and $y=z-r$. This definition also absorbs any rounding correction into $y$. Then

$$
g(z)=z+\frac{B(y)+K(y,r)+\tfrac12K(r,r)}{\operatorname{mean}(z^2)+\epsilon}+b_{10}.
$$

The five-bank computes only the last numerator term. The first two terms retain the pristine background, attention response and their cross terms through the actual Left/Right/Down maps. Their hidden products are summed before a shared Down projection. The denominator uses the actual whole $z$, not the residual response alone; the native MLP bias remains. [Actual-weight CPU control](DIAGONAL_MLP10_BRANCH_V1_CONTROL.json) agrees with direct FP64 normalized MLP evaluation within $3.84\times10^{-15}$ on signed amplitudes, arbitrary attention/background terms and FP32-rounded inputs.

The [managed native comparison](DIAGONAL_MLP10_NATIVE_V1_RESULT.json) is complete:160historical prefixes,1280suffix evaluations,11.27seconds. All three registered criteria pass. Native reference corners replay exactly; maximum post-MLP10 state discrepancy is $2.97\times10^{-7}$. Maximum endpoint change versus the previous generated program is $6.68\times10^{-6}$.

Original-interaction target errors are0.033–0.084%across four regional groups and0.433–8.295%across four FineWeb groups, within their existing thresholds. [The sign audit](DIAGONAL_MLP10_NATIVE_V1_AUDIT.json) retains six small FineWeb sign reversals, with native reference magnitudes below $5\times10^{-6}$, and none at or above $10^{-5}$. The old program also had six small reversals, but the affected rows are not identical; this is tolerance-level preservation, not bitwise-equivalent behavior.

This establishes native-panel fidelity of the complete conditional branch implementation. It does not establish fresh OOD behavior, independent extraction, selective reuse or a whole-branch speedup. Prepared Left/Right basis readings, five output vectors, full native matrices, context generation and suffix still cost memory and computation. The earlier16.7%bank reduction compares diagonal product representations; it is not a net reduction relative to direct native MLP execution, which does not prepare that bank.

## Full-branch price: no adoption gain, 13 September 09:08

[The matched CPU benchmark](diagonal_mlp10_cost_v1.py) and [receipt](DIAGONAL_MLP10_COST_V1_RESULT.json) now compare the implemented full five-bank branch against direct FP64 MLP10 evaluation on identical batched inputs. Bank preparation is charged separately from an already-prepared reuse case. Both sides receive the same input states; shared upstream response-context/attention generation is outside these local timings.

| Batch | Direct | Five-bank, preparation included | Five-bank, already prepared |
|---|---:|---:|---:|
|1|3.56ms|12.97ms|3.71ms|
|3|5.53ms|14.22ms|5.42ms|
|16|6.93ms|18.03ms|8.56ms|
|64|19.99ms|33.83ms|24.01ms|

All numerical comparisons pass, but both registered10%speedup criteria fail. The small warm batch3 advantage is about2%, not robust evidence for adoption. Seven interleaved timings per case use two CPU threads; these are not GPU measurements.

The operation count explains why simply amortizing preparation is insufficient. Both implementations still apply Left, Right and Down to one full vector per input: $3\cdot1152\cdot4608=15,925,248$multiply-accumulates in their main dense contractions. The five-bank implementation adds response reconstruction, basis projections, cross products and bank combination; it has not eliminated a main contraction. It also stores33,408extra prepared scalars per context (267,264bytes in FP64), excluding the already-shared response context. Native matrices remain identical.

Therefore retain the conic identity as an exact structural simplification of the diagonal product bank, but do not adopt this full-branch schedule as a faster or smaller replacement for direct MLP10. The executed warm-bank countercheck rules out preparation cost as the sole explanation. Further computation savings would require simplifying the surviving background/attention contractions or a justified truncation of the actual amplitude-dependent response, rather than merely choosing five coordinates instead of six.
