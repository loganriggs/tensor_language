# Folding the fixed residual writer together with attention10

13 September 2026. A conditional mixed interaction has a useful exact compilation even though it does not exhibit strong low-rank or whole-head sparsity.

The current response basis contains a globally fixed vector $v_0=\lambda_{10}w\in\mathbb R^{1152}$, alongside two context-dependent vectors. Let $u$ concatenate the nine attention10 head writes before output projection $O$. The portion mixing this fixed residual direction with attention is

$$
K(v_0,Ou)=D[(Lv_0)\odot(ROu)+(Rv_0)\odot(LOu)].
$$

Here $L,R$ are $4608\times1152$, $D$ is $1152\times4608$, and $O$ is $1152\times1152$. Define

$$
J_{v_0}=D\left[\operatorname{diag}(Rv_0)L+\operatorname{diag}(Lv_0)R\right],
\qquad M=J_{v_0}O.
$$

Then the mixed interaction is exactly $Mu$. An attention *change* can be supplied by subtracting pristine from changed head writes. The scalar coefficient of $v_0$ multiplies the result outside this fixed operator. The same M can serve child, remainder and parent branches, every token position and different contexts, provided those inputs use the same writer and checkpoint.

This is ordinary exact contraction of an interaction path, not a new factorization theorem. It joins parts of the attention output projection and bilinear readers/writer into one executable component. The head-write producer, varying residual modes, normalization, background and suffix are external dependencies.

## Weight structure and negative-result countercheck

[The CPU structure screen](FIXED_WRITER_ATTENTION_OPERATOR_V1_CONTROL.json) verifies direct expansion against M within $3.95\times10^{-15}$. It finds:

- The top three head-column blocks contain 46.88% of M's squared Frobenius norm, compared with 47.37% for O. Folding does not concentrate the term into three heads; the 90% prediction fails.
- Rank 128 leaves 60.46% Frobenius error; the 10% prediction fails.
- The optimal rank required for 10% error falls from 871 for O to 642 for M. Composition does expose additional spectral structure, but not the strong compression proposed in the screen.

These ranks come from exact numerical SVDs. More local-optimizer restarts cannot improve this matrix-rank objective. They do not exclude structured arithmetic representations, sparse hidden products, output-specific contractions, or stronger constraints on the actual QK/value-generated inputs. Head-block weight energy is not causal head importance.

## Exact execution price

A negative low-rank result is not a negative verdict on compiling the interaction. [The matched CPU benchmark](FIXED_WRITER_ATTENTION_OPERATOR_V1_PRICE.json) compares direct bilinear expansion, a previously compiled J followed by O, and the single compiled M. All use actual weights, FP64, two threads and identical head-write inputs.

| Inputs per call | Direct expansion | J and O | Folded M |
|---:|---:|---:|---:|
| 1 | 4.74 ms | 0.527 ms | 0.272 ms |
| 16 | 7.54 ms | 1.123 ms | 0.582 ms |
| 128 | 41.41 ms | 5.618 ms | 2.818 ms |

M is 1.93–1.99 times faster than the already compiled two-matrix baseline. This avoids attributing all gains to folding O when most of the direct-expansion gain comes from precomputing J. Numerical agreement with direct expansion is within $4.03\times10^{-15}$.

M stores 1,327,104 scalars, versus 2,654,208 for independent J and O. It requires 1,327,104 multiply-accumulates per input, half the two-matrix baseline. Constructing J takes 0.230 seconds; folding O into it adds 0.0248 seconds. Relative to an already available J, that additional work breaks even after approximately 97/734/1134 inputs at the measured batch sizes. The cache is global, so repeated contexts can amortize it.

The storage reduction applies only if this isolated component replaces the two matrices. If J or O remains needed elsewhere, M adds storage to the whole program. Likewise, shared native reader projections used by other terms may change the best integrated baseline. No full-branch or GPU speedup, selective removal, fresh/OOD circuit prediction, or semantic circuit identification is established here. The new executable component needs integration and a native intervention before promotion beyond this exact conditional algebra and local cost result.

## Native removal and rank-128 replacement, 13 September 10:05

[The managed removal test](FIXED_MIXED_NODE_REMOVAL_V1_RESULT.json) evaluates 160 historical prefixes in 13.46 seconds. In each changed branch, it deletes only

$$
-\frac{a}{\rho_{10}}M\,\Delta u_{\mathrm{attn}},
$$

where the head-write difference is computed by the full squared-attention generator, including both normalized QK factors, rounded rotary tables and value mixing. The direct residual coefficient is $-a$; no extra factor of one-half belongs in this cross term. Pristine attention, other residual modes, self products and normalization remain. Compiled-node versus direct bilinear error is at most $8.37\times10^{-12}$, and native reference endpoints replay exactly.

Both preservation predicates fail: two regional groups exceed the 2% target bar, and FineWeb0 exceeds its 10% target bar. [The aligned audit](FIXED_MIXED_NODE_REMOVAL_V1_AUDIT.json) finds deletion changes the original interaction by 1.63–3.47% for regional targets and 1.00–1.42% for controls. One material regional target sign reverses. This term should not be discarded under the registered criterion.

[Signed effects](FIXED_MIXED_NODE_SIGNED_V1_AUDIT.json) again caution against a supportive-feature label: removal-defined contributions oppose the regional target in all four groups, with cosines from -0.238 to -0.883. Controls have mixed alignment. These projections describe compensation and are not additive causal percentages.

The next [rank-128 replacement](FIXED_MIXED_RANK128_V1_RESULT.json) uses the exact truncated SVD of M, fitted only to weights. It subtracts the exact term and adds the compressed term in the same branch. There is no text-guided optimization. Native algebra/reference checks and the full regional preservation criterion pass, but the FineWeb criterion and the stricter component-preservation criterion fail.

The distinction matters: regional full-interaction target error is only 0.086–0.179%, while error relative to the mixed term's **own** removal effect is 3.73–6.37%. Thus actual generated inputs and target readout are substantially easier than the arbitrary-input weight error of 60.46% suggests. However, own-effect regional control errors are 27.11–45.44%, and FineWeb own-effect errors span 23–109% across endpoints. The full FineWeb0 target error is 12.91%, above its 10% bar. Small full-interaction error alone would have concealed these failures.

The [rounding countercheck](FIXED_MIXED_RANK128_V1_ROUNDING_AUDIT.json) allows half a final-scalar FP32 ULP for all six nonshared corners in candidate-minus-reference interactions. Even after this allowance, regional control error remains at least 23.8–38.8% of the component's own effect norm. Restricting descriptively to component effects of magnitude at least 0.00001 still gives 19.3–40.9% regional control error. Final scalar rounding alone cannot rescue the failure. This does not bound internal transformer rounding or change the original predicates.

Rank 128 would use 294,912 factor scalars, 22.2% of the exact matrix, but it is not adopted as a faithful general replacement. The evidence supports task-dependent compressibility of this composed term, with weaker preservation of other consumers. It does not establish a selective semantic circuit, universal low-rank structure, or fresh/OOD transfer.
