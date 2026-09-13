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
