# Sharing attention10 changes across three downstream readers

13 September 2026. This weights-only check follows the three-term interaction compression. Its remaining dense background computations motivate folding attention's output into both MLP10 readers.

Let $u\in\mathbb R^{1152}$ concatenate the nine attention-head writes before the output projection. Let $O\in\mathbb R^{1152\times1152}$ be that projection and $L,R\in\mathbb R^{4608\times1152}$ the MLP readers. An attention change is consumed as

$$
Ou,\qquad LOu,\qquad ROu.
$$

All three matter: the residual state and normalization need the first, while the bilinear product needs both others. Compressing only one consumer can conceal damage to the rest.

For a shared orthonormal input basis $Q\in\mathbb R^{1152\times r}$, approximate all consumers by $M_iQQ^\top u$, with $M_i\in\{O,LO,RO\}$. Equal normalized Frobenius error gives the objective

$$
\frac13\sum_i\frac{\|M_i-M_iQQ^\top\|_F^2}{\|M_i\|_F^2}.
$$

The leading eigenvectors of

$$
G=\frac13\sum_i\frac{M_i^\top M_i}{\|M_i\|_F^2}
$$

solve this objective globally, up to eigenspace degeneracies and numerical accuracy. This follows directly from maximizing $\operatorname{tr}(Q^\top GQ)$ under orthonormal columns. There is no local optimizer to restart. Equal block weights are an explicit choice; this is not a behavioral or minimax objective.

## Executed result and countercheck

[The actual-weight CPU check](ATTENTION10_JOINT_READERS_V1_CONTROL.json) takes 0.84 seconds. At rank 768, errors for O/LO/RO are 16.61%, 15.39%, 15.37%; the registered 10%-per-consumer prediction fails. The optimal joint error is 15.80%, so no rank-768 subspace can make all three errors at most 10% under this definition. At rank 1024, all errors fall below 3.15%, but nominal multiplication savings almost disappear.

[The capacity countercheck](ATTENTION10_JOINT_READERS_V1_CAPACITY.json) finds rank 877 is the first member of this spectral family with all three errors below 10%: 9.99%, 9.22%, 9.19%. Independent optimal approximations need ranks at least 871/846/845 respectively. In particular, the residual output consumer alone rules out common rank below 871. Sharing is therefore not the principal obstacle at this tolerance: even independent consumers require substantial rank. The first passing spectral rank is not claimed to be the exact minimax-optimal rank.

The trace-subtraction diagnostic at full rank has numerical residuals up to roughly $4.4\times10^{-8}$; these are floating-point cancellation, not evidence of a nonzero exact full-rank error. The materially larger reported errors do not depend on that numerical floor.

## Literal cost and limitations

Computing the attention change's three readouts through the original maps costs

$$
1152^2+2(4608)(1152)=11{,}943{,}936
$$

multiply-accumulates per input. The shared-factor route costs

$$
r(1152+1152+2\cdot4608)=11{,}520r.
$$

At rank 877 that is a nominal 15.4% reduction for these readouts alone. It introduces 40,412,160 bytes of FP32 factors. Original maps are still needed for pristine context and other computations, so those factors are additional storage unless those dependencies are separately replaced. These counts exclude preparation, memory traffic, other background terms, normalization, and the downstream suffix; no runtime gain or whole-model compression has been established.

This rules out a strong low-dimensional common input subspace for the full attention-write domain at the tested tolerance. It does not rule out sparse/block decompositions, structure in the actual QK/value-generated changes, selected head paths, or output-specific contractions. In particular, the independent input domain is deliberately broader than the actual folded producer's domain. The earlier three-term success depended on retaining such producer constraints. A native test of this broad rank-877 approximation is lower priority than finding a representation that exploits those constraints without adding this much storage.
