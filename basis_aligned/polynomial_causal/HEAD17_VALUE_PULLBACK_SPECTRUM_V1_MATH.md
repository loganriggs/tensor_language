# Folding the value producer: modest matched-metric gain, unchanged linear span

13 September 2026. Compression setting 2; reuses the known head17 current/first value interface and the twelve-token MLP17 mixed tensor. No new semantic circuit is claimed.

**At shared rank 64, fitting after value folding gives 39.46% coefficient error versus 40.80% for fitting before folding**, with both evaluated on the same folded target. This is a 6.46% reduction in squared error. The six spelling contrasts show a similar 6.34% gain. Linear V folding helps this restricted approximation modestly; it does not yet produce a high-fidelity circuit.

## Object and exact calculation

The prior [mixed-tensor construction](HEAD17_OUTPUT_INTERACTION_COMPRESSION_V1_MATH.md) gives

$$
T_{oia}=\sum_k(OD)_{ok}
\left[L_{ki}(RW)_{ka}+R_{ki}(LW)_{ka}\right],
\qquad T\in\mathbb R^{12\times1152\times128}.
$$

It represents the mixed bilinear numerator between an arbitrary residual input z and head17.2 write Wh, read by twelve individual token rows O. The quadratic-in-h contribution, linear route, RMS factors and final token softcaps remain outside this particular target.

The known value producer is

$$
F_c=(1-\mu)V_{17},\qquad
F=[(1-\mu)V_{17}\;\;\mu V_0],
\qquad h=Fs,
$$

with \(\mu=-0.0888671875\), \(F_c\in\mathbb R^{128\times1152}\), and \(F\in\mathbb R^{128\times2304}\). Each V is the existing head2 slice. The two source inputs represent the current and first-layer normalized states. First-layer values are unchanged by the downstream head9 intervention, so current-only F_c is relevant to its value differences; the inherited contraction can also use the first-layer part.

The folded coefficient object is

$$
\widetilde T_{oij}=\sum_aT_{oia}F_{aj}.
$$

Flatten the first two axes into a matrix M. To avoid materializing the expanded tensor, factor

$$
FF^\top=HH^\top,
\qquad \|MF\|_F^2=\|MH\|_F^2.
$$

The best shared-head-mode rank-r approximation in this folded coefficient metric follows from the exact rank-r SVD of MH. Its right factor compiles back through H inverse using a triangular solve. This is a matrix-rank restriction, not unrestricted CP, sparse Tucker or arithmetic-DAG optimization.

## Why the comparison must use one metric

The original head-port rank64 error is 52.46%, whereas the optimal value-folded error is 39.46%. Those numbers use different norms, so their difference alone is not the gain from better compression.

The executed [matched comparison](HEAD17_VALUE_PULLBACK_MATCHED_V1_RESULT.json) evaluates both the port-fitted and producer-fitted factors on the same MF object:

| Shared rank | Port-fit error on MF | Producer-fit error on MF | Squared-error gain |
|---|---:|---:|---:|
| 16 | 72.30% | 71.23% | 2.94% |
| 32 | 59.68% | 58.35% | 4.42% |
| 64 | 40.80% | 39.46% | 6.46% |
| 96 | 24.49% | 23.36% | 9.07% |
| 120 | 10.58% | 9.82% | 13.83% |

For the six pair contrasts at rank64, the comparison is 40.38% to 39.08%. The similar improvement is not solely a fit to common token means. It still measures numerator coefficients, not the separately saturated token margins.

Both value maps have row rank 128, with condition numbers about 2.63. Therefore this fold changes the metric but does not reduce the head-write **linear span**. At most 2% coefficient error still requires all 128 shared directions in this rank family. That statement does not rule out sparse full-rank arithmetic or further shared-source/QK constraints.

At rank64, the twelve-output folded matrix factors contain 1,032,192 scalars, including a 64-by-2304 source reader. If the original value map F is supplied as an external interface, the head-coordinate factors instead contain 892,928 scalars and F remains charged separately. Both fitted alternatives have the same respective price. These are local factor prices, not whole-model savings.

## Controls and next consequence

[Spectrum script](head17_value_pullback_spectrum_v1.py), [receipt](HEAD17_VALUE_PULLBACK_SPECTRUM_V1_RESULT.json), and [matched-metric audit](audit_head17_value_pullback_v1.py). Direct source/head evaluation agrees to 2.32e-15 and expanded-slice energy to 2.22e-16. The initial spectrum computation took 0.63 CPU seconds.

This is only V folding. It treats source inputs independently and does not include the dependence of both QK factors on the same source, their normalization, the retained contraction gates, or the common source's polynomial identities. The more consequential next question is whether folding those **joint** computations exposes cancellations or reusable products. Reuse existing cubic-source and interaction-path oracles before authoring another decomposition of that object. The full-U multistart run remains unchanged and live in parallel.
