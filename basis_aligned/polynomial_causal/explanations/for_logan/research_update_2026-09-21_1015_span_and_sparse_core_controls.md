# Input span and sparse interactions are separate restrictions

21 September 2026, 10:15 UTC. **The native-coordinate error has two distinct sources: which input directions the graph can use, and which interactions it permits between those directions.** The new CPU controls separate them while the registered two-width fit runs. None is an adopted circuit.

For the three selected component pairs, all errors below use the native isotropic coefficient metric, normalized equally across pairs.

| Representation or relaxation | Native coefficient error |
|---|---:|
| Current product dictionary, best native readout | 59.43% |
| Same 766-dimensional reader span, arbitrary dense quadratic cores | 45.32% |
| A different 766-dimensional span selected from the joint mode Gram matrix, dense cores | 36.55% |
| Necessary lower bound for any 766-dimensional reader span | 27.81% |

The middle rows allow dense interactions that the product dictionary does not contain. They therefore cost much more: six full symmetric cores of width 766 have **1,762,566 entries**, before storing the input basis or output interface. The 36.55% construction is feasible, but is not asserted to be the globally best span. The 27.81% number is a lower bound, not an achieved fit.

Five small independent checks compare the Gram-matrix bound with an explicit unfolding SVD and verify cases where diagonal forms attain the bound. Discrepancies are below $1.4\times10^{-16}$.

**What sparsity should count**

For a quadratic Tucker representation,

$$
s=P^\top x,\qquad y_o=\sum_{p\leq q} C_{o,pq}s_ps_q,
$$

one product $s_ps_q$ can serve several outputs. Counting every nonzero $C_{o,pq}$ as a new multiplication would charge that product repeatedly. This control instead chooses a set of **distinct input pairs**, computes each once, and separately counts its output coefficients.

With a fixed orthonormal input basis, pair atoms are orthogonal in coefficient Frobenius geometry. Consequently, choosing the pairs with the largest grouped coefficient energy gives the optimal fixed-basis support for a given distinct-product budget. Diagonal and off-diagonal entries have different multiplicities and are accounted for explicitly.

Five planted examples with one through five shared products recover their known functions and execution values below $10^{-15}$. These test the selector and accounting; they are not additional evidence of learned native semantic structure.

**Native fixed-basis results**

| Fixed input basis | Distinct products | Active input features after pruning | Stored floats | Native coefficient error |
|---|---:|---:|---:|---:|
| Joint mode basis, initially 766 features | 399 | 111 | 141,798 | 89.27% |
| Joint mode basis, initially 766 features | 592 | 137 | 172,908 | 88.31% |
| Current reader span's orthonormal basis | 399 | 204 | 248,934 | 98.37% |
| Current reader span's orthonormal basis | 592 | 249 | 301,932 | 97.96% |

Float prices include the active input basis, output coefficients and existing affine/later-state interface. Pair indices require an additional two integers per product. Unused input features are physically omitted from the price. These low-storage controls are inaccurate: even the better 592-product joint-mode-basis case has about 39% third-component error on the opened states.

The conclusion is specific: **entry sparsity in either of these fixed bases does not economically preserve the chosen native tensor.** It does not rule out learning a different basis, using nonorthogonal features, factoring dense low-rank core slices, or further graph rewrites. The mixed-product fits already allow directions that these fixed-basis controls cannot choose.

This makes the two-stage distinction concrete. A good input subspace need not have a sparse core in an arbitrary basis, and a small coefficient norm does not identify a cheap arithmetic program. We need to optimize the interactions and their reusable implementation, with both cost and behavior checked.

The 12-fit width/metric comparison remains live. An independent audit is prepared to decode its exported programs, recompute both norms and component errors, and verify physical storage. No final winner or prediction verdict is reported here.

Evidence: [span relaxation and independent checks](../../direct_tensor_match/SOURCE_SPAN_RELAXATION_V1.json), [shared-pair sparse-core control](../../direct_tensor_match/SPARSE_PAIR_CORE_V1.json), and [pending fit audit code](../../direct_tensor_match/audit_dual_geometry_source.py). All native component evaluations reuse the opened 448 states; there is no fresh/OOD or standalone extraction claim.
