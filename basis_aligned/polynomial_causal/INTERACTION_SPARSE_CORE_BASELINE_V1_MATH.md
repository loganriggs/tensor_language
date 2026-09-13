# Sparse edges in the composed head17–MLP17 interaction

13 September 2026, 06:18 UTC. These are weight-only fixed-frame baselines, not learned sparse Tucker fits or validated replacements.

The object is the mixed coefficient tensor $T\in\mathbb R^{12\times1152\times128}$ from `head17_output_block_objective_v1.build`: twelve selected output readers, a residual input, and head17.2's value coordinates. It represents the cross term from folding the head write through the final bilinear layer. Normalization, the linear route and the head/head quadratic term remain outside this tensor.

For orthogonal mode frames $A,B,C$, write

$$
T_{oia}=\sum_{pqr} A_{op}B_{iq}C_{ar}G_{pqr}.
$$

Orthogonal changes of coordinates preserve Frobenius error. In any fixed frame, retaining the largest squared entries is therefore the exact best support of a given size. We tested native coordinates, all three full HOSVD frames, and all eight subsets of those mode rotations. HOSVD here means eigenvectors of each mode's Gram matrix; there is no rank truncation or rotation optimization. Full reconstruction and energy preservation were checked numerically.

## What happened

The dense folded tensor contains 1,769,472 coefficients (7,077,888 FP32 bytes). Full HOSVD makes its entries more concentrated, but its adapters contain 1,343,632 additional coefficients. Those adapters erase the apparent strict-accuracy savings.

An initial coordinate-list price charged a 32-bit index per retained coefficient. We red-teamed that negative result with a one-bit occupancy mask and packed FP32 values: the mask costs only 221,184 bytes. Native-frame pruning then saves 6.08% of dense storage at 2% coefficient error. Thus a blanket claim of no sparse storage saving would have been wrong. This is storage accounting, not a measured sparse execution speedup.

We then executed the stronger control: rotate only selected axes, paying only for those adapters.

| Coefficient error ceiling | Cheapest tested rotations | Bytes / dense tensor |
|---|---|---:|
| 50% | Output and head input | 20.11% |
| 20% | Output and head input | 50.92% |
| 10% | Output and head input | 69.19% |
| 5% | Output and head input | 81.71% |
| 2% | Output only | 91.74% |

At 2% error the best tested representation saves 8.26% of this dense tensor's bytes. At 5% it saves 18.29%. The large residual-space adapter is not worth its price among these choices. Neither conclusion rules out a learned sparse frame, a cheap structured transform, joint sharing across interactions, or a different arithmetic graph.

## Scope and next decision

This finds edge sparsity in an already folded operator. It does not remove reader nodes, demonstrate reuse across behaviors, beat every factored implementation, or establish native behavioral fidelity. Dense T is a local comparison baseline; the original factored L/R/D/W program is another baseline that must be priced before adoption. A 2% coefficient error need not preserve a small signed circuit effect.

The next useful comparison should either optimize a genuinely different graph or validate the cheap partial-frame candidate on the already defined circuit ports. Repeating more global-output fits would not answer this interaction-specific question.

Receipts: [native/full frames](INTERACTION_SPARSE_CORE_BASELINE_V1_RESULT.json), [bitmap countercheck](INTERACTION_SPARSE_CORE_BITMAP_V1_RESULT.json), [all partial frames](INTERACTION_PARTIAL_FRAMES_V1_RESULT.json). Executable partial-frame analysis: [source](interaction_partial_frames_v1.py). No model forwards or GPU jobs were needed.
