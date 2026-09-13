# Sparse interaction: executable, smaller on disk, slower on CPU

13 September 2026. The frozen output/head-frame 10% coefficient-error candidate now has a packed executable artifact. Numerical and stored-size criteria pass; the CPU speed criterion fails.

The program stores an occupancy bitmap, FP32 active coefficients, and FP32 head/output adapters. At loading, it expands the support to twelve compressed sparse row (CSR) matrices. Execution transforms the head coordinates, multiplies each sparse matrix by them, contracts with the residual input, and transforms the twelve output coordinates back. Input generators, normalization and remaining paths are still external.

| Quantity | Measured result |
|---|---:|
| Serialized sparse program | 4,899,795 bytes |
| Packed tensor payload | 4,896,936 bytes |
| Dense folded T | 7,077,888 bytes |
| CSR resident coefficients, indexes and adapters | 9,340,736 bytes |
| Original local L/R/C/LW/RW factors | 47,407,104 bytes |
| One-time decoding | 11.86 ms |

The native factored price is a standalone local operator price. Some native factors may already be present or reusable elsewhere, so this is not a net whole-model saving. Likewise the CSR resident price excludes temporary activations and the retained packed file during loading.

FP32 sparse execution matches the dense pruned FP64 operator to relative error 2.08e-7 on regional ports and 3.03e-7 on deterministic independent Gaussian ports. Exact T matches the native factored formula below 3.4e-15. The approximation itself has 2.88% aggregate raw mixed-numerator error on those regional ports and 9.82% on independent ports; these are different metrics from the earlier regional signed-effect results. Independent random ports are an algebraic control, not OOD language evidence.

Two CPU threads, two warmups and seven measurements per implementation:

| Batch | Sparse CSR median | Dense T median | Native factored median |
|---|---:|---:|---:|
| 1 | 0.481 ms | 0.221 ms | 0.699 ms |
| 120 | 5.583 ms | 1.973 ms | 12.422 ms |

Dense pruning stored as a dense array gives similar speed to exact dense T. Folding alone already reduces this standalone computation substantially; irregular sparsity does not improve that CPU implementation.

## Executed countercheck and graph interpretation

A post-result alternative stacks the twelve sparse matrices into one call. It agrees with the original execution within 1.51e-7. Against an identical-frame dense implementation it still loses: 0.353 versus 0.202ms at batch1, and 5.953 versus 2.000ms at batch120. Thus twelve Python/sparse calls are not the sole explanation. The stacked batch120 intermediate alone occupies 6,635,520bytes. These timings are local CPU measurements, not a universal bound or GPU benchmark.

The support audit finds all 12 output, 1152 residual and 128 head nodes active. Of 147,456 possible coordinate input products, 147,453 remain, and 147,412 feed at least two output coordinates. Nearly every 4×4 tile (99.981%) and every 8×8/16×16 tile is occupied. There is edge pruning but essentially no node pruning or block sparsity in this support. Coordinate-product sharing is possible; it is not evidence of identified semantic reuse. A product-once implementation would also need its intermediate state and output accumulation priced.

This changes the next structural experiment: fit shared blocks or intermediate functions directly, and compare against this fixed-frame baseline. Further optimizing the same irregular kernel is lower priority. The current artifact is a conditional executable operator and a storage result, not an adopted faster circuit.

[Primary receipt](SPARSE_INTERACTION_EXECUTOR_V1_RESULT.json) · [Stacked-call countercheck](SPARSE_INTERACTION_STACKED_V1_RESULT.json) · [Support graph](SPARSE_INTERACTION_SUPPORT_GRAPH_V1_RESULT.json) · [Executor](sparse_interaction_executor_v1.py) · [Packed program](SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt).

## Broader native-port transfer, 13 September 08:52

The same frozen packed operator was evaluated on64existing FineWeb prefixes that were not used in its weight-only fit. This test uses the **full native head17.2 contribution**, rather than the retained child/remainder mixed write of the regional experiment. Consequently it changes both text and input-port regime; it does not isolate corpus shift. FineWeb is the training corpus, and these are historical prefixes: no fresh OOD claim.

At the final position, let $z$ be the actual input to MLP17, $a$ the native head2pre-output vector, $v=Wa$, and $z_0=z-v$. The tested numerator is the same operator $T(z_0,a)$. Native final state $h$, RMS factors and other terms remain fixed. For the twelve selected token logits, remove only this mixed numerator to define its own-effect reference:

$$
\ell_{\rm zero}=\ell_{\rm raw}-\frac{T(z_0,a)}{(\operatorname{mean}(z^2)+\epsilon)\sqrt{\operatorname{mean}(h^2)+\epsilon}},
$$

then add the compressed numerator and apply the native $30\tanh(\ell/30)$ softcap. Compare all six paired-output margins per prefix. These are selected-output probes, not next-token CE or whole-head removal effects.

[Managed receipt](SPARSE_INTERACTION_FINEWEB_V1_RESULT.json):64bodyforwards,2.05seconds. Exact tensor/native-factor replay error is $2.93\times10^{-15}$, and the actual FP32 CSR executor agrees with its dense reconstructed operator within $2.06\times10^{-7}$. Raw mixed-numerator error is7.66%, but the paired-margin own-effect errors are19.80%,19.18%,19.62%,22.14%across the four16-prefix groups. All miss the10%bar. There are18sign reversals, all at reference effect magnitude at least $10^{-5}$; the sign criterion also fails.

The [executed countercheck](SPARSE_INTERACTION_FINEWEB_V1_AUDIT.json) finds17.24–25.54%error across individual output pairs, so a single pair does not explain the aggregate miss. Restricting to reference effects at least $10^{-3}$ leaves351/384probes,20.27%error and8sign reversals. Thus tiny references alone do not explain the failure. These descriptive slices do not replace the registered criteria.

This limits the current candidate's broader reuse. It preserves its earlier regional retained-port result, while showing that10%total coefficient error and smaller storage are insufficient for reliable selected-output effects on these native full-head ports. A matched-port comparison across text families is needed before attributing the difference specifically to text distribution. No additional fitting or weight changes were made.
