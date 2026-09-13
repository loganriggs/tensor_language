# Sparse native product blocks: a dictionary limitation, not just a greedy miss

13 September2026, setting1 compression baseline. The target remains the explicitly restricted weight operator \(T(z,y)=K(Jz,y)\), with arbitrary independent inputs. This is not the entire normalized regional circuit or its behavioral objective.

## Representation and fit

Retain selected paired native-channel terms

$$
F_k(z,y)=D_{:k}\big[(L_kJz)(R_ky)+(R_kJz)(L_ky)\big],
\qquad \widehat T=\sum_{k\in S}\alpha_k F_k.
$$

One selected block uses two scalar products and a shared output vector. This permits many input directions; it does not impose the preceding low-dimensional input projector. It nevertheless fixes the native readers and writers, so it is only one sparse dictionary.

For \(A=LJ,B=RJ\), the exact coefficient Gram matrix is

$$
G_{kl}=\langle D_{:k},D_{:l}\rangle
\big[\langle A_k,A_l\rangle\langle R_k,R_l\rangle
+\langle B_k,B_l\rangle\langle L_k,L_l\rangle
+\langle A_k,B_l\rangle\langle R_k,L_l\rangle
+\langle B_k,A_l\rangle\langle L_k,R_l\rangle\big].
$$

Its total sum agrees with the independently constructed input-mode tensor norm. Greedy orthogonal least squares selects the block giving the greatest next exact residual reduction; selected coefficients are refit by a linear solve. Incremental and direct residual calculations agree within1e-8. Coefficient normal-equation residuals are at most1.44e-15. This establishes accurate coefficient optimization on the selected support, not globally optimal support selection.

| Paired blocks retained | Scalar products | Coefficient error | Scalars including original J |
|---:|---:|---:|---:|
| 32 | 64 | 98.78% | 1,437,728 |
| 128 | 256 | 96.79% | 1,769,600 |
| 512 | 1024 | 91.02% | 3,097,088 |

The unpruned operator stores17,252,352 scalars for \(J,L,R,D\). Counts retain \(J\) once and use selected native \(L,R,D\) entries; they do not inflate the baseline by materializing \(LJ,RJ\). Normalization and the remaining response computations are outside this local price. Refitting supports chosen simply by block norm gives almost identical errors; changing greedy ordering offers little evidence of a large gain. Both baseline and bound calculations take about four seconds on two CPU threads, with no text fitting.

## Executed red team: could a much better subset rescue this dictionary?

Normalize the atoms to unit Frobenius norm, with Gram \(H\), and let \(b_k=\langle F_k/\|F_k\|,T\rangle\). For any support of size at most \(s\), Gershgorin's inequality gives

$$
\lambda_{\min}(H_{SS})\ge\gamma_s
=\min_k H_{kk}-\max_k\sum_{j\in\text{largest }s-1}|H_{kj}|,
$$

where the sum excludes the diagonal. If \(\gamma_s>0\), the best possible coefficient fit on any such support captures at most

$$
b_S^TH_{SS}^{-1}b_S
\le\frac{\sum_{k\in\text{largest }s}b_k^2}{\gamma_s}.
$$

For512 blocks, the evaluated \(\gamma_{512}=0.74070\) yields **at least87.45% relative error for every512-block subset in this fixed dictionary**, compared with the greedy result91.02%. For128 blocks the corresponding lower bound is96.11%, versus96.79% achieved. These are analytic bounds evaluated in FP64, not interval-arithmetic certificates. Their margins are much larger than the numerical checks.

Therefore the poor result cannot plausibly be repaired merely by better subset selection or longer coefficient optimization in this dictionary. It does not bound learned CP factors, LL1 blocks with changed readers, sparse cores in another frame, shared arithmetic DAGs, output-selected operators, or the actual producer-constrained input family. The remaining question is representational. Move to those comparisons and the second recent interaction rather than spending more starts on fixed native-channel pruning.

Receipt: `INTERACTION_PRODUCT_NODES_V1_RESULT.json`; reproducible fit and bound: `interaction_product_nodes_v1.py`. No native intervention or circuit adoption is claimed.
