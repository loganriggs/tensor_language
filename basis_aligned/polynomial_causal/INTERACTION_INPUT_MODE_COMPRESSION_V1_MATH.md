# First setting1 compression baseline: exact metric, limited low-rank benefit

13 September2026. This is a weights-only compression result under the newly authorized interaction-compression agenda. It tests a particular linear reader bottleneck, not all sparse graphs, blocks or DAGs.

The retained MLP9 response contains the fixed map \(J=J_w\). Fold this map into one leg of the MLP10 cross-product:

$$
T(z,y)=D[(Az)\odot(Ry)+(Bz)\odot(Ly)],\qquad A=LJ,\ B=RJ.
$$

Here \(z,y\in\mathbb R^{1152}\), and \(L,R\) have4608 rows. This baseline allows arbitrary independent \(z,y\); the actual response family has additional relationships and amplitude-dependent normalization. We do not call this unrestricted operator the entire circuit.

## Exact compression metric

For \(H=D^TD\), define

$$
G_1=A^T[H\odot(RR^T)]A,\quad
G_2=B^T[H\odot(LL^T)]B,\quad
G_{12}=A^T[H\odot(RL^T)]B,
$$

$$
G=G_1+G_2+G_{12}+G_{12}^T.
$$

These contract the output and partner-input dimensions of the coefficient tensor, retaining the actual Down weights and cancellation between the two terms. For an orthogonal input projector \(P\), squared coefficient error is \(\operatorname{tr}[(I-P)G]\). Leading eigenvectors give the optimum **within this projector class**, so this test has no local-optimizer uncertainty. Three direct matrix-contraction checks agree within2.57e-15.

Independent term projectors come from \(G_1,G_2\). Their combined residual is scored including \(G_{12}\), not by adding two errors while ignoring cancellation. Those individual optima are a baseline, not a proven optimum of the combined objective over two different projectors.

For independent rank \(r\) per term, stored reader/adapter scalars are \(2r(d+m)\). A shared rank \(s\) costs \(s(d+2m)\), so choose \(s=\lfloor2r(d+m)/(d+2m)\rfloor\). Compare these additional weights to the original \(J\), which has1,327,104 scalars. Comparing only to materialized \(LJ,RJ\) would inflate the baseline eightfold and manufacture savings. Common \(L,R,D\) remain15,925,248 scalars in both implementations.

| Independent ranks | Shared rank | Independent error | Shared error | Shared extra storage / original J |
|---|---:|---:|---:|---:|
| 16+16 | 17 | 91.50% | 91.35% | 13.28% |
| 32+32 | 35 | 86.82% | 86.37% | 27.34% |
| 64+64 | 71 | 79.14% | 78.27% | 55.47% |
| 128+128 | 142 | 66.90% | 65.57% | 110.94% |

The shared projector needs rank1019 for2% coefficient error. There is a modest joint advantage at these budgets, but no useful high-fidelity compression of this unrestricted operator in this representation. The calculation took4.49seconds on two CPU threads, using no text.

## Red team and executed discriminator

The strongest immediate objection is that arbitrary \(y\) is much broader than the actual retained partner functions. We therefore also computed exact singular spectra for \(K(Jz,w)\) and \(K(Jz,Jw)\), where the second expression uses the fixed vector \(Jw\). They are matrices acting on \(z\), obtained directly from weights.

At rank128 their errors are42.23% and46.25%;2% accuracy needs ranks831 and845. A shared bank for the two unit-Frobenius-normalized functions needs941 directions for2%. Unit normalization is an explicitly different multi-function weighting, not the circuit's amplitude distribution. Restricting these two partners helps but does not expose a very small linear bottleneck.

This is **not** evidence against sparse arithmetic structure: even an identity matrix is full rank yet has a simple sparse computation. It also does not test output-selective readers, producer-constrained joint inputs, LL1 blocks, nonlinear intermediate sharing, or native behavioral tolerances. These are stronger reasons to change representation than to run more restarts of the already solved spectral problem. The next comparison should permit sparse product nodes/edges or block/DAG structure, with the same honest executable pricing; the second recent circuit remains a separate target to examine.

Receipts: `INTERACTION_INPUT_MODE_COMPRESSION_V1_RESULT.json`, `INTERACTION_FIXED_PARTNER_SPECTRA_V1_RESULT.json`. Reproducible executors: `interaction_input_mode_compression_v1.py`, `interaction_fixed_partner_spectra_v1.py`. No behavioral adoption or new circuit is claimed.
