# A useful algebraic LL1 initializer, with restrictive assumptions

11 September2026. The paired planted comparison foundonegoodrecoveryineight for both joint and output-projected optimization. Exactoutput elimination can change basins without improving overall recovery rate. Smallgradients occurred in badsolutions. [Pairedreceipt](LL1_PROJECTED_RECOVERY_V1_AUDIT.json).

An alternative is to recover input blocks algebraically from output mixtures. This is a symmetric restriction related to LL1/simultaneousblockdiagonalization. [DeLathauwer's2012primaryreport](https://ftp.esat.kuleuven.be/sista/delathauwer/reports/ldl-12-61.pdf) gives a broader formulation using a bilinear-kernel construction and rank/independence conditions. The code here does not implement that full algorithm or inherit its guarantees without checking assumptions.

Suppose

$$
T_o=\sum_g c_{og}A_gH_gA_g^\top.
$$

Require R=sum_g L_g<=d, concatenated A fullcolumnrankR, each symmetricH_g invertible, and two output mixtures with nonzero first coefficients and distinct coefficient ratios acrossgroups. Input spaces can be nonorthogonal, but under this fullcolumnrank condition they have no nontrivial shared intersection. This is a restriction relative to general overlapping/reusedinputgroups.

Compress the commoninputspace to R dimensions using its exact range. The compressed concatenated A is square/invertible. For output mixtures u,v define

$$
B_0=\sum_o u_oT_o=A\operatorname{blockdiag}(\alpha_gH_g)A^\top,
\qquad
B_1=\sum_o v_oT_o=A\operatorname{blockdiag}(\beta_gH_g)A^\top.
$$

Then

$$
B_1B_0^{-1}=A\operatorname{blockdiag}\left(\frac{\beta_g}{\alpha_g}I_{L_g}\right)A^{-1}.
$$

Each repeated eigenvalue therefore identifies one input block's invariant subspace. A dual input basis transforms the output slices into blockdiagonal form. Within one block the output-by-flattened-core matrix has rankone, recovering its outputvector and quadraticcore bySVD. Signed eigendecomposition of that symmetriccore gives executable squarefactors. Different internal bases represent the sameblock, so comparison uses fullgroupfunctions.

The implementation keeps real invariant spaces from both real/imaginary eigenvectorparts; simply discarding imaginaryparts can lose rank within a repeatedreal eigenspace. Known equal blockranks determine clustering here. This is not adaptive rankdiscovery. In a nativeimplementation, matrixcontractions can stayimplicit; the dense smallcontroller is not designed to materialize the fullvocabulary tensor.

[Executedcontrol](SYMMETRIC_LL1_PENCIL_V1_CONTROL.json) recovers the sameplanted3rank3blocks at relativeFrobenius error4.22e-14 andmatchedgroupcosine1, withoutoptimization. It also detects the collapsed eigenvaluegap when two outputloadingvectors are made identical. The raw reconstruction in that invalidcase isbad(error27.6); detection is essential, and returnedgroups must not be used as a successful decomposition whenconditionsfail. A production wrapper still needs explicit rejection/merging logic.

With relativeFrobenius noise1e-5, the initial mixturepair gives error9.77e-4 tocleantruth, narrowlypassing the1e-3bar. This motivated a targeted conditioningcheck rather than a broadnativeclaim. Eightfixedmixturepairs are compared on the sameobservednoisytensor; lowest observedreconstructionerror chooses seed2006, withoutusingcleantruth forselection. Its cleantrutherror is3.09e-5, a31.67foldreduction, andmatchedgroupcosine>.999999997. [Mixturecheck](SYMMETRIC_LL1_PENCIL_NOISE_V1_AUDIT.json), [selectedexecutor3.32e-16replay](SYMMETRIC_LL1_PENCIL_NOISE_V1_EXECUTOR.json). V1's Aflag recordedcandidatevalidity; the supplementaryexecutorreceipt completes its registeredreplay requirement.

These results show that a better algebraic initialization can recover structure missed by localoptimization in a case satisfying explicitconditions. They do not show that nativeweights satisfy thoseconditions. The nativepilot has approximationerror, potentiallyintersectinginputspaces and unknownblockstability. General LL1 may need adaptive ranks, moregeneralblockdiagonalization or nonconvexrefinement. No genericnoisy/global/semanticidentifiability conclusion follows from this controlledexample.

The alreadyqueuedmatched nativepilot staysunchanged. A potential later algebraicinitializer must first measure its native rank, separation and conditioning assumptions and preserve failedtests. Do not add a native dense kernel with dimension proportional to the vocabulary-input product.
