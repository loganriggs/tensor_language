# Relax shared squares to paired blocks and a sparse interaction core

22 September2026,02:57UTC. Same true native MLP17quadratic tensor,16fixedoutputreaders/all1152inputs. Choose one orthogonal inputbasis from eigenspaces of sum_v (T_v/||T_v||F)^2. This basis is weight-derived but not optimized for core sparsity; negative results apply to this construction, not arbitrary orthogonalTucker.

Compare diagonal-only core; greedy disjoint2x2blocks (all1152diagonals plus576crossproducts); and topK shared pairs atK1152/2304/4608, chosen by total normalizedcoefficientenergy (factor2foroffdiagonalentries). Pair products are shared across outputs and charged once. Repeatednonzero coreentries do not imply repeatedmultiplication. Nativebaseline uses4608bilinearproducts and hasexactzeroerror forselectedquadratic map.

Report normalizedcoefficient error, eachoutputerror, naturalpooled coefficient error and exact isotropicGaussianfunctionerror including trace terms. None is text/OOD fidelity. Orthogonality makes fixedbasis topK optimum for coefficienterror overKsharedpairs; basis itself notoptimized. Greedy matching notgloballyoptimal. Five toystructures include planted diagonal/block/sparse/dense/signed forms, with directpolynomialreplay and knownbasis witnesses. Nativefold replay<1e-10. Noexports/adoption fromthisscreen.
