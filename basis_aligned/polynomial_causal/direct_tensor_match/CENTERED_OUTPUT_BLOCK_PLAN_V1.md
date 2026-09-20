# Output-shared quadratic baseline — 2026-09-20 20:07 UTC

Realize the exact output-rank relaxation of the frozen8×16×16 quadratic tensor atoutputranks2,3,4,6. Compute outputSVD, project originalquadraticwriter into retainedoutputdirections, and implement via original8sharedproducts plus a small mixingmatrix. Do not independentlyexpand everyquadraticform and duplicate sharedproducts.

Each computedoutputfeature is a linearcombination of8quadraticproducts; h_g=sum_i Z_gi q_i, y=W h. Orthonormal W makes quadraticoutputcomponents orthogonal, but saysnothing aboutwithin-feature primitivecancellation or semantics. Report both aggregateoutputcomponent andprimitivecomponentcancellation. Center/linearbranchfixed; constantmatchesfrozenstudentGaussianmean.

Price39,168+1,160r scalars;8products; rank4cost43,808. Compare four computedoutputfeatures here versus four products in theCPrefactor; these aredifferentstructuralassumptions, notequalcost claims.

pred_a: retainedenergy matchesoutputrankbound<1e-10; orthogonalcomponentenergyratio1within1e-10; archive/scalarDAGreplay<1e-10.
pred_b: rank4panel2error within.01absolute of23.1229%source.
pred_c: primitivecomponentenergyratio<10 at rank4; iffalse orthogonaloutputsharingdoesnotsolve internalcancellation.

Selectionisclosed-formweightSVD, notempiricalbest-rankselection. No semantic/causalclaim.
