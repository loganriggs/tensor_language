# Nearly matched-parameter native group pilot

11 September2026. Both families decompose the actual MLP17 quadratic branch with the complete unembedding metric, not text or attention. Prior controls: MATCHED_GROUP_OBJECTIVES_V1_CONTROL plus shared-group and symmetric LL1 controls. The original unbound shared-reader-only runner is superseded by this combined pilot; do not enqueue it separately.

Family shared:64groups, eachoneunitinputreader withrank8partner,1,253,376matrixfloats. Family LL1:64outputgroups, eachsymmetricinputrank16 through16signedsquares,1,254,400floats. Bias1152 and originalbackground/U are retained additionally. Alloutputs live in exactUisometric1152coordinates. Zero corpus access/forwardpasses. No claim these smallbudgets match the prior9.14Mdictionary.

Two initializations per family: spectral leading64input-second-moment eigenvectors versus energy-weighted64nativeLeftreaders (seed1711); LL1 leading64outputfunction eigenvectors versus the corresponding energy-weighted nativeDowncolumns. Allreaders/outputvectors initiallynormalized. Fixed sharedinputreaders get exactrank8partnerSVD againstnative target. Fixed LL1outputvectors get signed top-absolute16eigenpairs oftheircontractednativequadratic. Then solve all64groupamplitudes jointly with whole-groupenergy penalty eta=.01. Retain bothnegative andpositive coefficients.

Jointoptimization reuses the normalized-coordinate L-BFGS-B controller, max1000iterations, maxcor10,maxls30,ftol0,gtol1e-9,120softseconds perarm (timecheckedafteracceptedstep). Feature/output parameterblocks normalized byinitialFrobeniusnorm; sharedinputreader/partnerrows and LL1input/outputrows are normalized insideobjective. Whole-group penalty is invariant under internal representations ofthe samegroup. It prevents physicalenergy blowupon a nonincreasing sublevel but doesnot guarantee goodbasins. The initialization/eigendecomposition cost is reported separately. Controller checkscanonicalpackedgradient descriptively; no jointconvergenceclaim.

For eachfamily independently:
- pred_a_instrument: nativefinite-difference relativeerror<=1e-6; independentCPresidual, amplitude-normal residual andgroupedexecutorrelativeerror<=1e-8; acceptedobjectivenonincreasingwithin1e-10. FD andnormal failures abort beforefit.
- pred_b_objective_gain: BOTHstarts improve penalizedobjective by>=.001.
- pred_c_capture: BOTHstarts achieve>=.1fullcoefficientcapture.
- pred_d_cost: BOTHinitialfullgradient evaluations take<=10GPUwallseconds.

Report completefunctioncosine betweenstarts, no registeredstable-unitpass. Report input-incidence and output-rankceilings usingexistingcaches. Preserve eachmiss andtime-limitedendpoint. A negative doesnot reject eitherfamily, adaptiveheterogeneousranks, or betterinitialization. A positive nominatesfactors, not fourpropertycircuits. No automaticidenticalcontinuation.

Capture is1-relative squaredcoefficienterror; functionalvalidation remains a laterfrozencandidate step. The distinguishing circuit question is whethermanyinputs writeonevariable versus onereader servingmanyoutputbranches. Goodreconstruction alone doesnot establish that thosegroups arebehaviorallyselective orstable.

Allsources/controlreceipts/caches/checkpoint/protocol arecontent-bound before reviewed-hash lane1enqueue. Oldshared-onlydraft stays unqueued. An overall2400-second alarm protects thisboundedpilot; interruptedarmskeepcompleted primaryreceipts and no silentlyresumedoptimizerhistory.
