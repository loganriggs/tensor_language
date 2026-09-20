# Next native target: Gaussian-projected quadratic part of the full quartic

Registered20September2026 after mean and cancellation controls. Status: exact implicit cross and gradient oracle implemented; native profiling and optimization notyetexecuted.

The existing centered candidate compresses the Taylor degree2 coefficient. It omits the covariance trace of degree4, which contributes to the optimal degree2 Gaussian approximation. For F(x)=H[x,x,x,x], x=mu+delta, delta~N(0,M), its Gaussian projection is

    m + B delta + Q : (delta delta - M),
    m=E F, B=E grad F, Q=(1/2)E Hess F.

Thus Q=6H(mu,mu,.,.)+6Tr_M H. The matrix-free helper gaussian_projected_quadratic_cross.py computes <Q_M,studentQ_M> against a few studentproducts, without expandingH. Small densevalue andgradient checks are3.1e-16 and4.0e-16. gaussian_low_degree_projection.py separately checks orthogonality and Gaussianmean derivatives.

First native experiment will hold the previous rank8linear branch and inputcenter fixed, and fit only4quadraticproducts plus an analytic constant. It therefore improves the quadratic Hermite component, not the full optimaldegree2 projection (B remains a fixed control). Price remains34,560scalars4products. Both centeredcovariance and trace-matchedspherical metrics use the samecalibrationmu.

Controls permetric:
A. Frozen regularized4product program withits existingTaylor-derivedconstant.
B. Sameprogram withfullteacherGaussianmean correction.
C. Refitquadraticpart againstQ above withthe samefixedlinear branch andfullGaussianmean correction.
This distinguishes mean correction from higher-degree contributions toquadraticresponse.

Normalizequadraticfeatures in eachmetric and useexplicitwriterpenalty.001. Analyticwriters andconstant, original-coordinate readers. Warmstarts fromregistered.001 refinements; noempiricalselection. Profileoneforward/backward andcompareonecrosscolumn inFP64 before choosing a bounded native stepcount. Proposed first grid: two metrics,two optimizers,two pairedwarmstarts,100steps; exactruntime/memory budget must be fixed fromprofiling beforeenqueueingthefit.

Instrumentbar: smalloracle<1e-10; nativeFP32/FP64crossagreement<1e-3, finitegradients andarchivereplay<1e-5. Sciencebar toregisterinfinalrunner: C improves registeredGaussianquadraticobjective versusB; empiricalC-vsB outcomesreported independently and mayreverse. No fullGaussianlossclaim withouttheconstant,linear,cubic andquartic residualterms beingaccountedfor. No semantic/OODclaim.

Algebraic implementation: Taylor quadratic factors plus the h_M cross terms use existingimplicitquadraticself/cross. The covariance term is C times the symmetrized row innerproducts of L J(u) M and R J(v), where J(u)=D[diag(Au)B+diag(Bu)A], evaluated atu=M a andv=M b forstudentreadersa,b. NativeGaussianmean isalreadyavailable exactly. This remains weights-first withdeclareddata-informedinputmoments.


## Native profile and fixed initial fitting budget

The native profile passed all three bars: FP32 four-product forward/backward
15–17 ms, peak allocation 1.04 GiB, FP32/FP64 relative cross error below 4e-7.
Reference FP64 contracts the same initially FP32-formed teacher factors.

Fix the first fitting grid at 2 metrics × 2 optimizers × 2 warmstarts × 100
steps, learning rate .005 with cosine decay and penalty .001. Warmstarts are
Adam seed 2 and Muon seed 2 from the registered .001 refinements, mapped back
to original input coordinates. This pairing is fixed before new native fits.
Budget: 800 optimization steps, estimated <60 seconds FP32 contraction time;
allow 180 seconds total for loading, exports, precision checks and evaluation.
No automatic rank/rate expansion. Hold rank8 linear branch and mean fixed.

The shared fit implementation is fit_projected_quadratic.py, with analytic
ridge writer and unit quadratic features in the selected covariance metric.
Science predictions for the first native fitting runner:
(a) each metric has at least one arm improving penalized quadratic objective
by >=1% relative to its initial objective magnitude;
(b) selected centered-metric refit C improves panel2 total prediction error
by >=.005 absolute versus the same initial program B with full-teacher mean;
(c) selected exported program replays <1e-5 relative error at 34560 scalars
and four products. Nulls and losses remain visible; selection uses weight-space
objective only. Compare A/B/C separately for both warmstarts and metrics.
Native fitting runner and exports remain to be completed; this is not a result.
