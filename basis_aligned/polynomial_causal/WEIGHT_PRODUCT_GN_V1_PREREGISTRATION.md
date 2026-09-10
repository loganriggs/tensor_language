# Native joint Gauss–Newton benchmark — 10 September 22:14 UTC

Same128products, frozen stalled initialization, all-U coefficient objective and
explicitlambda.01 component-energy penalty as the earlier penalizedfit andALS.
No data. Whiten output coordinates exactly withC=Cholesky(U^T U)^T and divide
writers/nativeDown bysqrt(total coefficient energy), avoiding metric imbalance.
Keepboth inputreader families andwhitenedwriters in the joint GN linearization.
Exact tangent consists of delta-writer, delta-left anddelta-right product sums.
The adjoint uses productGrams; no fulltoken tensor, Jacobian or Hessian.

DampedPCG: diagonalJ^TJpreconditioner, mu=.01initial, true relative residual1e-3,
max100iterations. Inexactinnersteps may beaccepted only with actualdecrease and
positivepredicteddecrease; countthem. Trust ratioaccept>1e-4; damping /3 above.75,
x2 below.25, x10 onrejection, bounded1e-8..1e8. After trialupdates, normalize
readerrows and solveconditionalwriters exactly. This is joint GN with conditional
projection, not a fullreduced-Hessian claim. Numericalpredicteddecrease below
64*eps*max(abs(objective),1) terminates as resolutionlimit, neverconvergence alone.

120second initial benchmark, timer between fullsteps; save~6MB current/best
checkpoint. Prior540s warmstart remains charged. Native bodyforwards0.
A: denseJacobian/normal/gradient/diagonal/PCG andLMcontrolspassed; whitenedinitial
objective agreeswithbaseline<=1e-8; finaloriginalmetricreplay<=1e-10; no accepted
objectiveincrease>1e-10. B: finaloriginalmetricrelative stationarity<=1e-4,
maxgradient<=1e-7 and lastfive loggedobjectivechecks change<=1e-5 ofcapture.
C: finalpenalizedobjective<=.9141457259720446 andcancellation<=10. C meansreference
quality, not speedup/globaloptimum. Changedsolvernegative requires curvature,
numericalresolution andremaininggradient audit before structuralinterpretation.

Theory mapping: https://arxiv.org/abs/1910.12331 uses tensor-contraction implicit
Gauss–Newton/CG forCP. We adaptto partialsymmetry andexplicit augmentedcomponent
residuals; verifythesechanges against a denseJacobian. Convergenceguarantees or
performance numbers fromthatpaper are not automatically transferred here.
