# Independent sparse-frame restart — 10 September22:58 UTC

Same128reader/256edge centered-weight RCG problem andconvergencebars asV1.
IndependentGaussianQR initialization seed937, frozenCPUscore andframehash;
no data/labels. This tests reproducibility ofthe fittedfunction, not merely
whether its own factorization isunique. Fullcommonchannelremains separate.

A: inherited derivative/orthogonality/resumecontrols plusindependent dense
sparse-function cosinecontrol<=1e-10; nativeinitial agreeswithfrozenCPUscore
<=1e-10, finalreplay/orthogonality/monotonicity<=1e-10. B: same fivecheckplateau,
relativeStiefelgradient<=1e-4,maxgradient<=1e-7,supportgap>1e-12. C: capture>=99%
ofpreviousconvergedcapture AND fullfittedcoefficientfunctioncosine>=.95. Report
qualityandcosineclausesseparately; failuredoesnotproveabsenceofnativecircuits.
No factorpermutation/sign/unusedbasispenalty infunctioncomparison.

240s initialchunk,900s alarm,20MBfreeguard,~6MBresumablecheckpoint. No GPU
warmstart fromfirstfit. Previousfunctionusedonlyasregisteredcomparison target.
Centeredprice442368floats+512indices; commonquadraticadditional664128floats.
Nativebackgroundremains. Globaloptimization/restartstability notestablishedby
onepair; this isfirstindependentrestart in this family.
