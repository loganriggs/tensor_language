# Native sparse-frame Riemannian CG — 10 September22:46

Weight-only centeredoutputtensor, immutable spectral128frame,256edges. Exact
commonoutputquadratic remains separate andpriced664128symmetriccoefficients;
no corpusdata or silentdiscard beforeRMS/tanh. No coefficientenergy penalty:
orthogonalfeatures have no cancellation and exactconditionalwriterprojection.

Optimize actualtop256edgecapture onStiefel frames Q^TQ=I. RiemannianPR+ with
projectedvectortransport, beta clamped0..10, directionreset unless gradientinner
product>=.1*gradientnorm^2. QRretraction; oldsupportArmijo sufficientascent with
coefficient1e-4,25backtracks,halving. Reselectioncannotlowerthat fixedsupport
score, and actualselectedscore monotonicity ischecked. Initialstep predicts1%
relativegain, laterhint1.5xacceptedstep, displacementbounded<=1Frobenius and
stepsize<=1e6. These are registeredalgorithmchoices, not a globaloptimumclaim.

A: existingdense/finitegradient/ascent/resumecontrols pass; nativeinitial score
matches CPUbaseline<=1e-10; everyframeorthogonality<=1e-10; scoredecrease<=1e-10;
finalwriterreplay<=1e-10. B: fivechecks relativeprogress<=1e-5, tangentgradient
norm*sqrt128/capture<=1e-4, maxgradient<=1e-7, supportgap>1e-12 normalizedenergy.
C: centeredcoefficientcapture>=.05. Reportunconverged andsupportties explicitly.
Originalspectralbaseline AheldB/Cfailed remainsunchanged.

Initial240s fit, timerbetweeniterations,900s alarm. Save~6MB resumableq, previous
gradient/direction, stepsize andhistory, plusfinalexactwriters. Require20MBfree.
Centeredcandidate442368floats+512indices; commoncomponentextra, othernative
background remains. Scorealone doesnotestablishsemanticcircuitsoradoption.
