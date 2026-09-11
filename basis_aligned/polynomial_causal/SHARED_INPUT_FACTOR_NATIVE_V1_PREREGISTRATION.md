# Native one-reader/shared-partner factor

Math0149consequence. Fit a shared scalar input reader with unrestricted linear
output partners, rather than a few global output factors. No data or text.
All50304tokenquadratics,1152inputs,4608nativeproducts; fullandcenteredU objectives.
The extracted residual component always uses nativeDown and originalU.

For unit a, P=aa^T, project everyQ ontoQstar=P Q+Q P−P Q P. This is the exact
post-normalization input edit f(x)−f((I−P)x), without renormalizingeditedx.
It executes as (a^T x) M_a x with
M_a=D[diag(Ra)L+diag(La)R]−D[(La)⊙(Ra)]a^T.
Bias/background remainnative; two orthogonal reader removals need their shared
cross term subtracted once. CPUprojection/gradient/removal/composition controls
pass<=5.1e-16 and randomplantedfactor recovery converges22steps.

Score E(a)=2a^TSa−sum_v(a^TQ_v a)^2, S=sum_v Q_v². Upperbound
E(a)/traceS<=2lambda_max(S)/traceS. Priorverifiedspectra imply2.0673%full,
1.1309%centered ceilings forANYone-reader factor. These are coefficient-norm
limits, not limits on a circuit's behavioral importance or multiple sharedreaders.
Recompute the bound natively and compare, not another global tiny-rank claim.

Fourstarts/objective: topSeigenvector andGaussianseeds1103,1109,1117. Sphere
projectedgradient ascent withArmijosearch,4000steps, tangentgradient/total<=1e-8.
AllupdatesFP64. Saveeachfit'sgradient/history; no convergence fromtimeexpiry.

A: nativeenergy/priorbound replay, score versus partnerweightednorm, and64formal
postnorm-input removal plusorthogonaltwo-reader composition replays<=1e-8.
B: all8starts converge to tangentstationarity<=1e-8.
C: afterA/B, centeredbestcapture>=.008 andall4centeredreaderabsolute cosines
withbest>=.99. Stability is numericalmulti-start evidence, not globaluniqueness.
D: afterA/B, rank16partnerlinearapproximation retains>=90%of centeredstarenergy.
This is the exact best fixed-reader rank16 partner under the same quadratic norm.

ForpartnerM and a, useJ=(I+aa^T)^(1/2). Quadraticstarenergy=.5||Uc M J||F².
WhitenUc usingCholesky(Uc^TUc); spectrumofroot^T M J gives exactweightedrank
partnerapproximation without fullvocabularymatrix. Reportrank90/16capture and
full/centered/common energy forbothselectedreaders. No outputrankcut duringa-search.

Price beforepartnertruncation:1152reader+1152²partnercoefficients; nativeU,
normalization/backgroundretained. Rankrpartner uses2*1152*r+1152numbers plus
rintermediatevalues; reportapproximation separately. Cachetwofullpartners~22MB
and8readers insharedmemory, durableJSONchecksums.

Null: localsearchmiss cannotbeatanalyticupperbound but is not globaloptimum.
Ifstablegate's partnerisdense, it is an explicit sharedinput/port rather than a
small standalonecircuit. CheckexistingMLP17calibration dossier andcommonchannel
before namingthegate; naturalFineWeb/causalvalidation remainslater.
