# Joint full-U structured bilinear weight fit

Four full1152-wide bilinear branches with12mixed-radixlinear maps. Each map
uses seven radix2and two radix3stages. Total276480trainablecoefficients,
4608products, allowedfullinput/outputranks. Nativeunembedding/bias/RMS/background
remain charged. Explicitstageexecution requires281088multiplications and
155520additions for this bilinearcomponent, excludingunembedding/background;
these are arithmeticcounts, not a measuredlatencyclaim.

Two initializations: seed0uses nativeproductorder; seed937permutesall4608units
beforepartitioning into4groups. The native tensor is identical under this
paired Left/Right/Down permutation. Each structuredmap starts as orthogonal
smallstages scaled to the corresponding nativefactor's Frobeniusnorm.
Fit the12factor matrices with Adam(lr=.03), <=300steps/60seconds. This is only
initialization; nativeunitordering is not assumed to identify circuits.
Then solve4branch amplitudes exactly under the eventual tensorobjective,
absorbing the scalars into outputstages without extraprogramparameters.

Joint objective: normalized full-U Frobeniuscoefficient residual plus .01sum
of individual productcomponentenergies. Use exact chunkedCP loss and analytic
factor gradients, then exact autograd through the learned smallstages. Every
token and all4608products are included; no sampledcoefficient/sketch/data loss.
Chunk256; fullnativegradient compared withchunk512 and parameterdirectional
finite differences beforefitting. No denseJacobian orfulltokeninteractiontensor.

Nativejoint optimizer: persistent Torch L-BFGS history20, strongWolfe, lr1,
oneouterstep percall, up to2000outersteps/300fitseconds perstart. Internal
absolute stopping disabled; externalconvergence requires relativeparameter
stationarity max_p||grad_p||max(||p||,1)/capture<=1e-4, gradientmax<=1e-7,
and relativeobjectivechange overlast5diagnostics<=1e-5. Diagnostics every5steps.
A time/step/linesearch limit is not convergence. Preserve model and optimizer
state in/dev/shm for unfinishedstarts. Frameworkline search may finish its
currentstep beyond a softtimebudget; report actualtime.

Pred_a: CPU dense/analytic andparameterchain controls<=1e-10; nativechunk
loss/gradientrelativeconsistency<=1e-6, nativeparameterFD<=1e-5, branchsolve
relativeerror<=1e-8, finitefinalstate. Pred_b: bothstarts meetallthree external
convergenceconditions. Pred_c: bothcapture>=1.05times.08634383041327387.
Report unpenalizedcapture, penalty, cancellationratio andbothinitialization
losses separately. Penalty granularity andproductcountdiffer from earlier
smallmodels; no claim of a perfectlymatchedregularizer comparison.

Native finite differences use the normalized parametergradient direction,
scaled to the current parameternorm, with central relative step1e-4. Save the
initial model and preflight receipt before scoring, so a failed numerical
preflight can be diagnosed without repeating the factorinitialization.

Null: this wiring/initialization doesnotfind a good compactnativeprogram, or
optimization remainsunfinished. Red-team ordering, initialization, gauge and
convergence before a structuralnegative. The planted3starttest alreadyhasone
nonzero stationaryfit; no globalrecovery guarantee. Circuit target: reusable
shared linearcomputations andfull-rank productunits. Reconstruction alone
doesnotestablish OOD, extraction, selectiveremoval orcomposition/reuse.
Managedlane1only, sourcehashbound aftercontrols/dryrun. No corpusaccess.
