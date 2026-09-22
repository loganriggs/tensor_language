# Residual input sensitivity is diffuse and strongly metric-dependent

22 September2026,01:21UTC. CPU analytic Jacobian audit of selected MLP16→17 purequartic16outputmap and frozen mixedCP512seed1001/1002. One position31 from each of256already-opened newdocumentprefixes. This is local sensitivity, not intervention or feature attribution. No fitting.

For native F andcandidate Fhat define J=derivativeF/dx, E=derivative(Fhat-F)/dx. At normalizedinputx use tangentprojector

$$
P_x=I-\frac{xx^T}{x^Tx},\qquad J_t=JP_x,\qquad E_t=EP_x.
$$

Report pooledFrobenius gradienterror acrossstates,outputs,inputdirections. This removes pure radialrescaling; it is not the fullmodel inputJacobian anddoesnotdifferentiate through upstream stateconstruction. RMSNormepsilon and upstream coupling remainoutside thislocal polynomial geometry.

| Measurement | CP1001 | CP1002 |
|---|---:|---:|
| Value error on these256states |8.672%|8.728%|
| Ambientinputgradient error |25.192%|25.546%|
| Tangentinputgradient error |26.265%|26.636%|
| Tangentgradient error weighted by calibration covariance |9.841%|10.023%|
| Balanced-output tangentgradient error, isotropic inputs |59.996%|60.711%|
| Top64originalinputcoordinates' share of residual tangentgradientenergy |6.940%|6.919%|
| Top4learnedinputdirections' share of residual tangentgradientenergy |6.506%|6.448%|
| Top64learnedinputdirections' share |35.163%|35.269%|

The reference native tangentJacobian's leading4inputdirections contain76.49%ofitsenergy, whileleading64contain90.92%. Residualeffectiveinputrank, defined(traceK)^2/trace(K^2)forK=sum E_t^TE_t, is281.36/283.36, versus3.39native. This is an empirical derivativeenergyconcentration statistic, not tensor rank, circuit lowerbound, requiredchannels, or generalization guarantee.

Originalinputcoordinates alone are particularly diffuse: top64of1152captureonly~6.9%, near their5.56%countfraction. Rotatingintolearneddirections revealsmorestructure, butnoverylow-dimensionalresidual. Topcoordinates645,981,329,990are repeatably largest yet eachcarries<.3%ofresidualgradientenergy; naming these as a few responsible variables would be misleading. This doesnot ruleout compactnonlinearfeatures or local features varyingacrossstates.

Covariance comparison: S is the existing fitted calibration covariance square-root map (historically named'whitener' inGAUSSIAN_CP_DATA_PROJECTIONS_V1.pt). We calculate||E_t S||/||J_t S||, equivalently perturbdelta=P_x S z forunitisotropicz. It is much lower than isotropic tangenterror. Thus the26%figure emphasizes directions that varylittle under the calibration metric; the~10%figure is more alignedwith thatinputdistribution. Neither substitutes for actualsame-tokenstatechanges, finiteinterventions orOOD. The mean shift selectswhereJacobiansareevaluated; itdoesnotenter thederivativecovariancefactor directly.

Outputbalancing uses the calibrationmetricpreparedfor the queuedexperiment: inverseoutputmeansquare with1000ratio cap. Its~60%derivativeerror is notinconflictwith26%: it gives smalloutputcomponentsmoreweight. This reinforces thedecision toreport bothmetrics ratherthanselectthemoreflatteringone.

Implementation: nativechainanalyticJacobian andCPproductrule checkedagainstautodiffon5structures (independent,sharedinput,sharedoutput,squares,cancellation), maxrelativeerror<1e-12. NativeEuleridentityJx=4F agrees2.48e-15; centralfinitedifferences at1e-3/1e-4 agree<8e-10. NativeFP64values replaystoredFP32targets1.11e-6relative. NoGPUwork. Fullreceipt includesall16featurederivativeerrors.

Scientific consequence: lowoutput-residualrank doesnotimply a cheapread-sidecorrection. Themissingamplitudesdependlocallyonmanydirectionsunderisotropicgeometry, whiledatacovarianceheavilydownweightssomeofthosemisses. Favor native-responseconstrained discovery orcovariance-awarederivative tests overassuming a fewcoordinatefixes. The alreadyqueuedbalancedproducerfit stilltests capacityallocation; neitherdiagnostic guaranteesitwillwork.

Files: audit_residual_input_sensitivity.py, RESIDUAL_INPUT_SENSITIVITY_V1.json. Rawcoordinateconcentration isgauge-dependent; tangentGram eigenvalues are invariantunderorthogonalinputrotations withtheEuclideanmetric. Finitepanelestimates, no confidenceintervals orcausalclaim.
