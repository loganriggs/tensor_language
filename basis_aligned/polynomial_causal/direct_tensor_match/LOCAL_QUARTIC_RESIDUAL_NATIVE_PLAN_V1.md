# Native output-local quartic residual fitting

22 September2026,02:28UTC. Exact residual objective and native96atom gradients passed quadrature/densecoefficient/finite-difference checks. Learn genuinely new quartic products for the12smaller outputcoordinates; preserve the seed1001CP512 parent's coordinates0–3 exactly at the selected polynomial interface. This is a staged capacity addition, not a final smaller-than-parent claim.

Architecture: eight new quartic atoms per output,96atoms total. Each atom is a product of four normalized linear input forms; each has one output coefficient and a fixed destination among coordinates4–15. Add288variableproducts and442,464floatingcoefficients to the parent (total1824products2,828,384floats including commonwriter). Learned additions preserve homogeneity/evenness, unlike the earlier localquadratic correction. Output-local readouts deliberately trade possible sharing for protected destinations; later graph refactoring can recover reuse if candidate products prove useful.

Four matched arms: Adam and Muon, each two random starts25001/25002,250updates. Choose one rate for EACH optimizer from the completed500-step toy sweep by smallest median populationerror across allfivefamilies/twostarts; ties choose smaller rate. Freeze rates in input manifest beforeenqueue. Native rows do not select hyperparameters. Same constant rate, fixed250steps, select checkpoint by its own fittingobjective only. Native parent fixedseed1001 (otherparentsuntested atthisrung).

Objective: exact expected squared native-minus-parent-minus-correction under existing calibrationGaussian, plusridge1e-6 readouts. Profile each8coefficient readout separately. Outputweights are the existing calibration inverse-energy capped1000 metric, sliced4–15 and normalizedmean1. Those data-derived statistics define the weightmatching metric; no textoutput labels train coefficients. Pure coefficient+Gaussian gradient machinery was independently checked, but this particular native rung usesGaussian-only; priorcoefficient-focusedbranches remaincontrols, not silently substituted here.

Preregistered screen:
- Integrity: normalresidual<1e-8, finitegradients, exportedpredictionreplay<1e-4, dominantcoordinateassemblybitwiseunchanged.
- Learning: everyarm selectedobjective improves at least10% relative to absolute initialobjective; record if initialobjective nearlyzero.
- Primary response/value: BOTHrestarts of at leastoneoptimizer reduce smallfeatureRMS values AND existing30-pair all-small-output responses by>=15% versus frozenparent on original opened2048state panel.
- Absolute smallfeature10%bar reported separately; relativeimprovement cannot establish accurate components.

Opened256document/2494matchedpair followup separate afterterminal. No freshvalidation, fullmodelintervention, semantic, OOD orcompositionclaim. Ifscreenpasses, the addedproducts become candidates for graphsharing/pruning and nativefiniteeffects; iffails, retain optimhistory and do not call it evidence of impossibility. GPUexecution onlymanagedqueue, allsourcehashesfrozen, no livehelperchanges.
