# Receiver-only tangent for the mixed MLP4-to-value computation

10 September2026, before native execution. Both intervening-chain deletion
hypotheses failed; preserve attention AND MLP computations. Test whether
their finite local-value response can be predicted by the derivative at the
receiving context, without supplying the donor-induced intermediate states.

Same72 opened pairs, all valid MLP4 output positions, both swap directions.
Let f(m) execute the ORIGINAL native backend with MLP4 output replaced bym,
and return layer9 raw localV9 at heads1/4. For receiving native source m0
and paired source change delta_m, candidate V=V_native+Df(m0)delta_m.
Use torch.func.jvp with forward-mode derivatives through native/no_grad
execution; capture the value output WITHOUT detach. All prior components,
keys and first-value payload follow the native computation in this derivative.
The final value-reader assay inserts the predicted field in receiving-native
heads1/4, keeping their query/key/background native and recomputing suffix.

Derivative uses only the receiving trajectory; donor MLP4 output supplies
the intervention vector, not intermediate derivative coefficients. No fitting,
step-size selection, source selection or midpoint derivative. An exact
pair-conditioned secant is an oracle idea, not the candidate being tested.

Perpanel:2 nativecaptures,2 fullMLP4swap captures,2 JVP backend evaluations,
2 fullvalue reader forwards,2 tangent reader forwards,2 identity forwards.
48 model forwards/864 sequence evaluations including8 JVP evaluations on144
sequences. JVP compute cost is additional and reported; no savings inferred.

A: bind parent/source/rows/backend/checkpoint; tiny ORIGINAL native backend
JVP versus centraldifference maxabs<=1e-8 ANDrelative<=1e-6; zero-direction
and live-direction controls. Native JVP primal field and identity logits
maxabs<=1e-3 ANDrelFrob<=1e-5. Fullvalue margin effect/endpointnorm replay
parent atsamebars/1e-5relative. Exactcounts, source masks, unchangedunselected
valueentries, finite and restoredhooks. No newforwardimplementation.

B finite-response prediction: tangent value-reader effect relativeerror
versus fullvalue<=.10 in BOTH centered full-logit and answer-margin frames,
every panel/direction. Reference norm<=1e-8 requires absoluteerror<=1e-8.
C live model-level perturbations: fullvalue and tangent centered-logit
effect norms>1e-8 everycell. Save actual/predicted margin vectors and norms.
A failure invalidates; B miss remains a null, no dose or fitted gain rescue.

This is a receiving-context conditional predictor of an EXISTING PARTIAL
path. The previous has full-logit carrier miss stays failed. A passing screen
still needs fresh contexts, genuinely new edits and joint-intervention tests.
All545902902 native parameters, receiving computation and source generator
remain charged; zero actual weight saving, no trained coefficients/adoption.
600-second watchdog, managed enqueue only, immutable result JSON.
