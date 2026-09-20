# Four-output source observable and selectivity

The validated three-source agreement quadratic is extended to four outputs: correct is/are margin, can/will, may/might and should/could. These are specific collateral controls, not universal unrelated behavior. Inputs are amplitudes of middle-writes4–7, MLP8 and MLP10 port edits at a subject or attractor site before block11. Native residual reinjection and first-value cache are held fixed. The opened96 contexts include known native agreement capability failures; those are retained separately.

For each output o, damage is approximated by -g_o^T*a-(1/2)*a^T*H_o*a. The nine amplitude arms include six independent single/pair directions, unit, negative and mixed. This set covers the symmetric quadratic coefficient space. Doubled-amplitude failure is already known and remains a limit; the present experiment does not claim to repair it.

Preregistered gates: float64 functional effects match native float32 effects within absolute1e-4; directional finite-difference gradient1% and Hessian15% at steps.1/.2; Hessian symmetry1e-4; exact call counts. Prediction requires number error<=10% of each arm's number-effect norm, and each modal error<=5% of that norm, in every context cell. Selectivity separately requires each native modal-effect norm<=10% of the arm's number-effect norm. Passing prediction does not imply passing selectivity. Both denominators and absolute effects are retained in the receipt.

The native_source_observables.py implementation preserves explicit Down_bias, float32 RMS epsilon, rounded native rotary constants, all two-QK factors, cached first values, residual lambdas and final softcap. It returns only four selected readouts. Full native controls independently run the original modules. All baseline gradients and Hessians are generated from weights/operations, with no fit to effect labels.

Cost24prefix,224double-reference suffix and160native suffix batch forwards;64gradient and192Hessian-row reverse passes. Conditional coefficient runtime is36 values/context: twelve linear coefficients and24 distinct symmetric quadratic coefficients. The same three linear source amplitudes and six quadratic monomials can be shared across four outputs. All native source/context/derivative generation remains charged; this is not a standalone model or a native replacement.

Runner: ../bilinear_quotient/ops/run_semantic_source_multiobservable_v1.py. Result: ../bilinear_quotient/circuits/followups/semantic_source_multiobservable_v1_result.json. Full quadratic and linear modal predictions are both retained, allowing an explicit simpler baseline.

## Results

Instrument passes: native effect replay6.07e-6. Prediction passes every registered cell/arm: number max9.0965%, modal max2.8209% of number effect. Linear modal prediction reaches12.68% and fails the5% bar. Thus the four-output quadratic is more useful than the earlier linear collateral baseline on these source settings.

Native selectivity fails in129of288 arm/context cells. Worst collateral83.56% occurs for opposite-number attractor MLP8-only removal in near_greeted plural: agreement RMS0.00172, three modal RMS values0.000706/0.000568/0.00144. This is partly a small-effect denominator issue, but the failure is not confined to tiny singleton effects. For the combined unit B edit, maximum collateral ratios are9.95% opposite subject,10.34% matching subject,29.70% opposite attractor and48.60% matching attractor. Prediction accuracy does not make these native edits selective. Existing native agreement capability failures remain.

## Local selectivity bound and a failed candidate

Let n be the three-source gradient of agreement damage and M the3x3 matrix of modal damage gradients. For full-rank M, minimizing ||M a||2 subject to n dot a=1 gives b=M^{-T}n, a*=M^{-1}b/(b dot b), minimum leakage1/||b||2. This is a restriction to infinitesimal edits in these three source directions, with no amplitude bound. If the minimum exceeds sqrt(3)*0.1, it is impossible for all three modal ratios simultaneously to be below10% in that local linear model. It is not a global impossibility theorem for circuits, larger source interfaces, finite normalized dynamics or other collateral tasks.

All192 modal gradient matrices are numerically full rank under a1e-12 relative singular-value check. Constraint residual2.35e-14. Twelve contexts have a local lower bound above sqrt(3)*10%; median bound3.32%, maximum114.84%. See [bound receipt](SOURCE_SELECTIVITY_BOUND_V1.json).

A cheap candidate scales a* to the unit B quadratic number effect, then uniformly clips max amplitude to1 and evaluates the full quadratic. None of32 context cells retains80% aligned number effect while meeting all three10% collateral bars. This rejects that particular clipped linear-optimum proposal. It does not prove a constrained nonlinear optimum cannot succeed; clipping can discard the target effect. Do not spend native validation on a candidate that already fails its predictor screen. [Candidate audit](SOURCE_SELECTIVITY_BOUND_CANDIDATE_CPU_AUDIT.json).

The next interface question is whether the omitted embedding and early-write sources supply additional directions that preserve modal outputs while carrying number information. With five sources and three independent modal linear constraints, a nontrivial nullspace exists, but agreement may have zero or poorly conditioned projection onto it; that must be measured. Any proposed finite edit still needs native validation, retained number strength, unrelated-behavior controls and literal source-generator pricing. The next weight-folding handoff should target these readout-contracted source directions rather than another unconditional rank sweep.
