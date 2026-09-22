# Conditional programs save storage, but fail the native removal criteria

22 September 2026, 02:41 UTC. **Both conditional-program variants passed their numerical checks but failed the registered native-intervention criteria.** The full conditional program and the lean graph-pruned version behave almost identically here. Their lower storage and CPU runtime remain real, but neither can be promoted as a faithful native intervention substitute.

## What was changed inside the model?

We tested removal of output coordinate 1 of the selected pure quartic MLP16→MLP17 path. Coordinate 1 is one fixed learned output direction, not an identified semantic feature. At each token state, the native polynomial supplies a scalar amount f and the candidate supplies an approximation f-hat. Both write along the same fixed residual-space direction w.

The intervention subtracts a quarter or all of that contribution from the final residual state, retaining the actual MLP17 normalization denominator and the other native branches. The resulting residual is passed through the actual final RMS normalization, unembedding and logit softcap. We compare the candidate's final-logit change with the change produced by removing the exact native contribution.

The panels contain 16 FineWeb prefixes and16 Python-standard-library prefixes, with positions16–254 evaluated. Results are separated into newline-containing tokens and other tokens. **The 16 result cells are two candidate seeds × two domains × two removal strengths × two token strata. They are not16 examples or16 separately tested output features: this experiment intervenes on coordinate 1 only.** Each domain contains 3824 evaluated token states per seed/strength; states within prefixes are correlated. These panels were previously opened, so this is not fresh OOD confirmation.

## What were the success criteria?

The absolute criterion required at most10% relative final-logit-effect error in every cell. The retention criterion required each cell's error to be at most1.10 times the corresponding frozen CP parent's error. The parents already failed absolute fidelity and semantic-selectivity checks, so retention alone would still not establish a circuit.

| Candidate | Cells above 10% absolute error | Cells exceeding parent-retention limit | Worst absolute error | Worst error / parent error |
| --- | ---: | ---: | ---: | ---: |
| Full conditional | 5/16 | 4/16 | 13.86% | 1.214 |
| Lean conditional | 5/16 | 5/16 | 13.95% | 1.188 |

The numerical instrument passed: independent native-path replay error was below 4.8e-8, reader replay below 2.4e-7, zero-removal edits were exactly zero, and reference-effect energies reproduced the parent test exactly. Each test took about 1.8 seconds inside the runner, excluding queue waiting. These checks support a real negative result under this protocol rather than an obvious frame or reference mismatch.

## Where do the failures occur?

The absolute failures occur in the non-newline strata: four FineWeb cells and the quarter-strength code cell for seed 1002. Errors in the FineWeb non-newline cells remain roughly13–14%.

The retention failures show a different pattern. On FineWeb newline tokens, the conditional programs **improve** on the parent: for seed 1001 at full removal, error falls from 7.59% to about 4.0%. On code newline tokens, they **worsen**: seed 1001 full-removal error rises from 5.30% to6.44% for the full program, or6.30% for the lean one. Those remain below 10% absolute error, yet exceed the allowed10% relative degradation from the parent. The lean program also narrowly fails retention for seed 1001's quarter-strength non-newline code cell.

Thus a favorable aggregate value score does not preserve all intervention strata. Both positive and negative changes matter; reporting only FineWeb newline improvement would hide the failed retention requirement.

## What this changes about the research direction

The comparison separates two stages. Most of this failure already exists in the full conditional approximation; deleting its extra correction nodes to make the lean graph introduces little further change. The graph edit can preserve its fitted function while that function remains an inadequate native substitute. Parent-Gaussian accuracy is therefore insufficient for adoption.

The queued output-local residual fit addresses a different weakness: coordinates4–15. By design it preserves coordinates0–3, so **it cannot repair this coordinate 1 removal failure**. That experiment remains useful for component recovery, but must not be presented as the answer to this intervention result. A future coordinate 1 repair must directly address its own finite-response fidelity and domain/context variation.

The separate reader-sharing graph edit remains a successful parent-preservation screen, with about 6% array-storage savings; it inherits the CP parent's native limitations. Neither result supplies semantic selectivity, fresh/OOD transfer or composition. The full research goal remains unfinished.

[Cell-by-cell comparison with parents](../../direct_tensor_match/CONDITIONAL_REMOVAL_COMPARISON_V1.json) · [Full-program receipt](../../direct_tensor_match/CONDITIONAL_CP_REMOVAL_V1.json) · [Lean-program receipt](../../direct_tensor_match/LEAN_CONDITIONAL_CP_REMOVAL_V1.json) · [Registered full-program protocol](../../direct_tensor_match/CONDITIONAL_CP_REMOVAL_PLAN_V1.md) · [Registered lean-program protocol](../../direct_tensor_match/LEAN_CONDITIONAL_CP_REMOVAL_PLAN_V1.md).

## Follow-up: are a few documents causing the failures?

The answer differs by stratum. In the lean program's four failing FineWeb cells, **9–14 of the16 documents** individually exceed10% relative error. The two documents contributing the most absolute squared error account for **33–51% of error**, versus **18–21% of reference-effect energy**. Thus there is a heavy tail, but the failure is not confined to one or two prefixes.

For seed1002's failing quarter-strength non-newline code cell, **14 of16 documents** exceed10%. Its two largest error contributors account for22% of error and23% of reference energy. That failure is comparatively broad and uniform across these documents. These are document aggregates on an opened small panel, not evidence that individual token errors are uniform. [Document-level diagnostics](../../direct_tensor_match/REMOVAL_DOCUMENT_TAILS_V1.json).

A follow-up stage-by-stage test is prepared to distinguish raw scalar prediction error from reweighting by the MLP normalization denominator, final RMS normalization, unembedding and softcap. It will replay the completed final-effect scores and cache the native interfaces for reuse. These stages use different norms; their errors must not be added as if they were independent components. [Protocol](../../direct_tensor_match/REMOVAL_STAGE_GEOMETRY_PLAN_V1.md).
