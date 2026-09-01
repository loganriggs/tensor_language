# Plain-English update — 2026-09-01 01:43 UTC

(Damage means extra next-token cross-entropy above the native model; **lower is better**.)

## The goal

We are compiling the 546M-parameter bilin18 language model into a substantially smaller, explicit tensor
program. Success is not just a lower parameter count. The program must preserve predictions on new text,
compose when its replacements are installed together, reproduce the effects of named interventions, and
pay literally for every stored value and operation.

The immediate milestone is a compiled point that improves the present prediction/certificate frontier
without losing the intervention fidelity already measured. The larger goal remains a program whose parts
are understandable and editable, not a downstream imitation of the model's answers.

## The important correction: the 0.055 attention floor was our block-1 value table

For many experiments, every compressed attention configuration appeared to pay almost exactly 0.055 CE.
The damage vectors were nearly parallel and the same 11 of 62 circuit certificates survived. That looked
like a delicate model-wide mechanism stored in the smallest score-map directions.

A full-rank control falsified that story. Even when every QK score direction and the current values were
nominally restored, the replacement path still cost 0.052 CE. Two independent script lineages reproduced
that number exactly. The common error therefore came from something shared by the scripts, not from the
directions removed by compression.

Codex then audited what the "values real" arm actually installed. It still contained `a1v`: a table indexed
only by token ID that replaced block 1's value map. This is not an exact substitution. Block 0 has already
mixed information across positions, so the state entering block 1 depends on the prefix. The native map is

$$
v_1(t,\text{prefix}) = W_{V,1}\,h_1(t,\text{prefix}),
$$

whereas the table can produce only

$$
\widehat v_1(t,\text{prefix}) = T[t].
$$

Two occurrences of the same token in different contexts must receive the same table row even when their
native values differ. The control removed only this table and let the native block-1 linear map execute.
Everything else stayed the same, including full-rank factored QK recomputation.

The result was effectively exact:

- census damage: `-0.0000`;
- eight independent fresh windows: `+0.0000` in all eight;
- circuit certificates: `62/62`;
- remaining damage-vector norm: `0.000226` times the old path norm.

So the apparent global attention mechanism was almost entirely a context-blind value-table error. It was
not SVD arithmetic, floating-point operation order, or a fine-band score invariant.

## What survives after the correction

Subtracting the full-rank path vector from the old receipts gives a useful pre-outcome prediction for the
clean reruns:

| Configuration | Old measured damage | Path-corrected prediction | Predicted certificates |
|---|---:|---:|---:|
| ct96 | `+0.05530` | `+0.00327` | `56/62` |
| tail120 | `+0.05259` | `+0.00057` | `62/62` |
| pure value-r96 | `+0.06992` | `+0.01790` | `41/62` |
| mixed value-r96 | `+0.07628` | `+0.02425` | `33/62` |

These are algebraic companion estimates, not substitutes for clean execution. The corrected ct96 run is
now queued with those predictions registered in advance.

The function-space picture also becomes more modest and more coherent. Before path subtraction, ct96 and
tail120 damage had cosine `0.984`. After subtracting the shared path vector, their residual cosine is
`0.738`, and their mean costs are tiny. There is no longer evidence for a dominant rank-one attention
floor. Value, attention-rank, and knockout residuals are approximately orthogonal after correction, but a
strict vector-additivity law still misses its registered relative-error bar. That is a useful approximate
accounting basis, not a universal theorem.

## The price improves too

The erroneous token table stores

$$
V D = 50{,}257\times1{,}152 = 57{,}896{,}064
$$

values. The native block-1 value matrix stores only

$$
D^2 = 1{,}152^2 = 1{,}327{,}104
$$

values. Restoring the correct primitive therefore removes `56,568,960` stored values while improving
accuracy. Applied to the approximately 211M-value ct96 anchor, the rough corrected total is 154.4M values.
This needs a final ledger reprice, but the direction is unambiguous: the faithful operation is also much
smaller than the lookup table it replaces.

## Current plan

1. **Measure the corrected compressed model.** Run ct96 with native `a1v`, score census CE, all 62 circuit
   certificates, and eight fresh windows. It must also reproduce the signed vector predicted before the
   run from `cev_ct96 - cev_pathfull`; otherwise path subtraction was not compositional.
2. **Rebuild the Pareto table.** Rerun only the nondominated score/value ranks on the faithful path and
   reprice them with the smaller block-1 matrix. The old 0.055 floor and 11-certificate cap are retired as
   instrument-dominated measurements.
3. **Revalidate interventions on the corrected winner.** Repeat the registered knockout transfer battery.
   The previous baseline-subtracted rankings probably survive because subtraction canceled much of the
   shared table error, but that is a prediction, not a license to skip the test. Adoption requires the
   corrected program to retain collateral ranking fidelity and not worsen the already damped own effects.
4. **Only then add compute sparsity.** The per-token MLP top-k surcharge was additive at four anchors, but
   its absolute frontier must be recomputed after the path correction.

The previously planned rank-16 output-state repair is held. Learning a downstream patch would hide a
primitive that can be made exactly correct at lower storage cost.

## Different paths if the corrected frontier is still not enough

The project should not return to arbitrary rank and head sweeps. Four materially different routes remain:

1. **Tangent-aware compilation.** Fit both ordinary outputs and derivatives along registered knockout or
   activation-swap directions. This directly targets the remaining causal-magnitude undershoot rather than
   hoping predictive compression preserves interventions automatically.
2. **Predictive-state quotient.** Stop copying the native architecture module by module. Identify states up
   to their effects on future logits and controlled interventions, using held-out continuation/Hankel rank
   as the falsifier. This can bypass redundant native coordinates entirely.
3. **Causal-response coordinates.** Choose low-rank bases by preserved signed circuit and intervention
   responses rather than weight energy. The basis must beat ordinary SVD at equal literal price on held-out
   circuits and documents.
4. **Shared invariants, but only with intervention evidence.** Search for centering, normalization, or
   moment identities shared across score maps. A correlation is insufficient; manipulating the candidate
   invariant must move the predicted error while matched-energy orthogonal perturbations do not.

The lesson from this turn is methodological as much as architectural: an elegant low-rank law can be the
fingerprint of a shared instrument. Exact-path controls and live configuration checks must precede any
mechanistic interpretation of a compression curve.

## Result update: the corrected ct96 milestone passed

The queued corrected ct96 run subsequently landed at `+0.0034195` census damage with `56/62`
certificates and `0.0000` damage on each of the eight fresh windows. Its literal storage change is
`-56,568,960` values, giving an approximate total of 154.4M values from the old 211M anchor.

The strong vector-subtraction prediction was close but failed as registered: cosine between the new
damage vector and `cev_ct96-cev_pathfull` was `0.94857` against a `0.95` bar, and relative error was
`0.3227` against a `0.25` bar. Thus subtraction predicted the aggregate and certificate recovery well,
but the nonlinear program is not an exact additive vector system.

The signed m16 intervention sentinel then passed every adoption bar. Comparing the full 256,000-position
causal effects within each model gave:

- effect cosine: `0.996324`;
- normalized effect-vector error: `0.103727`;
- compiled/native effect-norm ratio: `1.054525`;
- collateral circuit Spearman: `0.998223`;
- median own-circuit magnitude ratio: `1.037211`.

These numbers use `CE(model+KO)-CE(model)` directly for both native and compiled models, rather than a
difference of unsigned damage summaries. The receipt was independently recomputed from the three saved CE
vectors and matched the recorded metrics. The corrected ct96 point therefore clears the next verified
milestone across prediction, fresh transfer, certificates, literal price, and signed intervention fidelity.

The next research milestone is no longer “break the 0.055 floor.” It is to rebuild the whole Pareto set on
the faithful path, extend signed intervention validation beyond the m16 sentinel, and decide whether the
approximately 154M-value ct96 point or a cheaper corrected mixed-spectrum point is the best adoption base.
