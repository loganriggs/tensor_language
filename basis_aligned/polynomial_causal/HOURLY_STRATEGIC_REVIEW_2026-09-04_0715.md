# Hourly strategic review — 2026-09-04 07:15 UTC

## Circuit interpretation targets

A proposed decomposition is useful only insofar as it advances these seven targets:

1. **Computational specification:** identify what information is read, what operation is performed, what is written,
   and which later computations use it.
2. **Cross-module grouping and within-module splitting:** group pieces of different heads/MLPs when later computation
   treats them as one variable, and split a native module when its pieces serve different operations.
3. **Held-out and OOD prediction:** predict activations and causal effects on untouched prompts and shifted task forms.
4. **Extraction or sufficiency:** reproduce the target computation or signed effect with an executable circuit or a
   precisely declared interface plus background computation.
5. **Selective manipulation:** swaps and removals affect the intended behavior while preserving unrelated circuits;
   interactions and redundancy must be measured rather than added by assumption.
6. **Composition and reuse:** recovered computations combine predictably and shared parts transfer between tasks.
7. **Stable identification:** the units survive data splits, fitting runs, and internal gauge changes, or are defined by
   downstream operational equivalence.

The program goal remains a smaller executable tensor program satisfying these properties.  Activation rank, retained
variance, CE damage, or parameter count alone does not identify a circuit.  Rank is relevant only when it prices an
already specified input-output computation, as in the predictive-state construction, rather than serving as the
meaning of the computation.

## What changed in the hour

### Task 21 completed as a valid native-capability failure

The exact reviewed task-21 adapter was authorized for one SHA-bound managed enqueue.  The runner verified adapter SHA
`43564464637c7c0fa7a609ec55bc05377c1d872ad0d0cdf1ef80e957e5026779` before parsing and executing it.  The result is a
complete receipt-last package with 24 evidence files, exactly 8 forward calls, 168 row-side evaluations, 1,344 raw
numeric bytes, no backward calls or updates, and no opened SELECT/TEST/OOD phase.

Independent reconstruction gives:

| cell | strict accuracy | mean answer-minus-max-foil margin |
|---|---:|---:|
| base A1 | 21/21 = 100% | +3.236 |
| base A2 | 21/21 = 100% | +3.236 |
| base P | 21/21 = 100% | +3.157 |
| base C | 21/21 = 100% | +3.236 |
| donor A1 | 21/21 = 100% | +3.100 |
| donor A2 | 12/21 = 57.14% | +0.685 |
| donor P | 20/21 = 95.24% | +2.835 |
| donor C | 21/21 = 100% | +6.229 |

Base-wide accuracy is 100%; donor-wide accuracy is 74/84 = 88.10%, below the frozen 90% gate.  Donor A2 is also far
below the 85% cell gate.  Both side-wide mean margins remain positive.  The strict linked-cell criterion therefore
caught a localized instability that an aggregate confidence or average-accuracy screen would obscure.

A1 and A2 use the same answer/foil token set in every one of the 21 groups.  A1 replaces the entire trailing run and
passes 21/21; A2 leaves the old target visible and replaces only the newest two-token run, losing 2.415 mean-margin
units and failing nine rows.  This ties the failure to the intended context manipulation rather than different token
roles or foil construction.  The evidence retains only the maximum foil logit, not its identity, so it does not prove
that the old target was the winning foil.  The independent post-execution audit is still completing its durable
receipt, but its reconstruction agrees exactly.

The registered consequence is terminal: task-21 localization, the full 21-candidate predictive-state analysis, and all
later phases remain closed.  The predictive-state method transfers to the next capable behavior; it is not a reason to
relax task 21.

### The next strict behavior reuses old subject–verb agreement work

A CPU-only dossier audit found three important prior results:

- an 80-row subject–verb agreement screen reported 100% overall and 100% on 40 examples where the nearer noun had the
  wrong number, with mean answer margin +3.769;
- earlier residual-state swaps suggested an early grammatical-number state formed by layer 1, carried through the
  middle stack, and consumed near layer 11; L11H3 contributed but was redundant and was not sufficient alone; and
- removing the previously proposed natural-text ensemble `{11.3, 15.5}` changed `is/are` accuracy by only 0.0052,
  refuting that ensemble as the agreement circuit.

These are task-selection priors and hypotheses, not strict four-phase evidence.  The next authority is being built
from linked transformations that separately change subject number, attraction structure, an answer-preserving
distractor, and surface form.  Its OOD phase uses longer/fronted templates to test fixed-position shortcuts.  Existing
results will be registered in the dossier so they are reused rather than rediscovered.

### The diagnostic roundness line supplied warnings, not adoption

The exploratory lane localized a numeric-roundness feature to an attention-8 direction and showed that downstream
MLPs are needed to turn it into a decision.  Subsequent controls found that whole-component removals saturated their
margin metric and that a copy control was weak on part of the behavior set; calibrated repeats preserved the negative
selectivity conclusion, but several intermediate headlines were withdrawn or narrowed.  A separate audit also found
five published GPU-price lines copied from budgets rather than receipts and corrected them.

The useful lessons are procedural: answer-preserving controls must be natively capable; intervention magnitude must
stay below outcome saturation; and literal prices must be generated from receipts.  The line is still diagnostic and
does not satisfy the strict four-phase circuit contract.

## Confound and shortcut audit

- **Task-21 aggregation:** positive mean margins did not rescue a failed cell.  Future tasks retain linked-cell gates.
- **Foil identity:** the present two-number evidence cannot say which wrong answer won.  Rich localization evidence may
  retain all registered candidate logits, but only after native capability.
- **Local repetition shortcut:** task 21 could be solved from the previous token; it failed before this alternative
  needed localization.
- **Agreement lexical shortcut:** singular/plural answer tokens, noun vocabulary, and template family must be balanced
  by phase and semantic role.  Joint tokenization and equal intervention positions remain mandatory.
- **Agreement position shortcut:** the OOD grammar must move the controller and add attractors, not merely swap words in
  the FIT template.
- **Module attribution:** old layer-1/layer-11 results choose hypotheses only.  They cannot preselect a successful
  circuit without fresh interchange, necessity, sufficiency, and unrelated-behavior controls.
- **Rank drift:** no covariance/eigenvector sweep is licensed by either capability result.

## Ranked alternatives

1. **Repaired subject–verb agreement authority.**  Highest information because archived native capability is strong,
   the controlling noun is not the final token, and old layer-1/layer-11 hypotheses can be tested without assuming
   them.  Kill or redesign before execution if linked counterfactuals cannot balance number, attractor, answer tokens,
   and positions.
2. **Remote copy/induction repair.**  Scientifically closer to the desired shared selector/content decomposition, but
   the R593 transaction check needs a prospective numerical redesign before reuse.  This follows agreement unless the
   agreement generator is vetoed.
3. **Reader-defined predictive state.**  Once a strict task passes capability, retain its full registered answer
   response, identify states by future/intervention equivalence, then map successful interchanges through exact
   bilinear reader terms.  This is a localization method, not a behavior to run in advance.
4. **Attention-5 constant write inside the global compiled program.**  It may reduce the frontier implementation price,
   but a constant approximation does not name a circuit.  Keep it as a priced compilation control, not the main
   interpretability route.
5. **More roundness/head rankings.**  Deprioritized until the existing diagnostic behavior is rebuilt under capable
   controls and the strict phased contract.

## Decision and immediate action

Task 21 is closed unchanged at a valid `hard_abort`.  The active action is a CPU-only repair of the subject–verb
agreement authority after a dossier-first audit.  If its generator and adversarial semantic tests pass, it will enter
the same immutable compiler → independent review → blocked producer → review → authorization → final review → one
managed capability run sequence.  No agreement localization or later-phase data is licensed in advance.
