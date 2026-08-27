# Polynomial causal track

This directory is a separate research track. It does not replace the
`bilinear_quotient` ledger, circuit registry, or Theseus verifier.

## Question

Can the polynomial / multilinear structure of bilin18 produce a causal
description that predicts interventions better than additive ablation accounting,
and can the resulting variables find circuit joints better than generic low-rank
compression?

The mathematical contract for what is being priced, what counts as an
intervention, and where polynomial reasoning is exact is in [FORMALISM.md](FORMALISM.md).
In particular, simplicity is reported as a conditional vector rather than a single
grammar-free number.

The whole-model strategy, coverage accounting, ranked workstreams, and pruning rules
are in [REVERSE_ENGINEERING_ROADMAP.md](REVERSE_ENGINEERING_ROADMAP.md).

The generated `whole_model_balance_sheet.json` is the north-star accounting view.
It intentionally keeps exact representation, analytic-interface substitutability,
named-variable understanding, causal path coverage, current composite distortion,
and intervention prediction in separate currencies.  Its current operational fact
is not merely that 36/36 top-level modules have replacements, but that the certified
fidelity-oriented ship is still roughly +0.93 CE above the frozen clean anchor.

## Registered evaluation ladder

Every representation is evaluated at matched rank or matched encoded bits. A lower
reconstruction error is not, by itself, a win.

1. **Joint-intervention prediction.** Fit scalar intervention effects on a discovery
   row set. The additive baseline sees clean and singleton cuts. The pairwise model
   additionally sees pair cuts. Both must predict the unobserved three-way cut on a
   disjoint row set. Primary metric: normalized absolute prediction error on the
   pre-softcap true-target logit. Secondary metrics: class CE and global CE.
2. **Coefficient generalization.** Mobius coefficients must keep their sign and at
   least half their magnitude on a second row set. A coefficient that does not
   replicate is bookkeeping noise, not a mechanism.
3. **Norm-gauge audit.** Repeat the intervention ledger with live RMS gauges and
   clean-frozen RMS gauges. Report the norm-mediated remainder for every arm. The
   polynomial claim is licensed only for the frozen-gauge arm.
4. **Circuit discovery.** At matched rank/bits, compare candidate bases by precision
   and recall against held-out certified circuit heads, selective-removal CE versus
   random subspaces, and recovery of known writer-to-slice joints. Discovery classes
   and evaluation classes must be disjoint.
5. **Replacement value.** Substitute the selected representation alone and inside
   the current composite. Report held-out global CE, class CE, KL to the clean model,
   and fresh-row replication. Local fidelity that does not compose is not a win.
6. **OOD prediction.** Repeat the prediction and removal legs on a different row
   range and, when available, a different corpus. No refitting is allowed.

## Primary win conditions

- Pairwise prediction error is at least 30% lower than additive prediction error on
  the unseen three-way cut, and its normalized error is at most 25%.
- The improvement replicates on a second row range.
- A discovered basis beats output-PCA, activation-PCA, and random bases on at least
  one downstream circuit metric without losing more than 10% on global replacement
  fidelity at matched rank/bits.
- Encoded complexity is computed by the canonical tensor-program prototype, not by
  counting the source implementation that happened to express it.

Failures are retained. In particular, an exact Boolean Mobius reconstruction on the
same eight arms is a mathematical identity and is not evidence of generalization.

## Files

- `mobius.py`: CPU-only Boolean Mobius and low-order effect fitting utilities.
- `vector_quadratic_complexity.py`: CPU-only product-rank certificates for joint
  vector-valued quadratic maps, including output/input flattening and scalar
  contraction-inertia lower bounds.
- `question_channel_ledger.py`: GPU experiment for writer/reader/final question
  channel interventions with live and frozen RMS gauges.
- `hankel_rank_audit.py`: prefix/continuation predictive-state rank audit.
- `output_slice_audit.py`: behavior-agnostic output directions versus class-seeded
  and random slice discovery.
- `FORMALISM.md`: conditional interventional description length, certified scalar
  quadratic complexity, normalization boundaries, and falsification gates.
- `REVERSE_ENGINEERING_ROADMAP.md`: operational end state, whole-model coverage
  ledger, ranked priorities, and explicit pruning rules.
- `whole_model_balance_sheet.py`: CPU-only, source-hashed aggregator over the frozen
  Theseus anchor/registry and existing causal/composition results. It derives
  summaries but does not create a competing registry or evaluator.
- `test_whole_model_balance_sheet.py`: denominator-closure and source-hierarchy
  regression tests for the balance sheet.
- `hourly_strategic_review.sh`: queues an hourly high-level reprioritization prompt
  into the active Codex session (the cron itself is session-local).
