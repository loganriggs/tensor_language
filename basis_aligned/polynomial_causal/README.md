# Polynomial causal track

This directory is a separate research track. It does not replace the
`bilinear_quotient` ledger, circuit registry, or Theseus verifier.

## Question

Can the polynomial / multilinear structure of bilin18 produce a causal
description that predicts interventions better than additive ablation accounting,
and can the resulting variables find circuit joints better than generic low-rank
compression?

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
- `question_channel_ledger.py`: GPU experiment for writer/reader/final question
  channel interventions with live and frozen RMS gauges.
- `hankel_rank_audit.py`: prefix/continuation predictive-state rank audit.
- `output_slice_audit.py`: behavior-agnostic output directions versus class-seeded
  and random slice discovery.

