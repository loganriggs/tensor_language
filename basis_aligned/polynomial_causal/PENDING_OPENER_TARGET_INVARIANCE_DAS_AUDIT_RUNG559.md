# Rung 559: structural audit of R556 pending-opener target-plus-invariance DAS

**Frozen:** 2026-09-03 17:23 UTC, after the R556 headline result was visible and before this audit ran

This is a post-result integrity audit, not an independent experimental replication. It makes zero model calls and
does not change any R556 threshold.

The audit independently recomputes from the saved row-level statistics:

- every target cell's mean, median, positive fraction, deterministic bootstrap lower bound, and pass decision;
- every control cell's mean absolute closer-margin change and mean full-vocabulary logit RMS;
- each control pass decision using the saved complete-head normalization ratios;
- each seed pass, each dimension's two-of-three-seed stability decision, every random-subspace mean and decision, the
  selected dimension, and the terminal null;
- model-call budgets, opened splits, checkpoint identity, result hash, and projector-bundle hash.

R556 did not save the row-level complete-head denominators used for the two control ratios. The audit can verify the
saved ratios are finite and reapply their thresholds, but cannot independently reconstruct those two ratios from raw
rows. That limitation is recorded in the output and prevents describing R559 as a full numerical replication.
