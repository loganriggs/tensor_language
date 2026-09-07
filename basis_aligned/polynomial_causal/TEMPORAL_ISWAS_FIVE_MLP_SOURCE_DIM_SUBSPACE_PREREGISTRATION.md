# Temporal/is-was five-MLP source DIM subspace — preregistration

The full-module greedy search stops at `MLP0, MLP1, MLP2, MLP3, MLP6` with functional fidelity but
non-selective controls. This run replaces each full MLP-output donor patch by a projected donor-minus-
base output delta. Bases are fitted only on the old temporal and is-was families used by the established
projector training population; the fresh evaluation and control families remain sealed.

For each source MLP, compute one mean output-delta vector per training row, averaged over positions up
to the registered semantic position. The frozen arms are:

- `full`: full output patches at all five modules;
- `dim1`: the normalized pooled mean delta at each module;
- `dim2`: the QR span of the temporal and is-was mean deltas at each module;
- `svd4` and `svd8`: leading right singular vectors of the per-row mean-delta matrix;
- `dim2_complement`: full delta minus its projection onto the two-task DIM span.

Every projected arm runs on both target and control rows. Rank and basis are never optimized against
evaluation metrics.

## Gates

- A: hashes, split disjointness, self-patch/finite checks, full-arm replay, and price pass.
- B: `dim2` is functional: both tasks have rank-four response projection at least `.8`, RSE at most
  `.2`, and behavior projection at least `.75`.
- C: `dim2` is selective: median control KL at most `.02` and zero top-one flips.
- D: the `dim2` complement is insufficient, with response and behavior projection at most `.25` on both tasks.
- E: `dim2` has no worse task functionality than `svd4` or `svd8` under the frozen bars and has the
  lowest median control KL among functional projected arms.

This directly red-teams the constrained-DAS result: pooled/task-span DIM supplies the cross-family
inductive bias without optimizing a same-family complement loss. Maximum 32 forwards, zero gradients
or model updates.
