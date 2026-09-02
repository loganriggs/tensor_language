# Rung 511 preflight addendum: score-absent calibration baseline

**Frozen:** 2026-09-02 23:33 UTC, after the first full execution failed prediction A and before any repaired model
outcome.

The first full execution completed the2,108-forward discovery collector, but prediction A was false because score
calibration failed. The result is instrument-invalid and its candidate count is not scientific evidence.

The exact cause is one baseline assignment in the new collector. The registered calibration compares each intact
score action with the score-absent action. The collector correctly computed and captured `absent_logits`, but stored
the separate direct-native logits in `base_task`. This made the native score effect exactly zero: its recovery was0
instead of1 in every window, while the alternative actions were divided by a zero native effect. The branch identity,
four-corner replay, hook count, subset patches, and all other exactness checks passed, but the false calibration is
sufficient to invalidate the entire receipt. Confirmation and physical substitution remained unopened.

The invalid artifacts are preserved as:

- result SHA256 `2d9452d1fdc67db4d1d024409dbf6086d56e216c8dc707e79dfdd41d81273bb6`;
- bundle SHA256 `a6ac5f1348c1fb92c17d49095a7598efcda9744da903416002c2dcb431982ba3`.

The implementation-only repair changes `base_task` from the direct-native loss to the already-computed score-absent
loss, matching rungs509--510. It also strengthens the no-outcome GPU smoke: the native all-copy score effect must be
nonzero, its recovery relative to itself must equal1 within`1e-12`, and its per-document cosine with itself must be at
least`1-1e-12` in both halves and pooled. The smoke retains only these booleans, not task or circuit outcomes.

No branch, subset, relation, document, circuit split, threshold, fitted scale, composition rule, route, or execution
price changes. The same frozen rung511 experiment is rerun only after CPU tests, static gate, preflight, and the
strengthened managed smoke pass.
