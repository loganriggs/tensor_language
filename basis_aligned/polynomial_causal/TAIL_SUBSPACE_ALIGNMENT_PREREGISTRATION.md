# Are the 32 correcting directions shared? A CPU-side subspace-alignment analysis. Preregistration

Registered 2026-09-04T13:35Z (substituted by `date -u` at write time, so the stamp cannot drift from the value read). Before the analysis.
Immutable; the script's frozen-hash check refuses to execute if this file changes.

## Why

§2921 established that the tail correction is a **rank-32 projection** per link map: shrinking the top 32 of 1152 singular directions
is worth 0.0541 nats more than shrinking all of them, interior in both axes, and it transports (−0.2872 held out vs −0.2828 fresh).
§2915 put the entire fitted stack on disk, bit-exact and CPU-loadable.

**So the obvious structural question costs zero GPU-seconds: are those 32 directions the *same* 32 across the eight tail layers, and
across the four link classes within a layer?** If they are, the correction is not 32 separate subspaces — it is **one** subspace that
every tail dictionary is over-large along, which is a structural claim of the kind the "0 of 68" certificate line has been waiting for.
If they are not, the correction is genuinely 32 local objects and the low-rank finding, while real, does not simplify the program.

There are `8 layers × 4 classes = 32` link maps `LW[li][c]`, each 1152×1152. For each, take `U₃₂` — the top 32 left singular vectors,
the output directions the map writes along and the ones §2921's arms rescaled. Subspace agreement is measured by the **mean squared
principal cosine**, `‖U₃₂ᵀ V₃₂‖²_F / 32`, which is 1 for identical subspaces and, for two independent uniformly-random 32-dimensional
subspaces of ℝ¹¹⁵², has expectation exactly **32/1152 = 0.0278**. That expectation is the null, and it is computed in closed form rather
than assumed — the script also measures it empirically over random orthonormal draws so the two can be compared.

## Predictions, each with its worked-example line

- **pred_a — the random baseline matches theory.** `|mean(random pairs) − 32/1152| ≤ .005`. *Worked example:* ≈ 0.0278. **This is the
  control**: it proves the statistic is computed correctly and calibrates what "no alignment" looks like. Without it, no other number
  in this analysis is interpretable, and per [[control-the-new-code-path]] the control must travel the same code path as the
  measurement — it does, the same function is called on random orthonormal matrices.
- **pred_b — the reload is faithful.** The stack loaded from `.fitcache` contains all 8 tail entries with 4 link classes each, and every
  `LW` is 1152×1152. *Worked example:* 32 maps found. §2915 verified bit-exactness; this checks the analysis reads the object it thinks
  it does, which is the mistake §2879 punished (measuring components that were not installed).
- **pred_c — across layers, alignment beats the random null.** `mean(cross-layer, same class) ≥ 4 × 0.0278 = 0.111`. *Worked example:*
  one shared subspace ⇒ ≈ 0.5–0.9; independent subspaces ⇒ ≈ 0.028 and this fails. **4× is a deliberately modest bar** — the claim
  "these are related" needs only to clear noise decisively, and I would rather report a real 0.15 than fail a bar set at 0.5.
- **pred_d — across classes within a layer, alignment beats the random null.** Same bar, same worked example. **The two can differ**, and
  that would itself be the finding: layers sharing a subspace while classes do not would say the correction is per-class and
  layer-invariant, which is a much more specific statement than either alone.
- **pred_e — alignment is not an artefact of the maps being near-identical.** `mean full-matrix cosine similarity between the raw LW
  pairs < 0.9`. *Worked example:* if two link maps were nearly the same matrix, their subspaces would align trivially and pred_c/pred_d
  would be vacuous. Distinct maps ⇒ ≈ 0.0–0.4.

## Nulls

- `a_null_the_statistic_is_miscomputed` (random baseline off theory by > .005).
- `c_null_the_layers_do_not_share_a_subspace`; `d_null_the_classes_do_not_share_a_subspace` — both perfectly good outcomes: they would
  say the rank-32 correction is 32 local objects, and the low-rank finding does not simplify the program.
- `e_null_the_maps_are_trivially_similar` (≥ 0.9) — alignment would then be uninformative and I report it as such.

**What I will do with each outcome, stated in advance.** pred_c and/or pred_d hold with a, b, e ⇒ record the shared subspace as a
structural finding and register a GPU rung testing whether a **single** shared projection, fitted once, recovers §2921's per-map gain —
which would replace 32 objects with one. Both fail ⇒ record the negative plainly: the correction is low-rank but local, and no
simplification follows. **Nothing is adopted from an analysis**; adoption requires a preregistered frontier rung with a held-out
measurement (§2914/§2916).

## Price

**0 GPU-seconds and 0 GPU forwards.** CPU only: one `torch.load` of the §2915 cache with `map_location="cpu", mmap=True`, 32 SVDs of
1152×1152 matrices, and a few hundred Frobenius products. It does not touch the runner, the queue, or the model. Receipt:
`tail_subspace_alignment_results.json`, in the canonical `Price:` / `Results:` form (§2853, §2858), under a filename no other section
cites (§2876).
