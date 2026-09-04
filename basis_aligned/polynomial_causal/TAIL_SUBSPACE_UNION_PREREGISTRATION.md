# How many directions is the whole tail over-large along? A union-dimension analysis. Preregistration

Registered 2026-09-04T13:45Z (substituted by `date -u` at write time, so the stamp cannot drift from the value read). Before the analysis.
Immutable; the script's frozen-hash check refuses to execute if this file changes.

## Why

§2923 adopted a **rank-32 projection per tail link map** — 32 separate objects, one per (layer, class). §2924 showed their subspaces are
**partly** shared: 6.1× the random null across layers, 8.4× across classes, but only **0.17–0.23** of full overlap, so a single shared
rank-32 projection cannot do the work of 32.

**The question §2924 registered as next is the one the certificate line wants: how many directions is the whole tail over-large along,
jointly?** Pairwise overlap cannot answer it — 32 subspaces can be pairwise 20% aligned and still span anywhere from 32 to 1024
dimensions. The union is what determines whether the adopted correction is **one modest object or thirty-two**.

Method, entirely on the §2915 on-disk stack. Stack the 32 top-32 left singular bases side by side into `M = [U₁ | … | U₃₂]`, a
1152×1024 matrix. Each `Uᵢ` is orthonormal, so `‖M‖²_F = 32·32 = 1024` exactly, and the energy profile `f(k) = Σ_{i≤k} σᵢ² / 1024` says
what fraction of all 32 correcting subspaces lies in the top k directions of their union. `k90` is the smallest k with `f(k) ≥ 0.90`.

**The control is the same computation on 32 independent random 32-dimensional subspaces**, drawn and processed by the same functions.
This matters more than usual: `f(k)` rises with k for *any* collection, so a curve that merely "looks concentrated" proves nothing
without the null beside it, and per [[control-the-new-code-path]] the null must travel the measurement's own path.

## Predictions, each with its worked-example line

- **pred_a — the total energy is exactly 1024.** `|‖M‖²_F − 1024| ≤ 0.5`. *Worked example:* **1024.0** if every `Uᵢ` really is
  orthonormal with 32 columns. This is the arithmetic control: it fails if the SVD bases were mis-sliced or not orthonormal, which would
  make every fraction below meaningless.
- **pred_b — the reload is faithful.** 32 maps, layers 10–17, classes {2, 7, 8, 9}, each `LW` 1152×1152, `M` of shape 1152×1024.
  *Worked example:* §2924 read exactly this. The §2879 check that the analysis manipulates what it thinks it does.
- **pred_c — the union is materially more concentrated than chance.** `f(128) ≥ 1.5 × f_random(128)`. *Worked example:* one broadly
  shared structure ⇒ f(128) ≈ 0.5–0.7 against a random ≈ 0.25–0.35, holds; 32 essentially independent subspaces ⇒ the two curves sit on
  top of each other and it fails. **1.5× is deliberately modest** — I want to detect real structure, not to clear a bar chosen to be
  impressive.
- **pred_d — the union is substantially smaller than 1024.** `k90 ≤ 0.7 × k90_random`. *Worked example:* if the correcting directions
  live in a few hundred dimensions, `k90` ≈ 300–500 against a random ≈ 700–900; if they are spread, the two match and it fails. **This
  is the number the certificate line wants**, and it is reported whether the predicate holds or not.
- **pred_e — the union is stable across depth.** The top-128 union directions of layers 10–13 and of layers 14–17, compared by mean
  squared principal cosine, exceed **1.5 × 128/1152 = 0.167**. *Worked example:* a genuine shared structure ⇒ 0.4–0.8; an artefact of
  pooling ⇒ ≈ 0.11 and it fails. **A half-split is the cheapest independent check available**, and §2924's caveat — that some link maps
  are similar as raw matrices (max cosine 0.787) — makes it necessary: raw similarity within a half cannot manufacture agreement
  *across* halves.

## Nulls

- `a_null_the_bases_are_not_orthonormal`; `b_null_the_reload_is_not_faithful` — either voids the analysis.
- `c_null_the_union_is_no_more_concentrated_than_chance`, `d_null_the_union_fills_the_available_space` — together these would say the
  adopted correction is irreducibly 32 objects. **That is a perfectly good and quite informative answer**, and it would close the
  "replace 32 projections with one" line rather than leaving it open.
- `e_null_the_shared_structure_does_not_survive_a_half_split`.

**What I will do with each outcome, stated in advance.** c, d and e hold ⇒ report the union dimension as the campaign's first *size*
estimate for the tail correction as a single object, and register a GPU rung testing whether a projection onto the union's top-`k`
directions, applied uniformly to all eight layers, recovers §2923's composed gain — which would replace 32 objects with one and is
exactly the kind of claim "0 of 68" needs. Any of them failing ⇒ record that the correction does not compress, and stop this line rather
than searching for a formulation that does. **Nothing is adopted from an analysis**; §2923 stands as the frontier of record either way.

## Price

**0 GPU-seconds and 0 GPU forwards.** CPU only: one `torch.load` of the §2915 cache (`map_location="cpu", mmap=True`), 32 SVDs of
1152×1152, one SVD of 1152×1024, and the random control at the same sizes. It does not touch the runner, the queue, or the model.
Receipt: `tail_subspace_union_results.json`, in the canonical `Price:` / `Results:` form (§2853, §2858), under a filename no other
section cites (§2876).
