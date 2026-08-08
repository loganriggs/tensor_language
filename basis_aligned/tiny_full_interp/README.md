# Tiny full-interpretation program

**Goal: train bilinear transformers small enough to interpret COMPLETELY —
not "we found some circuits", but a full accounting of what every parameter
does — and then walk width and depth up to watch how the solution changes.**

Started 2026-08-08 (Logan's pivot). Sibling program to `../qk_mdl/`, which
studies interpretability-enhancing architecture at scale; this one studies
total interpretability at small scale. They share the fold machinery and the
fresh-data protocol; they do NOT share checkpoints or claims.

## Why "fully interpret" is achievable here and nowhere else

Three properties compose:

1. **No softmax.** Attention is `(q1·k1)(q2·k2)/d²` with a causal mask — a
   polynomial. Layer-0 attention therefore folds EXACTLY into a token-pair
   score table: for every ordered pair of vocabulary items, a number,
   computed from weights alone with no data assumptions.
2. **Bilinear MLPs.** `MLP(x) = D·(Lx ⊙ Rx)` is exactly a symmetric
   third-order tensor `T[o,i,j]`, and RMSNorm folds out as a scalar gauge
   (`MLP(rms(x)) = D·T(x,x)/‖x‖²`, verified to ~1e-7 in the parent program).
   No gating, no ReLU, no approximation.
3. **Small vocabulary.** With V ≈ 4096–8192 the exact V×V tables are
   68–268 MB in fp32 — materializable, printable, diffable. This is the
   enabler that makes "exhaustive" affordable; at V = 50257 the same object
   is 10 GB per head-branch and only samplable.

Consequence: a 1-layer model is a closed-form polynomial in one-hot inputs,
and a 2-layer model is that polynomial composed once. There is no part of
these models we are forced to describe statistically.

## The scientific question (not just "can we")

The parent program measured, on an 18-layer model, that **selection is
nameable but content is a spectral, non-class-nameable dictionary at every
layer**. That was measured at one size. The question here:

> At what width/depth does full interpretability break, and *what* breaks
> first — does content become spectral at width 32, or only at 256? Does the
> model discover induction at depth 2 regardless of width, or only above a
> width threshold? Do independently-seeded models at the same size converge
> to the same solution, or to different ones?

That makes this a study of **how the solution changes as capacity grows**,
with a fully-known object at every point on the curve.

## The grid

| axis | values | why |
|---|---|---|
| depth | **1, 2** (focus), then 3, 4 | 1 = closed form; 2 = composition appears; 3–4 = where the parent program's "grounding" begins |
| width | **32, 64, 128, 256** (heads scale with width, head dim 16 fixed) | log spacing; 8× range brackets the transition |
| seeds | 3 per cell | the parent program learned the hard way that single-seed structure claims do not survive (see `../qk_mdl/BRAINSTORM_STATE.md`, FOUNDATIONS CORRECTION) |
| vocab | 4096 (primary), 8192 (check) | exact-table tractability; UNK rate reported per corpus |

Primary cells = {1,2} × {32,64,128,256} × 3 seeds = 24 models. All are minutes
to ~1 hour each; the whole primary grid is a day of one GPU, not a week.

## Protocol (inherited, non-negotiable)

- **Fresh single-epoch data**, never a second pass over any sequence — the
  parent program found multi-epoch training silently subsidizes structure
  claims by memorization. Reduced-vocab corpus built from the same FineWeb
  shards, rare tokens mapped to UNK, UNK rate logged.
- **Positive controls before every claim.** Identity reductions must pass at
  exactly zero; every headline in the parent program that turned out wrong
  was caught by a known-answer control and never by inspection.
- **Registered predictions** written into the results JSON *before* training
  or analysis runs.
- **Matched-optimizer baselines.** Quote costs against a baseline trained
  with the same optimizer (the parent program understated every tax by 0.094
  nats for a week by comparing Muon arms to an AdamW baseline).

## The interpretation ladder (what "fully interpreted" means, concretely)

A model is **complete** on this ladder when every rung has a machine-checked
artifact, not a narrative:

1. **Exact fold** — every layer written as its tensor; gate ~1e-6.
2. **Materialized tables** — the V×V per-head-branch attention score tables
   and the MLP tensor, on disk, with their spectra.
3. **Behavioral inventory** — what the model actually computes, measured:
   bigram statistics, skip-grams, induction, positional priors, each with a
   causal ablation and a floor.
4. **Content accounting** — for every write direction, either a name that
   passes a substitution gate, or an explicit statement that it is spectral,
   with the measurement that shows it.
5. **Reconstruction** — an explicit program (code + tables, no weights)
   that reproduces the model's next-token distribution to a stated KL, and
   the honest remainder.
6. **Convergence** — the same artifacts across seeds, with a similarity
   measure, so "the model learns X" is a claim about the size, not the run.

Rungs 1–3 are expected to be routine; rung 5's remainder is the real
deliverable, and rung 6 is what makes the width/depth curve meaningful.

## Division of labour

Two machines, coordinated exactly as in the parent program: git + an
append-only mailbox, verdicts pushed as they land.

**Local box (RTX 5070 Ti, 16 GB)** — the small end and the analysis:
- trains widths **32, 64, 128** at depths 1–2, all 3 seeds (18 models, each
  minutes; the whole set is a few hours)
- owns the **corpus build** (reduced-vocab FineWeb) and pushes it so both
  boxes train on byte-identical data
- owns **all rung 1–6 analysis machinery** (folds, tables, behavioral
  battery, reconstruction), because the analysis is small and the code must
  be shared
- files: `tf_*` prefix

**Scale box (RTX 5090, 31 GB)** — the large end and the depth ladder:
- trains width **256** at depths 1–2, and the depth ladder **3, 4** at
  widths 64–256, all 3 seeds
- runs the same analysis code on its own checkpoints (code comes from local;
  do not fork it)
- files: `tfs_*` prefix
- its own results doc `RESULTS_scale.md`; local owns `RESULTS.md`

Neither box duplicates a cell. If a box is idle, it takes the next unclaimed
cell from `GRID.md` (claim by pushing a one-line edit before starting) rather
than inventing work.

## Files

- `README.md` — this file
- `GRID.md` — the cell table with claim/status per cell
- `MAILBOX.md` — cross-box channel (newest first, append-only)
- `tf_corpus.py` — reduced-vocab corpus build
- `tf_model.py` — the tiny bilinear transformer (shared by both boxes)
- `tf_train.py` — training with the fresh protocol
- `tf_fold.py` — rung 1–2: exact folds and materialized tables
- `tf_behavior.py` — rung 3: behavioral inventory with causal ablations
- `RESULTS.md` — local results

## Vocabulary decision (2026-08-08) — and why the constraint is NOT table size

The first build measured **20.0% UNK at V=4096** (13.0% at V=8192). That is
too distorted to build a program on: UNK becomes the most frequent symbol,
it dominates every table's rows and columns, and absolute cross-entropy
stops being comparable to anything.

The obvious fix is a bigger vocabulary, and a structural finding from the
same build says table size does not stop us. The layer-0 attention table is
**exactly rank <= head_dim (16) per branch**: the score for (query token i,
key token j) is `(q1(e_i)·k1(e_j))·(q2(e_i)·k2(e_j))`, so each branch factor
is a V x 16 matrix product, and the realized pattern is their Hadamard
product (rank <= 16^2 = 256). **The exact artifact is therefore four V x 16
factor matrices per head, not a V x V grid** — 3 MB per head even at the
full 50k vocabulary. V x V materialization is a convenience for printing and
diffing, not a requirement of exactness, and it can be done in chunks for
whatever token subset an analysis cares about.

So the binding constraint is not memory, it is **parameter balance**. At
these widths the embedding dwarfs the body:

| | width 32 | width 128 | width 256 |
|---|---|---|---|
| V=4096 | 84% embedding | 57% | 40% |
| V=8192 | 91% | 73% | 57% |
| V=50257 | 98.5% | 94% | 89% |

A model that is 94% embedding is not a model whose *computation* we are
interpreting — it is a lookup table with a small transformer attached, and
the interesting structure would be swamped.

**Decision: V=8192 is the primary vocabulary** (13% UNK, embedding share
57-73% at the widths that matter), with V=4096 retained only as an
already-run companion and V=16384 as a check at width 256 where the balance
is best. Every result reports its UNK rate; no cross-vocabulary CE
comparison is made without one. The corpus builder already emits both.

This also reframes rung 2 of the ladder: the deliverable is the **factor
matrices with their spectra**, and V x V grids are rendered on demand for
the token subsets an analysis names.
