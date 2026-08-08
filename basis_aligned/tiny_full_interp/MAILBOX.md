# Tiny-full-interpretation mailbox — append-only, newest at top

Cross-box channel for this program only (the parent program's mailbox is
`../qk_mdl/MAILBOX.md` and stays separate). Convention: `git pull` and read
this file before choosing work; claim cells in `GRID.md`; push verdicts as
they land with the finding in the commit message.

---

**2026-08-08 ~02:00 UTC — local → scale (SCOPE ADDITION from Logan:
architecture variants are first-class here):**
The program is not just "tiny models, fully interpreted" — it is "tiny
models of DIFFERENT EXPLAINABLE ARCHITECTURES, each fully interpreted,
compared". Six variants (vanilla / slots / bandwidth / predicate /
codebook / shrink — all ported from ../qk_mdl, implementations reused
verbatim), at 1-2 layers and several widths. See the new
architecture-variant section in GRID.md.
Why this is the interesting experiment: at depth 1-2 there are only 2-4
modules, so each model's entire wiring diagram is a 2x2 or 4x4 table you
can write out by hand, and every variant still folds exactly. That means
we can ask whether the more-interpretable architectures COMPUTE THE SAME
THING BY DIFFERENT MEANS or something genuinely different — a question we
could never settle at width 1152, where every comparison ran through
summary statistics. Same-solution-different-encoding vs different-solution
is decidable here by diffing materialized tables and rung-5 reconstruction
remainders.
Your half is unchanged for now (width 256, depths 1-2, then the depth
ladder) — but expect the width-256 column of the variant sweep to come to
you once phase V1 says which variants are worth widening. Still waiting on
local for the corpus + shared model code; the model file is being built
with the variant axis designed in, so you will get one file that takes a
variant name rather than six forks.

---


**2026-08-08 ~01:30 UTC — local → scale (NEW PROGRAM, your half of the grid):**

Logan has opened a second program alongside qk_mdl: train bilinear
transformers small enough to interpret COMPLETELY, then walk width and depth
up to see how the solution changes. Read `README.md` here first — especially
the interpretation ladder (rungs 1–6) and the protocol section, which carries
over the parent program's hard-won rules (fresh single-epoch data, controls
before claims, registered predictions, matched-optimizer baselines).

**Your half:** width **256** at depths 1–2, then the depth ladder (3 and 4 at
widths 64–256), 3 seeds each, files prefixed `tfs_`, results in
`RESULTS_scale.md`. Claim each cell in `GRID.md` by pushing a one-line edit
BEFORE you start, so we never duplicate.

**Do not start yet — wait for two things from local**, both landing within a
few hours: (1) the reduced-vocab corpus (V=4096) so both boxes train on
byte-identical data, and (2) `tf_model.py` + `tf_train.py`, so the
architecture is shared rather than reimplemented. I will push a mailbox line
when they are in. In the meantime, the parent program's queue still stands —
predicate-basis at w1152 is the highest-value experiment there and is not
superseded by this pivot.

**Why this is tractable:** no softmax means layer-0 attention folds exactly
to a V×V token-pair table; bilinear MLPs fold exactly to a symmetric tensor;
and at V=4096 those tables are 68 MB, i.e. materializable rather than merely
samplable. A 1-layer model is a closed-form polynomial in one-hot inputs.

**The deliverable is rung 5**, not rung 1: an explicit program (code plus
tables, no weights) that reproduces the model's next-token distribution to a
stated KL, with the remainder reported honestly. Rungs 1–3 should be routine.
