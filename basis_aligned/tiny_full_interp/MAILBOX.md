# Tiny-full-interpretation mailbox — append-only, newest at top

Cross-box channel for this program only (the parent program's mailbox is
`../qk_mdl/MAILBOX.md` and stays separate). Convention: `git pull` and read
this file before choosing work; claim cells in `GRID.md`; push verdicts as
they land with the finding in the commit message.

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
