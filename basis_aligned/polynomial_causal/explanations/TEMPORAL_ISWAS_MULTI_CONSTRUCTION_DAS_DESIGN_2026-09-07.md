# Multi-construction is/was DAS design — 2026-09-07

## Decision being resolved

The current rank-one head-response projector is stable across row parity and improves target
transfer over difference in means, yet transfers only `.6489` to the second v15 construction. The
next experiment must distinguish two explanations:

1. a fixed construction-invariant causal coordinate exists, but A1-only fitting cannot identify it;
2. the four heads route tense through construction-conditioned coordinates, so no fixed projector
   can be simultaneously target-effective and control-selective.

This is an identification question, not a rank or compression question. It changes held-out/OOD
prediction, within-head splitting, selective manipulation, and the eventual writer/reader interface.

## Frozen structure before any v16 causal outcome

- Native modules: complete head responses at `L8H1`, `L9H1`, `L9H4`, and `L11H3`, followed by the
  already frozen complete attention-15 branch.
- Installation: unit-dose absolute clamp
  `H_live <- H_base + (H_donor-H_base) U_h U_h^T` in ascending layer order.
- Projector family: one rank-one projector per head. Rank is fixed because rank one was stable and
  higher ranks were substantially less stable; no rank search is licensed.
- Development environments: v15 A1 (`Today/Yesterday`) and v15 A2
  (`Currently/Previously`), with group parity cross-fitting.
- Development controls: the aligned v15 P and C panels. Both are explicit constraints; neither is
  collapsed into a single average.
- OOD environment: capability-qualified v16 A1/A2. No v16 head patch, gradient, activation, or
  causal metric may be opened before the objective, bars, and implementation hash are frozen.
- Initialization controls: the prior factor-SVD rank-one start, joint construction-DIM, frozen
  A1-only learned projector, and step zero. Noise and local Jacobian coefficients are omitted because
  the complete 30-configuration red-team showed no qualitative effect.

## Objective and selection

For construction `e`, parity fold `f`, and projector tuple `U`, let `r_ef(U)` be signed target
projection and `d_ef(U)` direction fraction. Let `K_cf(U)` be full-vocabulary KL on control panel
`c in {P,C}`. Optimization uses only one parity from both v15 constructions. Opposite parity chooses
the checkpoint lexicographically:

1. minimize the worst target violation
   `max_e[(.75-r_ef)_+ + (.875-d_ef)_+]`;
2. among fully feasible checkpoints, minimize
   `max_c median(K_cf) + .25 max_c mean(K_cf)`;
3. break ties by lower maximum KL, fewer flips, earlier step, and canonical initialization order.

This is a finite group-robust construction objective. It prevents one easy construction from hiding
another and prevents the zero-effect complement solution through hard target constraints. The two
parity-selected projector tuples must also have minimum principal cosine at least `.75` at every
head; this is a licensing bar, not a selectable penalty.

## Opposing predictions

- **Fixed-invariant hypothesis:** a moved checkpoint is target-feasible for both v15 constructions
  in both parity directions, improves v15 A2 by at least `.08` over the frozen A1-only projector
  without reducing v15 A1 below `.75`, and has no P/C top-one flips.
- **Construction-conditioned hypothesis:** no moved checkpoint satisfies both v15 target constraints,
  or a feasible v15 checkpoint fails either v16 target construction below `.65` or below the frozen
  A1-only projector by more than `.05`.
- **OOD identification bar:** before opening v16 causal outcomes, require the selected joint projector
  to reach at least `.65` signed projection with direction fraction at least `.875` in each v16 target
  construction and to outperform the frozen A1-only projector in the worse v16 construction by at
  least `.08`. V16 P is reported separately; the unequal-length inherited C panel is not patched and
  cannot support a selectivity claim.

If the fixed-invariant hypothesis holds, translate each `U_h` through that head's exact `W_O` slice
and rank downstream Q/K/V and MLP left/right contractions, then validate the predicted readers by
causal reset/rescue. If it fails after two observed construction environments, the next object is an
input-conditioned projector or finite mixture with routing learned on v15 and routing transfer
tested on v16; another coefficient grid is closed.

## Capability dependency

The queued v16 gate opens only native answer-versus-foil capability. Its A1, A2, and P token pairs
are exactly equal-length row by row under the GPT-2 tokenizer; inherited C is unequal-length and is
explicitly excluded from future patch claims. A capability null ends this bank without causal access
and requires a different prospectively authored cue family.

