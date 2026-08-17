# Plan: causal verification + red team of the layer-1 story, then the other layers

User directive (2026-08-17): causally verify the register-leader story and the
compression claim (replacement must win on fidelity AND on MDL), red-team it, causally
verify the semantic hypotheses at the text level, then give the other layers the same
depth of treatment, starting at layer 0.

## Phase A — layer 1 leader: verify, replace, red-team
- A1 semantic causal test: ablate ONLY the leader direction; damage must concentrate
  where the hypothesis says (whitespace-heavy contexts), measured per-position with
  contexts binned by layout-token fraction. Includes the reverse prediction: prose
  contexts nearly unharmed.
- A2 causal replacement: swap the leader coefficient c_0 = xhat^T M xhat for the
  compressed surrogate the story implies (a single squared projection of the head-4
  component, then ladder: +rank-2, +full attn1 restriction). Score each rung on
  (i) CE fidelity vs the intact model, (ii) parameter count -> honest MDL ladder.
- A3 red team: (i) matched-size surrogate on a random direction; (ii) surrogate fit on
  document-shuffled targets; (iii) transfer to held-out document rows (the §16
  heterogeneity is the obvious failure mode); (iv) text-level intervention: inject
  layout tokens into prose, verify the leader moves and downstream CE responds as the
  story predicts; remove layout from markup-heavy text, verify the reverse.
- A4 wrap into BILIN18_CONNECTION §19 + commit.

## Phase B — layer 0, same depth
- B1 Shapley (big-data basis from the start; 20 perms) -> concentration verdict.
- B2 writer folding (writers: emb + attn0 only) -> which pairs drive the leaders.
- B3 data structure (spectrum / kurtosis / ICC / hierarchy).
- B4 naming: excitation + emb-curvature vocab naming; unfold attn0 by head if
  attention-driven.
- B5 causal check of its leader, A1-style.
- Commit as §20.

## Phase C — remaining layers, triaged
- Layer 16 next (the compressible one, R=9: tractable), then 17 tail directions.
- Layers 2-15: §10 showed individual ablations understate them (2.87x superadditive);
  the per-layer treatment must therefore use Shapley-style attribution from the start,
  and a cheaper variant (fewer perms, coarser basis) is acceptable. Do as budget
  allows, deepest-first by delete cost: 3, 2, 4, 15, ...
- Keep the report updated at phase boundaries; correction-first writing throughout.
