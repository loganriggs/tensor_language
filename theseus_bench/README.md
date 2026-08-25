# theseus-bench (practice build)

The ship of Theseus, except every plank is replaced with a **glass** plank — the ship
must sail exactly as before, but you can see through it. Headline metric: how much of
the ship is glass?

This is the in-house practice build of the TheseusBench spec
(../basis_aligned/bilinear_quotient/THESEUSBENCH_SPEC.md, v0.2), built against the
bilin18 546M bilinear substrate using the bilinear_quotient program's assets:

- **Anchors (§9)**: the 198-component optimal-ablation sweep (all MLPs, heads, attn
  layers; loss curves + data budget recorded) -> bench/anchors/ via
  bench/make_anchors.py once the sweep completes.
- **Baseline zoo**: mean constant + optimal constant per component (from the sweep);
  identity; k-cluster (from the program's class-table experiments).
- **Worked example (bias-head)**: head 5.7, the attention sink — ONE fixed vector
  scores ~0.985 (program §1089/§1091). Our literal bias-head.
- **Mode A pre-seeds**: mlp1 token table (~.93), mlp4 = W[attn4; mlp3] (fidelity .69
  opt-anchored, §1428/§1433), mlp16/17 linear reads (.81/.84).
- **Mode B pre-seeds**: the four certified family kits + removal tests (closer,
  comparative, question, capitalized) and the unified bill as the composite prototype.

Status: M0 in progress (contract + anchors bridge). The GPU program continues in
../basis_aligned/bilinear_quotient; this repo grows as its results freeze into anchors.
