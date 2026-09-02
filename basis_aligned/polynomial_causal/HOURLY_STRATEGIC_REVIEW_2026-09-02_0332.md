# Hourly strategic review — 2026-09-02 03:32 UTC

## Goal and seven circuit requirements

The active goal is a smaller **predictive, composable, and manipulable tensor program** for bilin18, justified by
held-out computation and causal downstream effects. The live direction must: (1) group pieces that compute the same
downstream variable across native modules; (2) split a head or MLP when its pieces serve different tasks; (3) expose
shared parents, task-specific children, and reuse; (4) predict computation and behavior on held-out/OOD inputs;
(5) extract a sufficient task computation; (6) remove, swap, or edit it with bounded unrelated damage; and (7)
predict how shared pieces compose with task-specific branches.

**Anti-rank gate:** lower rank, bytes, parameter count, reconstruction error, or CE alone satisfies none of these
requirements. These are matched controls or implementation prices only after a causal component is identified.

## What changed

- Rung 457 found CE-level early/layer-8 equality redundancy but not a resolved layer-8 group.
- Rung 458 found no task-conditioned later reader for whole equality terms, ruling out whole terms as the true basis.
- Rung 459 split exact terms into double-QK score and value-after-output payload. The L5H5 score substituted into
  L8H4's payload on held-out natural text at MLP9, with 112.1% causal recovery and 8.47x interchange separation.
- The 03:23 mathematical review showed that, at fixed rotary displacement and under the audited full-rank
  conditions, the two bilinear QK score branches are intrinsic up to scaling and swap.
- Rung 460 transported 91.92% causal recovery, 15.51x interchange separation, and the score geometry to independent
  code. Its preregistered task-specific response-cosine margin failed (0.0317 versus 0.05), so the natural-plus-code
  claim is withheld.

## Red-team account

After opening code, response direction is broadly shared (positive/matched-negative/off-target cosine
0.9096/0.8779/0.8551), while response magnitude and causal magnitude are much more task-conditioned. That could
explain the failed cosine margin, but it is post-hoc and cannot repair rung 460. The code role is now opened, the
object was selected on natural text, context masks overlap, and scale or native-stake differences can imitate
specialization. Splitting QK immediately would add mechanistic freedom before explaining this near-miss.

## Ranked alternatives and kill conditions

1. **Fixed code-context diagnostic:** report near/far and one/multiple-predecessor cells for the same frozen hybrid.
   Kill the natural context law if its ordering does not repeat in both document halves or subgroup effects vanish.
2. **Genuinely independent amplitude-sensitive confirmation:** only if the diagnostic yields a stable law, freeze a
   new corpus/role before outcomes. Kill if amplitude and causal order do not reproduce.
3. **Four QK-branch hybrids:** after adequate OOD evidence, exhaust scale/swap ambiguity. Kill a shared-branch claim
   if no single branch transfers or the conclusion changes under branch relabeling.
4. **Downstream-conditioned grouping across more equality terms:** kill if held-out within-group interchange is no
   safer than between-group controls.
5. **Whole heads, generic SAEs, or rank sweeps:** pruned here; rungs 457--459 localize the useful object below whole
   term grain, and rank/reconstruction does not define the task circuit.

## Decision and immediate action

Run rung 461, an explanatory diagnostic on already-open code data. Freeze the pair, payload, MLP9 reader,
natural-fit scale, masks, halves, metrics, and thresholds before new context forwards. Test the natural rung-457
prediction that far exceeds near and one predecessor exceeds multiple predecessors for native stake and hybrid
effect. Also test whether direction stays shared while raw response size and causal strength vary. No search, new
corpus, QK split, SEALED outcomes, or claim that a new metric rescues rung 460.

The durable goal remains active after rung 461: score it, publish the dated explanation, and immediately start the
licensed independent confirmation or a falsifier. Completing one rung is never a stop condition.
