# Lessons (distilled rules, each with a real example and a near non-example)

Hard-won rules of this program. Every lesson cites the section where it
was learned. A "near non-example" marks the boundary — where the rule
does NOT apply — because a rule without its boundary gets over-applied.

## 1. Mean-ablate; never zero-ablate (§353–354)
Zeroing is an off-manifold magnitude shock; it measures scale
sensitivity, not content.
- **Example:** zero-deleting ANY single head of a14 on IOI cost an
  identical ~1.64 margin (all nine within 0.005, recompute verified
  exact). Within-prompt MEAN ablation immediately differentiated:
  head 4 carries 0.93, most others ~0.
- **Near non-example:** whole-COMPONENT mean-ablation (the census's
  probe) is fine and always was — the rule is about deleting
  *sub-parts* (heads, direction bundles) whose absence shifts the
  parent's output scale.

## 2. Only subtraction bites; value-writing is void — with an offset tax (§352, §360–362)
Interventions act by removing variance along directions. The written
constant contributes nothing (optimized value == zero, +4.42 both;
natural donor values ≈ 0 effect) — except that off-manifold constants
(zero) pay ~1.6× extra vs the slice mean.
- **Example:** DAS "steering" of the line-break channel: +4.42 with
  the optimized value, +4.42 with all-zeros, +0.16 with natural donor
  values. The optimizer had learned *which directions to delete*.
- **Near non-example:** r.6.0.0 (agreement 0.32 across constants) is
  the one leaf where constants disagree beyond the offset tax —
  earmarked; the law is a strong default, not yet a theorem.

## 3. Selection effects: census members are damage-TAILS (§net_utility, layout-brake page)
A leaf's members are the positions where machinery moves loss MOST —
including where it's most wrong. Member-mean damage can be negative
while the circuit is corpus-wide useful.
- **Example:** r.3.1.0: members −1.14 mean (ablation "helps") but
  corpus-wide +0.071/token (+3,840 nats) — optimality holds.
- **Near non-example:** if a leaf's GLOBAL net were ≤0, "the model
  would be better off without it on this corpus" is a fair claim —
  that's the slack-harvest regime, and it did improve the model
  (−0.048 fresh) when sign-calibrated.

## 4. Richer feature sets overfit fixed-data greedy search (§347, §357)
At fixed data, adding features can LOWER held-out performance.
- **Example:** quadratic 1680-dim probe: AUC 0.551 vs plain ridge's
  0.621. Exact pair-fold features: converged 55 vs 58 without them.
- **Near non-example:** adding the 10 class labels lifted program
  passes 2/16 → 5/16 — a few *high-prior* features help; the failure
  mode is bulk vocabulary, not features per se.

## 5. Two-signed policies are universal; stories must state both wings (§348–349)
100% of tested leaves are sign-mixed. "Helps X" alone is an incomplete
story by construction.
- **Example:** line-break circuit = mlp0 push MINUS mlp3 brake; the
  Westminster-Abbey token improves 6.8 nats when the wrong-there push
  is removed.
- **Near non-example:** wings are tails of ONE continuous mode, not
  two modules — don't reify "the positive circuit" and "the negative
  circuit" without a bundle dissociation test (§349's (a)).

## 6. Shape and gain are different objects in attention (§364)
Unnormalized squared-product patterns = shape × gain. Linear fits on
raw patterns measure gain variance.
- **Example:** archetype dictionary R² ≥0.7 for 1/162 heads while
  literal one-hot pattern swaps work for 71/74 — both true, different
  objects.
- **Near non-example:** functional claims ("this head reads prev")
  survive; matrix-level claims ("this pattern ≈ prev matrix") don't.

## 7. Tags are tree-instance-local (§344 caveats)
The same physical circuit gets different tags in different tree
builds. Identity = member overlap, never name.
- **Example:** mb3's r.0.0.1 ("rare multi-token words") vs the
  212-row tree's r.0.0.1 (line-break) — different circuits, same tag.
- **Near non-example:** within ONE census_state.pt, tags are stable
  and safe to use as keys.

## 8. Description-language limits are measured, not assumed (§341, §346, §357, §363)
The programmable frontier (55–58/118) is invariant to input-feature
vocabulary: surface, classes, triggers, shifts, folds, pair folds,
unbounded match transports. Membership for the rest is not
token-definable.
- **Example:** five feature classes, same plateau ±3.
- **Near non-example:** the plateau is about INPUT-computable
  features; stream-level features are a different class (probe: 3.8×
  vs surface 1.5×) and are tested separately.

## 9. "Mechanism" has grades; name them honestly (§365-pending, user standard)
Firing-condition mechanisms (trigger tables, 8–36× precision) are NOT
computational mechanisms (executable code replicating behavior).
- **Example:** the four "induction-grade" circuits are trigger-grade;
  zero circuits currently meet the write-the-code-and-replicate
  standard (first attempts queued: mech_replicate, suffix_code).
- **Near non-example:** the assembly's fold tables DO replicate
  behavior as code — at component grain; the missing piece is
  circuit-grain semantics, not code per se.

## 10. Say which distribution a "fresh" number used (§386)
The model trained on FineWeb; the program's fresh legs used Pile --
mildly OOD (+0.10 base CE). Transfer that survives OOD is extra
safe, but absolute numbers mix a distribution tax into the
replacement tax.
- **Example:** combined-readable +0.224 "fresh" = Pile; the
  in-distribution number is expected slightly lower.
- **Near non-example:** internal comparisons (config A vs B on the
  same rows) are unaffected -- the tax cancels.

## Ops rules (each cost at least one incident)
- queue.txt takes ABSOLUTE paths; bare names are silently dropped.
- After writing a transform-generated script: verify file exists AND
  ast.parse it BEFORE queueing (ioi_chain incident: queued, then the
  transform crashed, runner would have skipped silently).
- Watchers: count-based on _completed.txt, never grep of logs (stale
  tracebacks false-trigger).
- git: explicit paths during concurrent work; agents never commit
  (consolidator model); directory-wide `git add` swept another
  agent's uncommitted work into a pushed commit once (§356).
- Registry/features writes only through census_lib (locks, deep-merge,
  append-only certification).
- Mid-wave infra edits cause version skew inside running agents:
  batch infra changes between waves; provenance records lib_rev.
- bqrunner REQUIRES ABSOLUTE PATHS in queue.txt: it checks `[ -f "$line" ]`
  from the runner's own cwd, so a relative path (e.g. `weight_action_compose.py`)
  is popped and SILENTLY DROPPED -- nothing runs, no error. Always queue the full
  /workspace/.../script.py path (§753->754 cost one idle cycle to this).
