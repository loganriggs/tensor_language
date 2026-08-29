# Hourly strategic review — 2026-08-29 14:20 UTC

## UPDATE PART — what changed in this review

1. **The E4 four-head result is non-additive in both correct-token loss and the
   complete output distribution.** The earlier receipt-bound CE analysis found that
   replacing `L5H5+L7H3+L8H3+L8H4` jointly has copy effect `0.44870` nat, versus
   `0.10705` from summing their four singleton effects. The excess is `0.34165` nat
   with simultaneous document-bootstrap lower bound `0.20810`.

2. **A new CPU analysis partly red-teams the “the joint ablation is merely larger”
   explanation.** On copy positions, native-to-replacement KL is `0.38835` for the
   joint intervention versus `0.05986` summed over singletons: a `6.487x` ratio and
   `0.32848` excess. On off-target positions the ratio is only `1.683x` and the excess
   is `0.01481`. The copy-minus-matched KL excess is `0.29373`, simultaneous lower
   bound `0.16249`. Thus the joint replacement does not move the output distribution
   by one uniform multiplier across contexts. This remains post-hoc and does not
   measure residual-vector displacement; direct alignment/norm controls remain
   necessary.

3. **The fallback-map engineering question is better localized.** The parallel S1936
   discovery run reports exactly zero changed top-1 predictions at positions whose
   current token has a stored table row. All `6,936` changed predictions across its
   three roles occur at uncovered inputs. Rank 64 to 512 improves all four rare target
   buckets within the uncovered-input cell but reduces the common `125+` bucket by
   `1.20 / 2.48 / 2.16` percentage points. This confirms that the map is a fallback,
   not a hidden change to covered rows, and that its price is a genuine rare/common
   deployment trade. It is discovery-only and moves no strict ledger.

4. **The deadline condition has expired.** The special eight-hour window ended at
   12:00 UTC. Its final balance sheet remains six measured negative cells, three
   scientifically pruned cells, and the narrowed E4 copy screen completed as a
   scientific negative. Literal E4.1 outside that narrowed screen and E4.2/E4.3 remain
   incomplete. No plan, freezer, or unrun runner is counted as evidence.

## Honest fraction explained

The strict ledgers remain unchanged:

| Currency | Settled value | Meaning |
|---|---:|---|
| Structural interfaces | 36/36 | Every site can be executed or intercepted; not semantic understanding |
| Certified storage removal | 29,196,288 / 545,904,054 = **5.348245316%** | Consequence-certified whole-program storage removal |
| Named causal CE | 0.57968 / 5.30682 = **10.923302467%** | Named causal effects in the strict ledger |
| Unexplained CE | **4.72714 nat = 89.076697533%** | Largest quantitative explanation gap |
| Terminal actions | **0/68** | No extraction/removal/OOD action has passed the complete contract |

The model is structurally accessible but still mostly unexplained causally and
semantically.

## Largest remaining gaps

1. **The copy bundle is important but not selectively removable.** Its mean
   replacement has copy effect `0.44870` but off-target damage `0.02441`, above the
   frozen `0.01` budget. The missing interface is the copy-dependent component inside
   the four heads, not another whole-head ablation.
2. **Interaction order is unresolved.** Ten pair/triple arms are missing, so the
   robust non-additivity cannot yet be assigned to a pair, triple, or four-way term.
3. **No fresh native-Down behavioral port.** Family F found a 512-gate MLP3 support
   whose native Down columns beat a local decoder downstream, but its fresh row role,
   finite edits, and OOD behavior remain unmeasured.
4. **No component composition law.** Independently strong MLP0/MLP1/MLP2/attention
   replacements have not been installed in a common factorial telescope. Compensation
   and interaction therefore remain unquantified.
5. **The compiler still deletes most contextual computation.** Its 14%-scale top-1
   accuracy remains far below the live model's 39--42%, with especially poor rare and
   uncovered behavior. A higher-rank fallback helps some of those cells but harms the
   common bucket and does not name the context program.
6. **No accepted semantic coordinate system.** Low rank, shared dictionaries, and
   lexical structure are useful proposals, but no basis has yet combined whole-model
   faithfulness, stable semantics, composition, and an edit API.

## Candidate actions considered and pruned

- **More fallback-rank or table-allocation sweeps:** pruned as the main lane. They now
  expose a real price/accuracy frontier, but cannot explain the missing context or
  create a selective circuit.
- **Another unconditional mean ablation of the four heads:** pruned. E4 already shows
  the exact bank is too blunt.
- **Ranking the four heads by singleton CE:** pruned. Their joint effect is `4.19x` the
  singleton sum, so singleton importance misses most of the computation.
- **Raw HOSVD/PCA/SAE on the four head writes:** demoted to proposal generation. It
  does not distinguish copy from non-copy directions unless scored downstream and
  causally.
- **More local least-squares Down fitting:** pruned as a primary objective by Family F;
  it improves local NRMSE while worsening downstream KL.
- **A universal rank-64 state or unconstrained information bottleneck:** pruned by the
  earlier finite-transport negative and the risk of deleting rare intervention states.
- **Full tensor-network canonical form now:** deferred. Gauge canonicalization is
  useful after an explicit polynomial subnetwork/interface is isolated; it does not
  currently cross RMSNorm and residual nonlinearities with a causal guarantee.

## Top five actions, ranked

### 1. Complete four-head Boolean causal cube with displacement controls

Measure all 16 subsets on a new prospectively frozen natural role. Compute exact
document-paired Möbius coefficients separately for copy-positive, matched-negative,
off-target CE, and KL. Record residual-vector displacement norm and alignment for every
arm, plus a small scaled-full-set curve, so interaction is not confused with ablation
magnitude. This has the highest information gain because only ten subset arms are
missing and the current effect is large, specific, and robust. It can produce a circuit
boundary or decisively show that head subsets are the wrong coordinates.

### 2. Execute the rank-512 native-Down behavioral port

The candidate is roughly one ninth of native MLP3 and already has the best fit-set
downstream KL among matched 512-gate forms. Fresh CE/KL, finite-error secants, edit
transport, matched random/decoder-shift controls, and OOD replication directly test an
executable polynomial program. It ranks second only because its row freezer and
measurement lifecycle still require independent approval.

### 3. Conditional causal abstraction for copy

If the subset cube localizes a sparse interaction, fit a small executable macrostate
from prior-query existence, distance, successor relation, token identities/frequencies,
and position. Compare unconditional replacement, copy-gated replacement, and a
rate-matched random gate. Require off-target CE below `0.01` while retaining at least
half the discovery copy effect before replication. This is the shortest path from a
causal bundle to selective removal.

### 4. Exact MLP0/MLP1/MLP2/attention composition telescope

Use a common document role and factorial arms to measure singleton, pair, and joint
replacement effects. Möbius terms quantify compensation rather than inferring it from
separate runs. This is essential for whole-model composability but costs more GPU and
has a broader hypothesis space than the four-head cube.

### 5. Downstream-Fisher shared active basis

If head-subset interactions are diffuse, estimate the directions at the chosen
residual interface to which final logits are most sensitive. Compare ranks 8/16/32
against activation PCA/HOSVD and Haar controls on held-out KL, finite edits, collateral,
and stability under document doubling. This exploits tensor linearity while using a
behavioral metric, but the Jacobian/Fisher approximation is local and therefore ranks
behind finite causal interventions.

## Action executed during this review

The highest safe CPU action was completed, not merely planned:

- added `analyze_e4_distribution_nonadditivity.py`;
- added two fail-closed tests;
- tests pass `2/2`;
- ran 10,000 deterministic document-bootstrap draws;
- wrote `e4_four_head_distribution_nonadditivity_descriptive.json` bound to the
  receipt-backed ledger SHA `ca180ec...`;
- explicitly labeled the result post-hoc, non-confirmatory, and not an interface-norm
  measurement.

The result makes a uniform output-distribution-magnitude explanation less plausible:
joint/singleton-sum KL is `6.487x` on copy positions but `1.683x` off target. It does
not eliminate the stronger residual-displacement/alignment explanation. Those controls
are now part of the top-ranked full-cube design.

An independent audit agent was also restarted on the native-Down fresh-row freezer.
That audit opens no role and creates no scientific authority; its output will decide
whether the blocker is code, data/cache, or lifecycle approval.

## Exact blockers

- **No external data or checkpoint blocker is known.** The model and prior FineWeb
  roles are locally usable.
- **The native-Down port is lifecycle-blocked:** its prospective freezer exists, but no
  fresh-row receipt exists and its addendum requires independent approval before row
  publication.
- **The 16-subset cube is design/authority-blocked:** a new role and frozen scorer are
  required because E4's selection role is exposed and its negative receipt forbids
  opening E4 final/OOD.
- **The GPU is no longer occupied by S1936**, but absence of GPU contention does not
  waive either scientific lifecycle gate.

Nothing is blocked by a need for user authority. The next safe work is to finish the
row-freezer audit and freeze the full-cube contract with displacement controls; the
next model execution must wait for its own prospective authority.

## 14:40 UTC addendum — execution and strategy moved again

### Opposing simplicity objectives were confirmed in one build

The S1938 discovery run scored the nearest-neighbor, rank-64 map, and rank-512 map
fallbacks on identical positions with both top-1 and CE. The earlier cross-run
inversion is real:

| Fallback | Approximate fallback price | Pooled top-1 relative to map64 | Pooled CE relative to map64 |
|---|---:|---:|---:|
| nearest covered output row | 0.09M | **+0.61 / +0.41 / +0.60 pp** | **+0.00730 / +0.01614 / +0.00568 nat worse** |
| rank-64 map | 5.308M | baseline | baseline |
| rank-512 map | 42.467M | +0.22 / +0.12 / +0.08 pp | -0.04465 / -0.04832 / -0.04070 nat better |

All arms are exactly identical at covered inputs. The inversion occurs only on the
24--25% of positions whose current input token lacks a stored row. The neighbor wins
top-1 in four target-frequency buckets but is structurally poor when the correct target
was unseen in fit: its unseen-bucket CE is about `1.06--1.16` nat worse than the
rank-512 map.

This is direct evidence about the usefulness of simplicity definitions:

- literal price alone favors the neighbor;
- top-1 extraction also favors the neighbor;
- probabilistic faithfulness and the model's training objective favor the map;
- no scalar simplicity score can choose between them without first naming the intended
  use.

The correct object is a Pareto frontier: price, CE, top-1, and behavior-specific
consequences remain separate coordinates. A future executable hybrid cannot gate on
the unknown target-frequency bucket; it must predict which fallback to trust from
available input/context state, then beat both arms on untouched data at its charged
gate price.

### The four-head interaction experiment is now an executable mathematical contract

`TERMINAL_COPY_INTERACTION_CUBE_V1_PREREGISTRATION.md` freezes all 16 subsets, the ten
missing pair/triple arms, a scaled-full-set curve, per-layer displacement norm and
alignment statistics, the complete signed Möbius reduction, and strict claim
boundaries. `terminal_copy_interaction_cube_v1.py` passes four known-answer tests.
This is infrastructure, not a new model outcome; launch remains NO-GO pending rows,
dispatcher, scorer, lifecycle, audit, and authority.

### The native-Down row blocker was audited, repaired, and replayed

The independent audit correctly returned NO-GO on the old freezer despite its 12
passing tests. It found four P0 defects: one intentionally unmaterialized historical
row reference aborted the census, the parent preregistration was missing from source
closure, no independent-audit artifact was enforced, and the final protected-state
checks were not inside the create-only writer immediately before `os.link`.

The repair at pushed commit `a1896563` now:

- allows only the two exact registry JSON references to the same hash-bound spent
  authority/failure pair;
- schema-validates and hash-binds one 50,257-entry fit-frequency vector that the
  generic filename heuristic had incorrectly classified as row data;
- includes the parent preregistration in source closure;
- requires an independent outcome-blind GO binding every source byte;
- reruns claim, audit, source, registry, parquet, installed-cache, and namespace checks
  immediately before the receipt hard link;
- passes 19 focused tests and `py_compile`.

The complete read-only census now passes over 125 registry files, 35 tensor artifacts,
3,084 prior source documents, 8,597 full rows, and 8,396 prefixes. No cache, receipt,
lock, fresh role, checkpoint, or scientific outcome was opened. A second independent
audit of exact commit `a1896563` is running. Until it produces and validates a canonical
GO artifact, the freezer remains NO-GO.

### Updated top five

1. **Finish the native-Down row re-audit and, only after GO, freeze the fresh role.**
   This is now closest to a full executable experiment and tests the strongest
   one-ninth-size polynomial program.
2. **Finish the 16-subset copy-cube physical scorer/lifecycle.** The mathematical
   contract is frozen; the remaining work is exact tensor instrumentation and a new
   role. It remains the best behavior-specific causal route.
3. **Build a conditional copy macrostate only if the cube localizes a sparse term.**
   This prevents another broad unconditional replacement and gives a falsifiable
   selective-removal target.
4. **Run the MLP0/MLP1/MLP2/attention composition telescope.** This measures
   compensation and is required before independent simplifications can form a whole
   program.
5. **Estimate the downstream-Fisher basis if subset effects are diffuse.** It is the
   principled behavior-weighted fallback when head identity is not the right basis.

The neighbor/map hybrid is useful compiler engineering but remains below these five:
it improves a frontier rather than identifying the missing causal program.
