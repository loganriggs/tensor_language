# Hourly strategic review — 2026-09-03 11:33 UTC (Codex)

## What a successful circuit description must eventually provide

1. **Computation:** say what information is read, what operation is performed, what is written, and which later
   computations use it.
2. **The right units:** combine pieces of different heads or MLPs when later computation treats them as one variable,
   and split one head or MLP when different pieces do different jobs.
3. **Held-out prediction:** predict activation and causal effect on new documents, new circuit families, and OOD task
   variants.
4. **Extraction or sufficiency:** run the claimed circuit or interface separately and recover its signed effect.
5. **Selective manipulation:** remove, replace, or edit it while preserving unrelated behavior, including redundancy
   and interactions.
6. **Composition and reuse:** predict what happens when shared and task-specific pieces are recombined.
7. **Stable identification:** survive data splits, fitting restarts, and gauge changes, or be defined by downstream
   operational equivalence.

The program-level goal remains a smaller transparent tensor program that is predictive on fresh and OOD text,
composes when multiple replacements are installed, supports selective manipulation, and is cheaper in literal stored
values, operations, edges, states, or program description. Low CE, low rank, or good reconstruction alone does not
meet that goal.

## What changed since 10:31

- Rung 527 closed its registered MLP0 context-source grouping route: no pair passed and the term response was unstable
  across document halves. This is evidence against that proposed vocabulary, not a proof that no useful MLP0 circuit
  grouping exists.
- Rung 528 tested the complete physical post-MLP12 transitions for four equality actions. All were live and task
  response cosines were at least `.995`, but none met the frozen whole-state proportionality error bar; the complete
  states are not interchangeable.
- A response-only diagnosis found a different object: the leave-one-action-out average predicted Z7/Z8 better than
  single action responses, while each action's remaining response reproduced poorly across document halves. This is
  a screen, not physical evidence.
- Rung 529 now preregisters the corresponding physical shared/private test. Its CPU algebra and 26 combined tests
  pass. A code-level call audit caught two price omissions before outcomes: physical W7/W8 states and target runs
  under the three modified continuations. The corrected cost is `7,688` unconditional and `23,396` maximum forwards;
  no scientific gates changed.

## Is R529 still the highest-information next action?

Yes. It directly tests targets 2--6 rather than using rank as a proxy. For target action `a`, it inserts

`consensus_a = beta_a^-1 * mean_{b != a}(beta_b * delta_b)`

at the real 1,152-dimensional residual boundary after MLP12. It then compares the resulting circuit effects against
the native target and against all three single donors. A pass would group parts of distinct implementations by a
shared downstream computation; the exact remainder `private_a = delta_a - consensus_a` tests whether that shared
part can be selectively removed. New documents and 30 unopened circuit families test identification rather than
in-sample similarity. Four downstream continuations test whether the claim survives interactions with A14 and M17.

The decisive discovery requirement is not “average looks close.” The consensus must have relative circuit error at
least `.05` lower than **every** single donor, clear wrong-sign and circuit-permutation controls by `.10` cosine, and
meet the original absolute prediction bars. If no target passes, the post-MLP12 consensus route closes without
changing thresholds or switching to learned low rank.

## Confound audit before GPU

- **Baseline subtraction:** every arm is member-minus-matched-control CE relative to the same score-absent run.
- **Frame mixing:** all raw state changes are aligned with R528's three frozen positive scales before averaging; no
  scale is refit on R529 data.
- **Nonlinear loss composition:** consensus, private, single donors, and wrong controls are inserted as physical
  states and sent through the real suffix. Response-vector averaging alone cannot pass.
- **Shared token difficulty:** matched circuit controls and 16 circuit-coordinate permutations remain explicit.
- **Leakage/post-selection:** all four targets run; all three donors are compared; candidate identities, one donor,
  and one wrong control freeze before new documents; 30 validation circuits are unopened.
- **Dead edits:** the smoke and full runner require nonzero consensus/private/control states, nonzero downstream
  continuation changes, positive supports, exact call counts, and exact BF16 reconstruction of every target state.
- **Precision:** the state arithmetic occurs in FP32 and rounds once at the model boundary, with exact native replay
  and reconstruction checks. R528's managed smoke and audited null are hash-frozen.
- **Interaction averaging:** no continuation is averaged away for a gate; native, A14 removed, M17 removed, and both
  removed are scored separately.

## Genuinely different next routes, ranked

1. **Physical shared/private consensus (R529, current).** Changes grouping, held-out prediction, sufficiency,
   composition, and manipulation evidence. Kill it if no consensus beats every physical singleton or if the advantage
   fails on new documents/circuits.
2. **Circuit-labelled attention factor vocabulary.** Decompose Q/K/OV pieces across heads according to shared
   downstream uses, beginning with known copy/induction circuits. This directly targets cross-head grouping and
   within-head splitting. Kill a proposed unit if downstream interchange and selective interventions do not transfer
   across documents and heads.
3. **MLP0 token-only semantic code.** Use the finite vocabulary and exact folded embedding-to-MLP0 map to find token
   groups with the same downstream causal effect, leaving token identity as an explicit uninterpreted lookup when it
   cannot be simplified. Kill a group if it does not predict held-out token effects or selectively transplant.
4. **Later-MLP interaction atlas.** For a circuit-labelled target, expand a later bilinear MLP over earlier attention
   and MLP writes, use gradients only to screen pairs, then measure selected self/cross terms causally. Kill it if the
   selected terms do not reproduce across document halves or fail finite interventions.
5. **Attention/frontier engineering.** The existing QK truncation frontier may eventually supply literal savings, but
   it is secondary until its retained coordinates are tied to reusable computations. Rank or byte savings alone do
   not identify a circuit.

The exact high matrix ranks recently measured in MLP branches are useful lower bounds against one particular linear
compression story. They do **not** establish that downstream computation cannot group nonlinear terms or that an MLP
cannot be split by task. Therefore they lower the priority of generic low-rank MLP sweeps but do not close circuit-
labelled MLP decomposition.

## Action selected

Run the 37-forward managed R529 smoke after committing the preregistration, algebra, runner, and tests. Only if the
smoke proves exact reconstruction, live edits, continuation changes, and closed outcomes will the full discovery run
become eligible. This is the shortest safe path to evidence that can change the circuit decomposition.
