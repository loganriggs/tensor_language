# Existing artifacts for task-conditioned circuit decomposition

Date: 2026-09-02 01:03 UTC

This audit asks what can already support grouping computation across attention heads and MLPs, and what new
measurement is actually missing. It deliberately does not treat lower rank as circuit progress.

## Main conclusion

Start with the equality/copy/induction computation, not another attention or MLP rank sweep and not the existing
ordered-successor compression plan.

The repository already contains an exact additive equality-edge term for each of four heads: L5H5, L7H3, L8H3, and
L8H4. The current executor can independently retain, remove, or replace each term while replaying the model. The
existing experiments only changed all four together. Therefore the cheapest missing experiment is a four-head subset
factorial plus downstream-component responses. It directly asks whether parts of different heads are redundant,
complementary, or task-specific, and which later attention/MLP computations read each part.

The general 62-item battery is useful for collateral effects but cannot define this first task decomposition by
itself. `circuits/BATTERY.json` explicitly records `ind_band` as absent from its census state, and its behavior tags
contain no named copy, induction, previous-token, or successor task. Dedicated task masks and controlled task
variations are required.

## Equality/copy/induction: what is already known

For query position `q` and source position `k`, the shared support relation is

`M[q,k] = 1[token[q] = token[k-1]] × 1[1 <= k <= q]`.

For head `h`, its equality-edge term is

`e_h(q) = O_h sum_k M[q,k] score1_h(q,k) score2_h(q,k) value_h(k)`.

Thus the four terms share the same token-equality relation but retain their own two continuous QK scores, value
features, output map, and layer. This is already a meaningful proposed grouping: one input-side relation reused by
four different output branches. It is not yet established that the four branches are interchangeable.

### Existing causal evidence

- Removing all four equality terms adds `.51225` nat on the registered induction-positive positions.
- Starting with all four heads deleted and restoring only their equality terms recovers `.97397` of the positive-cell
  loss effect on the discovery set.
- The token-permuted equality relation recovers `-.00216`, so correct token identity matters.
- Discovery off-target damage is `.006264` nat.
- On new natural documents, extraction recovery is `.90851`; on repository-disjoint code it is `1.01041`.
- The final natural and code runs fail their collateral guarantees: the code off-target point is `.13831` nat, and
  the natural simultaneous upper bound is `.19516`. This is consistent with equality copying being a broad reusable
  service rather than an induction-only circuit.

The broad collateral failure is useful for decomposition: it predicts that the equality matcher should be shared,
while later output/use branches should distinguish induction from other copying.

### L8H3/H4 is already split more finely

At a repeat position `p`, let `j` be the nearest earlier occurrence of the current token. Almost all copy-specific
effect from L8H3/H4 comes from the single edge to `j+1`, the token after that earlier occurrence:

- deleting that edge adds `.12792` nat on copy-positive positions;
- deleting both full head writes adds `.13403` nat, so this edge accounts for `95.4%` of the matched head effect;
- deleting only the shared block-0 value-bus part adds `.11692` nat;
- deleting only the fresh contextual value part adds `.00544` nat; and
- deleting the adjacent wrong edge adds `-.00057` nat.

This separates three roles: the repeat/equality relation chooses a source, the continuous QK product decides how
strongly to use it, and the shared value bus carries mostly token identity. A static old matcher and distance-only
gates recovered only `38.7%` and `30.0%`, so the contextual QK scalar remains a real unresolved computation.

## The exact missing head-grouping experiment

Let `H = {L5H5, L7H3, L8H3, L8H4}`. The current executor already computes every `e_h` separately. Run both subset
families for all `S subseteq H`:

1. **Removal subsets `G_S`:** start from the native model and subtract `e_h` for every `h in S`.
2. **Extraction subsets `F_S`:** delete all four complete head writes and restore `e_h` only for `h in S`.

This is 16 removal and 16 extraction configurations. Measure signed CE and logit effects on:

- induction/copy positives;
- matched repeat negatives;
- generic copy positions that are not induction positives;
- nonrepeat and broad off-task positions;
- controlled repeat distance, number of matches, distractor, and payload-identity variations; and
- later fresh/OOD documents.

For each task cell `c`, the complete subset table allows an exact finite interaction expansion. For example, for an
extraction loss `L_c(S)`, the interaction assigned to head set `T` is

`mu_c(T) = sum_{S subseteq T} (-1)^(|T|-|S|) L_c(S)`.

Singleton terms measure individual contribution on a common background. Pair and higher-order terms distinguish
redundancy from synergy. This is not a rank decomposition.

Interpretations:

- two heads with similar singleton response functions and negative joint redundancy are alternative producers;
- two heads with weak singleton but strong joint effect are complementary or serially dependent;
- heads with different effects across induction, generic copy, and matched negatives should remain separate output
  or use branches even though they share equality support; and
- extraction and removal disagreement reveals compensation or redundant backup.

## Finding the MLP and attention consumers

During each subset intervention, capture the exact write of every later attention and MLP component. For a source
term `e_h` and later component `j`, define the response

`Delta y_j(h,x) = y_j(model with e_h changed, x) - y_j(native, x)`.

The first pass clusters or groups these responses only as a screen. A later component becomes a claimed consumer only
if patching its induced write change into the corresponding removal run predicts or repairs the task effect on
held-out examples.

For a selected downstream bilinear MLP, then expose its exact product contributions

`write_k(x) = Down[:,k] × (Left[k] x) × (Right[k] x)`.

Native product channels are an overcomplete proposal set, not assumed circuit atoms: earlier MLP2 results show strong
cancellation among them. Group or learn mixtures of terms by how they respond to the equality-source interventions
and which task effects their patches repair. Split the MLP only when two term groups have different held-out task
fingerprints and selective interventions.

This supplies the missing connection from “four heads share an equality relation” to “these particular downstream
attention/MLP computations use that relation for these particular tasks.”

## Ordered successor: useful second case, not ready as the first

Existing evidence identifies natural-text L8H7 as the main causal owner:

- its weight-derived digit successor target ranks first for all eight tested transitions;
- mean removal adds `.1478` nat on a small general-successor set versus `.00267` elsewhere;
- a larger digits/weekdays/months/years screen gives `.2306` nat pooled damage versus `.0024` elsewhere; and
- L14H4 is mostly dormant, so the old two-head backup hypothesis is not supported.

However, the autonomous successor extraction has not run. Its row freezer found that the registered 192-document
budget could not power the required cells; a later 384-document design remains incomplete. More importantly, the
current successor plan mainly asks how far the L8H7 OV map can be rank-reduced. That does not answer the present
grouping/splitting question.

After the equality factorial establishes a task-conditioned method, successor should be redesigned to separate:

- the input relation representing order;
- the QK routing that chooses a relevant previous item;
- the output that writes the successor prediction; and
- downstream readers that use the result.

It is then a valuable contrasting case because it appears to have one dominant head rather than four shared heads.

## Next action and circuit-goal mapping

The next experiment should be the equality-term subset factorial with downstream component captures. Before GPU
execution it needs a dedicated preregistration, fresh document role, exact task masks, common-background definitions,
finite subset interaction equations, and controls.

It directly advances:

- **right boundaries:** tests whether four head parts merge or split;
- **computation:** keeps equality selection, QK strength, payload, and downstream use distinct;
- **extraction and necessity:** runs both add-to-deleted-background and remove-from-native configurations;
- **selective manipulation:** measures induction, generic copy, matched-negative, and broad collateral separately;
- **composition/reuse:** measures every finite head interaction and searches for shared downstream consumers; and
- **OOD prediction/stability:** freezes groups before fresh task variations and shifted text.

Rank and storage are absent from the discovery decision. They matter only if an identified shared matcher or branch
is later compiled into a smaller executable program.
