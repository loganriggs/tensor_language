# Rung520 preregistration: group earlier module writes by their complete MLP10 interaction stars

**Registered:** 2026-09-03 03:45 UTC, after the valid rung519 strong null and before any rung520 model forward.

## Goal and decision

The project needs circuit units that can cross native module boundaries, split a module when its parts have different
uses, predict held-out behavior, and survive finite removal or substitution. Rung519 showed that the exact individual
MLP0 interaction terms of one attention source are large but not specific to one known circuit. The registered route
therefore leaves MLP0 source refinement.

A historical audit rules out the obvious duplicates:

- rung418 compared attention0 folded Q/K subspaces and found diffuse sharing but no reusable cross-head half;
- rungs426/430 trained sparse cross-head Q/K and coupled score vocabularies; they reconstructed real structure but did
  not yield stable, specific atom identities;
- rung480 found gauge-stable directions inside the successful continuous attention0 block, but their32-circuit
  meanings did not transfer across views;
- rungs495/496 tested complete attention1 score/value pieces and Q/K-side downstream allocations; the best Q2 relation
  changed from cosine`.55` to approximately zero or negative on the second discovery half;
- rung510 already removed all1,012 action-by-MLP10 individual source-pair terms and found no pairwise downstream
  equivalence; rerunning all253 single terms is therefore not new.

The missing architecture-defined object is a **source interaction star**. Rather than ask whether one pair term is a
circuit, jointly remove every MLP10 bilinear term involving one earlier attention or MLP write. Cancellation among
those terms may make their complete downstream role simpler than any individual pair. This directly tests grouping
earlier modules by how MLP10 uses them, with the existing circuit families defining sameness. It is not rank,
reconstruction, quantization, an SAE, or a top-k search.

## Exact computation

Use rung507's coefficient-correct decomposition of the residual stream before MLP10:

`r = r_E + sum_(i=0)^10 r_Ai + sum_(i=0)^9 r_Mi`.

`E` is the accumulated embedding/skip contribution, `Ai` is attention layer`i`'s output contribution after residual
mixing, and `Mi` is MLP layer`i`'s output contribution. After applying the token's native RMS-normalization gain, the
22 named normalized sources are `z_s`. An explicit numerical residual closes their sum to the actual normalized
MLP10 input.

For the bilinear MLP

`B(u,v) = Down[(Left u) * (Right v)]`,

the253 unordered named terms are `B(z_s,z_s)` and `B(z_s,z_t)+B(z_t,z_s)` for`s<t`. Under equality-score action`a`,
rung507 defines the score-dependent change of term`p` relative to the score-absent trajectory as

`delta_a(p) = term_a(p) - term_absent(p)`.

For named source`s`, define its interaction star

`STAR_a(s) = sum_(p containing s) delta_a(p)`.

Each star has22 pair terms: one self term and21 cross terms. Removing a star subtracts this complete tensor from the
deployed MLP10 output and runs layers11--17 normally. Stars overlap: the cross term `(s,t)` belongs to both `STAR(s)`
and `STAR(t)`. They are candidate causal roles, not an additive partition. The overlap is reported explicitly and no
sum of stars is called the MLP10 output.

## Four actions, data, and observations

Use the same calibrated equality-score implementations as rungs505--515: native`N`, positive L5H5 replacement`P`,
and correctly sign-adjusted L7H3/L8H3 replacements`Z7/Z8`. One observable star node is `(action a, source s)`, giving
`4*22 = 88` nodes.

Discovery uses documents500:748, split500:624 and624:748, and the frozen32 even-root circuit tags. These data were
opened by rung510 for individual pair terms, so they are training evidence here. Confirmation is conditional and uses
documents752:1000, split752:876 and876:1000, plus the other30 circuit tags that rung510 left unopened. Documents
748:752 remain unused.

For every finite star removal, record:

- four copy-task CE changes: near copy, far copy, one previous match, and multiple previous matches;
- all-copy and off-target CE changes; and
- each circuit's member-position CE change minus its matched-control-position CE change.

The32/30 member/control masks are deduplicated by exact hash before response comparisons. Both document halves must
have positive support for every task and every retained circuit member/control cell.

## Discovery equivalence

For nodes`u,v`, fit one signed scale on discovery half0 circuit effects:

`beta(u <- v) = dot(C_0(v),C_0(u)) / dot(C_0(v),C_0(v))`.

Test all`88*87/2 = 3,828` unordered pairs; do not rank or choose nearest neighbours. A pair passes only if:

1. each node has pooled circuit RMS at least`.0005` nat and pooled four-task norm at least`.00025` nat;
2. `.25 <= abs(beta) <= 4`;
3. circuit cosine is at least`.90` with relative residual at most`.35` in half0, and cosine at least`.80` with
   residual at most`.50` in half1, using the frozen scale;
4. task cosine is at least`.70` with relative residual at most`.65` in both halves under the same scale; and
5. both prediction directions pass explicitly.

Keep every passer. Prediction B requires1--16 real pairs and a real count strictly above the higher-interpolation95th
percentile of16 fixed controls. Each control independently permutes every node's32 circuit identities while retaining
its halves, task effects, scale, and marginal effect distribution. Seeds are`520100..520115`. More than16 is a
non-identifying observation basis, not permission to select16.

Report separately:

- **action portability:** same named source under different score actions;
- **cross-source equivalence:** different named sources with the same finite downstream role;
- **cross-kind equivalence:** a passing pair joining embedding, attention, or MLP source kinds; and
- **multiple-mediator discrepancy:** for each star, compare its actual finite response with the sum of rung510's
  already-measured individual-term finite responses. This difference is descriptive evidence of joint interactions,
  not a selector.

## Held-out prediction and physical substitution

Only B opens confirmation. The candidate identities and discovery scale remain fixed. In both confirmation halves and
pooled data, require circuit cosine at least`.75`, residual at most`.55`, task cosine at least`.70`, task residual at
most`.65`, both materiality floors, and the original scale sign. No confirmation-only pair may enter.

For each confirmed pair, perform both real substitutions on the confirmation documents. If
`u=(a,s)` and `v=(b,t)`:

1. in action`a`, subtract `beta * STAR_b(t)` instead of `STAR_a(s)`;
2. in action`b`, subtract `(1/beta) * STAR_a(s)` instead of `STAR_b(t)`.

The donor tensor is computed on the same tokens from its own action trajectory. Layers11--17 are recomputed. Each
direction must predict the corresponding native star-removal response with circuit cosine at least`.75`, residual at
most`.55`, task cosine at least`.70`, and task residual at most`.65`, pooled and in both halves. Added off-target CE
must be at most`.002` nat in each half. Similar response vectors without physical substitution do not count.

Multi-node quotient groups require every within-group pair to pass both substitutions and fitted scales around every
cycle to multiply to within25% of one. Otherwise report only the passing pairs.

## Frozen predictions

### A — exact, live source-star instrument

All parent/result/source/preregistration hashes match; the22 named normalized sources,253 named pair terms, and every
22-term star index close exactly; all32/30 circuit mask pairs are distinct or deterministically deduplicated; native
and score-action calibration replay; every requested star edit is finite, nonzero, and dispatched once; task/circuit
supports and forward/capture/patch counts match; and eight planted88-node response tables recover their exact pair and
destroy it under the circuit controls.

The sum of a star's22 term tensors must equal an independently computed fixed-normalization removal of source`s` from
both MLP10 input branches to relative squared error at most`1e-8`. Deployed star subtraction and native analytical
replay use rung507's repaired BF16 relative-squared bounds.

### B — a small circuit-defined source-star relation exists

Between1 and16 node pairs pass every discovery rule without ranking, and their count strictly exceeds the fixed
permutation-control95th percentile.

### C — at least one relation predicts unseen documents and circuits

At least one frozen B pair passes every confirmation rule over the other30 circuit tags and both new document halves.

### D — at least one relation is physically interchangeable

At least one C pair passes both tensor substitutions, including off-target preservation.

### E — the quotient crosses a native source boundary

At least one D pair uses different named sources. Same-source action portability is retained but cannot satisfy E.
Report whether any E pair joins attention and MLP sources or different layers.

`strong_null = not (A and B and C and D and E)`.

## Price and stopping rules

Discovery has62 four-document batches. Each runs one score-absent capture and, for four actions, one intact suffix plus
22 star removals: `62 * [1 + 4*(1+22)] = 5,766` full-model forwards,0 backwards. Conditional confirmation has the same
price. If`q<=16` pairs confirm, bidirectional substitution costs
`62 * [1 score-absent + 4 intact + 2q substitutions] = 310 + 124q` forwards. The maximum is13,826 forwards,0
backwards, at most16 fitted scalars, and0 deployed parameters added or saved.

- A false repairs only the named exactness/support/dispatch clause; no scientific result is interpreted.
- A true/B false leaves source-star grouping. Do not lower thresholds, return to individual terms, add rank, or select
  the closest pair. Move to a task-defined finite state transition spanning several downstream sites.
- B true/C false retains a discovery screen only.
- C true/D false means response similarity is not physical interchangeability; localize the first consumer that
  separates the pair.
- D true/E false establishes action portability only, not grouping of earlier modules.
- A--E identifies a cross-source MLP10 interaction role but is not adoption. A later joint installation must compose
  with other circuits and earn literal storage/compute savings.
