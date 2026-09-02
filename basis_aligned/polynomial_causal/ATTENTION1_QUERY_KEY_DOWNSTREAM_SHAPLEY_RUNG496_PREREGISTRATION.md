# Rung496 preregistration — split attention1 Q/K sides and group them by downstream use

**Registered:** 2026-09-02 17:12 UTC

**Parent:** lawful rung495b strong null (`A=true, B=false`)

**Claim level:** held-out downstream-use screen. A pass identifies a candidate shared input side, not an executable
circuit, semantic label, or compression.

## Question

Rung495 split attention1 below whole heads into complete score1/score2/value interaction pieces. No cross-head pair
had a stable enough downstream effect. That can miss a smaller shared computation: two heads may read the same
query-side feature but combine it with different key-side features, or share a key-side feature with different
queries and outputs.

This rung asks whether any query or key side from different attention1 heads is treated as the same variable by the
existing downstream circuits. It advances cross-head grouping, within-head splitting, computational specification,
held-out prediction, and stable identification. It does not optimize rank or reconstruction.

## Exact five-factor attention computation

For each head, attention1 has five factors:

1. `Q1`: the first normalized, rotary-positioned query vector;
2. `K1`: the matching first key vector;
3. `Q2`: the second query vector;
4. `K2`: the second key vector; and
5. `V`: the value vector.

At query position `i` and earlier source position `j`, the head's raw contribution is

`[(Q1_i dot K1_j)/128] * [(Q2_i dot K2_j)/128] * V_j`,

followed by that head's slice of the output projection into the 1,152-dimensional residual stream. The causal mask
is retained exactly.

For each of the four exact MLP0 removals `T/C/I/S`, capture the five factors in the normal and branch-absent runs.
Evaluate all `2^5=32` arms that choose each factor from one of those two runs. All arms are reconstructed in float32
from the real deployed states and first-value tensors; the suffix and its gradient remain the production BF16 model.

Five-way Möbius subtraction produces 31 nonempty interaction terms `m_S` per head. Rather than treating those 31
terms as separate candidate states, allocate every interaction equally among the factors participating in it:

`phi_i = sum over S containing i of m_S / |S|`.

This is the Shapley allocation: a factor receives its average marginal contribution over every possible ordering of
the five factors. It is used here because it is symmetric and preserves the exact finite change:

`phi_Q1 + phi_K1 + phi_Q2 + phi_K2 + phi_V = normal write - branch-absent write`.

The word "allocation" is important. It is a convention for assigning interaction terms, not a claim that the model
internally computes five additive modules.

## Allocation-robust views

To prevent the average allocation from manufacturing a match, retain two endpoint marginal contributions for every
factor `i`:

- **factor first:** `F({i}) - F(empty)` — change only this factor while every partner remains branch-absent;
- **factor last:** `F(all) - F(all except {i})` — change this factor after every partner is already normal.

A candidate is selected using the Shapley view, but must keep the same signed relation in both endpoint views. These
three computations use the same 32 exact arms and add no model forwards.

## Gauge and candidate set

Compatible invertible changes of the private 128-dimensional Q/K coordinates leave every arm, raw residual write,
and downstream contraction unchanged. The candidate objects are therefore gauge-invariant even though individual
Q/K coordinates are not.

The primary bank has 36 input sides: nine heads times `Q1,Q2,K1,K2`. `V` remains in exact closure and is reported,
but rung495 already tested cross-head output/value pieces, so it is not eligible for primary selection. Compare only:

- query with query (`Q1` or `Q2` across distinct architectural heads); and
- key with key (`K1` or `K2` across distinct architectural heads).

Q1 may match Q2 across heads, and K1 may match K2. Query-to-key pairs are not eligible because their token roles are
different.

## Downstream-use signatures

Use the same real normalized suffix derivative as rung495. For each of the 62 curated circuit tags, separately take
member-position and matched-control-position CE gradients with respect to the raw attention1 write on each
branch-absent trajectory. Contract that gradient with every factor contribution. A fingerprint is member response
minus matched-control response, stacked over all four MLP0 branches and circuit tags.

Compute raw fingerprints and fingerprints after subtracting, at each circuit coordinate, the mean over the four
MLP0 branches. A factor is material if its Shapley fingerprint norm is at least 5% of the complete attention-route
fingerprint norm in both discovery halves. Its factor-first and factor-last views must each be at least 1% of the
complete norm so their robustness checks are live.

## Frozen data, selection, and controls

- Discovery uses the same 32 discovery tags and documents0:500. Select on documents0:250; confirm without
  reselection on documents250:500.
- Only an A/B discovery pass opens the 30 validation tags and documents500:1000. These are held out from pair
  selection, but are not described as a new corpus.
- Sixteen one-sided circuit-label permutations use seeds `20260902960..20260902975`.
- A preliminary survivor triggers a second discovery pass for that fixed pair with token-position rolls1..16. The
  rolls cannot change the candidate.
- The closest eligible pair is chosen by Shapley cosine, then lower scaled residual, then lexical name. It must be a
  mutual nearest neighbour in the Shapley view.
- No failed factor role, architectural head, MLP0 branch, circuit tag, allocation view, or document half may be
  dropped.

## Predictions

### A — exact and live instrument

All parent hashes, row/mask identities, supports, calls, and backward counts match. Independent float32 factor arms
reconstruct both endpoint attention writes with relative squared error at most `1e-10`; the 31 Möbius terms reconstruct
the endpoint difference to `1e-10`; the five Shapley pieces reconstruct it to `1e-10`; and summed Shapley-gradient
contractions reconstruct the complete contraction to `1e-9`. Every branch and every eligible side has nonzero
Shapley, factor-first, factor-last, gradient, and control norm.

### B — one shared query or key side survives discovery confirmation

On documents0:250, the selected pair must have Shapley cosine at least `.90`, best-scale residual at most `.45`, both
members material, mutual-nearest status, and cosine at least `.10` above the circuit-permutation and position-roll
95th percentiles. On documents250:500 without reselection, require cosine at least `.80`, residual at most `.60`,
materiality, discovery-scale drift at most 50%, and both control margins at least `.05`.

All clauses hold for raw and branch-mean-removed fingerprints. In both halves and both centering views, factor-first
and factor-last cosines must preserve the Shapley scale sign and be at least `.65` on selection and `.55` on
confirmation; each must exceed its circuit-permutation 95th percentile by `.05`.

### C — the frozen side predicts held-out documents and circuit families

Only if A/B hold, evaluate the fixed pair on documents500:1000 and the 30 validation tags. In both fixed halves,
Shapley cosine is at least `.75`, residual at most `.65`, both pieces remain material, the scale sign is preserved,
and circuit/position margins are at least `.05`. Factor-first and factor-last cosines are each at least `.50`, keep
the scale sign, and beat their circuit-permutation 95th percentiles by `.05`.

### D — the pair shares one side rather than duplicating the whole head

For each direction, compare the selected side with its same-head partner on the other side of that score branch
(`Q1` with `K1`, `Q2` with `K2`, and conversely), and compare the two heads' complete five-factor attention changes.
In both discovery halves, at least one of these must hold:

- selected-side Shapley cosine exceeds the corresponding opposite-side cross-head cosine by `.20`; or
- selected-side Shapley cosine exceeds the complete-head-change cosine by `.20`.

The same alternative must repeat on validation. This distinguishes a shared input side combined with different
partners from two broadly duplicate heads.

### E — interpretation

E is true only if A/B/C/D are true. The result is then called a **shared query-side or key-side downstream-use
candidate**. It still requires a separately preregistered finite input-side interchange with target-effect
preservation and unrelated-circuit controls.

## Nulls and routing

- A failure repairs the instrument only; it licenses no scientific successor.
- A true/B false: no robust shared Q/K input side is visible under these 32 downstream probes. Move to the broader
  predictive-state causal quotient across module boundaries; do not tune rank, sparsity, allocation, or thresholds.
- A/B true/C false: the relation is discovery-data/circuit-specific and is not identified.
- A/B/C true/D false: retain whole-head redundancy as a possibility, but reject the shared-side-with-different-partner
  interpretation.
- A/B/C/D true: preregister the finite input-side interchange before opening its outcomes.

## Literal price

Discovery has 125 batches. Each batch uses one native prefix capture and four branch-absent suffix/gradient forwards:
125 native prefixes and 500 full branch-absent forwards. The exact discovery backward count is derived from the
nonempty frozen masks before model loading and must match the receipt (expected 28,364 under the current authority).
The 32 factor arms and all allocations are tensor contractions, not model forwards. A conditional fixed-pair
position pass has the same 625-forward ceiling; conditional validation has the same ceiling. The experiment saves
zero deployed parameters and adds zero runtime parameters.

No threshold, factor allocation, candidate role, data split, control, or validation rule changes after any rung496
downstream-use outcome is opened.
