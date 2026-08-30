# Induction equality-tensor extraction/removal discovery

**Frozen before this model outcome:** 2026-08-30 04:30 UTC.

## Question and claim boundary

The earlier four-head induction intervention had strong target necessity but failed
collateral under unconditional position-mean replacement. Test the more mechanistic
fixed tensor: retain or delete only edges whose key immediately follows an earlier
token equal to the current query token.

This uses the already-opened 192-document terminal-copy SELECT role and its previously
frozen positive, matched-negative, and off-target masks. It is discovery only. FINAL
natural, OOD code, and synthetic outcome containers remain unopened and unauthorized.

## Fixed tensor and heads

The registered heads remain `L5H5`, `L7H3`, `L8H3`, and `L8H4`. For token one-hots
$e_{t_i}$, define

$$
M_{qk}=\langle e_{t_q},e_{t_{k-1}}\rangle\mathbf 1[1\le k\le q].
$$

For each named head with native quadratic score $A^{(h)}_{qk}$ and value $v^{(h)}_k$,
the equality-fetch tensor is

$$
z^{(h)}_{q}=\sum_k M_{qk}A^{(h)}_{qk}v^{(h)}_k.
$$

All matches are summed. There is no nearest-match, argmax, TopK, or `has_match`
branch. Integer equality in the executor is the sparse compiled evaluation of the
fixed vocabulary identity tensor.

The same-price null replaces identity by the fixed cyclic permutation
$e_i\mapsto e_{i+1\bmod V}$ on the query leg.

## Arms

- `native`;
- `full_replay`: analytical replay of all attention at the three affected layers;
- `remove_equality`: native named-head computations minus equality-fetch edges;
- `heads_deleted`: the four named heads set to zero;
- `extract_equality`: `heads_deleted` plus only equality-fetch edges; and
- `deranged_equality`: `heads_deleted` plus only fixed-permutation equality edges.

Interventions are applied sequentially on the live residual stream, so later heads see
the effects of earlier replacements. All other heads and all MLPs remain native.
Analytical arms must call none of the native attention modules or Q/K/Q2/K2/V/O
submodules at layers 5, 7, or 8. Full replay must have maximum logit error at most
$10^{-4}$ and mean KL at most $10^{-8}$.

## Metrics and frozen gates

Report per-document CE, native-to-arm KL, and top-1 change on the frozen `positive`,
`matched_negative`, `off_target`, and `all` masks. Use 20,000 shared-draw document
bootstrap repetitions.

The candidate is eligible for a fresh terminal run only if:

1. equality-edge removal has positive target-damage and target-minus-matched-negative
   95% lower bounds;
2. extraction recovery from `heads_deleted` is at least 0.80 point and 0.60 lower bound;
3. off-target removal CE has 95% upper bound at most 0.01 nat and point at most 10% of
   target damage;
4. the deranged tensor's recovery 95% upper bound is less than half the native-equality
   extraction point;
5. every named cell contains at least 200 tokens in at least 30 documents; and
6. replay and call-census gates pass.

No head, mask, vocabulary permutation, gate, or intervention may change after seeing
SELECT. Failure is preserved and moves the campaign onward.
