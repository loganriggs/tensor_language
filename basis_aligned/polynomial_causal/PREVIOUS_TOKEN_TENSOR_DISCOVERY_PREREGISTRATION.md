# Previous-token fixed-tensor extraction/removal discovery

**Frozen before model outcomes:** 2026-08-30 04:13 UTC.

## Question and claim boundary

Does the fixed previous-position component of layer-0 head 3 predict a specific
previous-token behavior, survive unseen-bigram transport, and admit both extraction
and selective removal with less collateral than same-price wrong-offset tensors?

This is a discovery run on already-opened FIT/SELECT roles. It may define the exact
candidate and gates for a later fresh FINAL/OOD transaction, but it cannot issue a
terminal certificate or add new strict-ledger credit.

## Fixed tensor circuit

For head score tensor $A_{qk}$, value $v_k$, and the fixed shift mask

$$
S^{(d)}_{qk}=\mathbf 1[k=q+d],
$$

the registered previous-token component is

$$
z^{(-1)}_q=\sum_k S^{(-1)}_{qk}A_{qk}v_k.
$$

This is a fixed tensor contraction. The head scores remain the native product of two
bilinear QK forms with RoPE and the causal mask. There is no argmax, TopK, parser, or
content-dependent branch inside the candidate. An argmax is used only to define
evaluation strata and never controls candidate execution.

The same-price nulls replace $S^{(-1)}$ by $S^{(-2)}$ or $S^{(+2)}$. Offset means
`key_position = query_position + offset`; the `+2` arm is still intersected with the
native causal mask and should therefore vanish.

## Frozen data and behavior cells

Reuse the already-opened P512 FIT and SELECT documents, 96 each, scoring query
positions 64--255. FIT outcomes do not select a head or offset: L0H3 and offset -1 are
fixed by the prior exact-fold evidence. FIT supplies only the set of observed ordered
token bigrams.

On SELECT, compute the native absolute-score top source of L0H3. Frozen cells are:

1. `previous_top`: top source is query minus one;
2. `previous_top_unseen_bigram`: the ordered `(previous token, query token)` pair does
   not occur anywhere in FIT;
3. `previous_top_seen_bigram`: its complement within `previous_top`;
4. `self_top`: top source is the query itself;
5. `other_top`: every other scored position; and
6. `all`: all scored positions.

Cell masks are computed once from tokens and the pinned native analytic pattern, then
held fixed across arms. Report supports and refuse scientific interpretation for a
named cell with fewer than 200 token positions or fewer than 30 source documents.

## Frozen arms

- `native`: the deployed model.
- `full_replay`: all layer-0 attention recomputed from pinned weights without calling
  the native layer-0 attention module.
- `remove_previous`: full layer-0 attention with only L0H3's shift -1 edges deleted.
- `head_deleted`: full layer-0 attention with all L0H3 edges deleted.
- `extract_previous`: `head_deleted` plus only L0H3's shift -1 tensor.
- `deranged_minus_2`: `head_deleted` plus only L0H3 shift -2.
- `deranged_plus_2`: `head_deleted` plus only L0H3 shift +2.

All non-native arms analytically replay the complete other eight layer-0 heads, leave
every later component native, and must make zero calls to native layer-0 attention and
its Q/K/Q2/K2/V/O submodules. `full_replay` must agree with native logits to maximum
absolute error at most $10^{-4}$ and mean KL at most $10^{-8}$ before other outcomes
are interpreted.

## Metrics and discovery gates

For every arm/cell report CE, native-to-arm KL, and top-1 change. Use a shared-draw
document-paired 20,000-replicate bootstrap.

Define removal damage on cell $C$ as

$$
\Delta_{\rm remove}(C)=CE_{\rm remove\ previous}(C)-CE_{\rm native}(C),
$$

and extraction recovery as

$$
R(C)=\frac{CE_{\rm head\ deleted}(C)-CE_{\rm extract\ previous}(C)}
{CE_{\rm head\ deleted}(C)-CE_{\rm native}(C)}.
$$

The candidate is eligible for a fresh terminal run only if, on SELECT:

1. `previous_top` removal damage and previous-minus-self specificity have 95% lower
   bounds above zero;
2. extraction recovery point estimate is at least 0.80 and its 95% lower bound at
   least 0.60;
3. unseen-bigram recovery has the same sign and at least 50% of seen-bigram recovery;
4. all-position removal CE is at most 0.01 nat and at most 10% of target damage;
5. each wrong-offset null recovers less than half the previous tensor's target CE
   benefit; and
6. replay and exact call-census gates pass.

Failure is preserved. No threshold, cell, offset, scale, or candidate may change after
SELECT in order to rescue a failed gate.
