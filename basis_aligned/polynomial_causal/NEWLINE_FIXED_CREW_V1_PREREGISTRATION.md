# Newline fixed-crew v1 preregistration

Status: frozen source design only.  This document grants no row, checkpoint, model,
GPU, FINAL, OOD, or publication authority.  No outcome may be opened until an
independent source audit and a create-only authority bind the final source hashes,
checkpoint receipt, tokenizer-derived token-ID sets, row receipts, and role ledger.

Base source pin: commit `b48ed78c03d3dd04e7d2420cf75b1e3edcd2d5ef`, which adds the
constant diagonal head projector to `tensor_preserving_attention.py`.  The model pin
is revision `ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240`, config SHA256
`428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c`, and weight
SHA256 `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.

## Frozen component objects

The canary is zero-indexed `L12H6`.  It must complete before the five-head run is
authorized.  The five-head crew is exactly

```text
{L7H2, L8H2, L10H2, L11H0, L12H6}.
```

The fixed head-label controls are the `h -> (h+1) mod 9` images at the same layers:

```text
{L7H3, L8H3, L10H3, L11H1, L12H7}.
```

They are controls, never alternate candidates.  The source outcome that named the
crew may not select ranks, terms, token classes, or gates in this experiment.

Every replaced site executes the complete owned nine-head squared-attention tensor:
six dense `1152 x 1152` projections, both RMS-normalized QK contractions, RoPE, the
causal mask, the block-0 value bus, lambda mixing, and `c_proj`.  A constant nine-entry
diagonal tensor is applied to the head leg before `c_proj`.  `EXACT` uses nine ones;
`REMOVE` changes only the registered crew coordinate to zero; `HEAD_LABEL_CONTROL`
zeros only the frozen control coordinate.  No newline token, parser state, TopK list,
decoded label, or score mask enters execution.

## Arms and roles

The canary plan has `NATIVE`, `EXACT`, `REMOVE`, and `HEAD_LABEL_CONTROL`, replacing
attention only at site 12 in each nonnative arm.  The five-head plan has the same arms
and replaces attention only at sites `7,8,10,11,12`.  All other attention sites and all
18 MLP sites remain native.  A fresh one-use owner and one-use attention executor are
required for every arm/batch.  Each replaced native attention call count must be zero;
each replacement count must be one; all other native component counts must be one.

`CANARY_SELECT` is opened first and may only decide pass/fail of source integrity and
the registered L12H6 canary.  `FINAL` and `OOD` remain unopened unless a later authority
licenses the five-head stage.  Documents are disjoint across roles.  Recommended
minimums are 96 documents for `CANARY_SELECT`, 192 for `FINAL`, and 192 for `OOD`, with
at least 256 newline positions and 128 newline-bearing documents in each scored role;
otherwise that role is `UNEVALUABLE`, never replenished after outcomes.

OOD strata are whole held-out domains and structures: prose, code, and lists/tables;
line-length bands; and document domains.  Random row splits from one harvested window
do not count as OOD.

## Score-only masks

The future authority freezes exact tokenizer token-ID sets and their hashes for:
newline-containing targets, other punctuation, capitalized tokens, and quote/bracket
tokens.  Positions before prediction column 64 are excluded.  The primary target is
`next_token in newline_ids`.  Position-jitter and random negatives are deterministically
count-matched within each document and exclude target and named collateral cells.
Other punctuation, capitalization, quote/bracket, and all nonnewline positions are
collateral.  Masks reduce logits to document sufficient statistics and cannot be read
by any replacement callback.

## Registered gates

Integrity precedes science: exact source/authority/row/checkpoint replay, immutable
arm order, hook census, complete call ledgers, exact first-value-bus values with
nonmutating pass-through at each replacement boundary, finite
tensors, exact storage receipts, and receipt-last publication must all pass.

The canary requires:

1. `EXACT` logits and site-12 write match native exactly under the checkpoint dtype;
2. L12H6 `REMOVE-NATIVE` newline CE damage and target-minus-position-jitter
   specificity have simultaneous 95% document-bootstrap lower bounds above zero;
3. removal global off-target CE has a simultaneous upper bound at most `0.01` nat and
   at most 10% of target damage; and
4. L12H6 removal exceeds the frozen L12H7 control removal with a simultaneous lower
   bound above zero.

The five-head stage requires the same integrity gates, extraction recovery relative to
`REMOVE` at least `0.80` point and `0.60` simultaneous lower bound, positive removal
and removal-minus-head-label-control lower bounds, and the same collateral bounds.
Every OOD stratum must preserve sign; pooled OOD removal retains at least 50% of the
FINAL point estimate with a positive lower bound.  Top-1 and native-to-arm KL are
always published.

Inference is clustered by source document.  Positions are not independent bootstrap
units.  All registered coordinates share one simultaneous family.  No rank, writer
pair, threshold, head, domain, or mask may be chosen after `CANARY_SELECT`, `FINAL`, or
`OOD` outcomes are opened.

## Literal price

For residual width `D`, `H` heads, sequence length `S`, batch `B`, and `m` replaced
sites, each dense arm stores

```text
m * [6 D^2 + 1 + D/(2H) + H]
```

floating values.  The last `H` values are the fixed head projector.  At `D=1152`,
`H=9`, this is `7,962,698` values for the canary and `39,813,490` for the five-head
plan.  Per replaced site the forward price is

```text
B * [6 S D^2 + 3 S^2 D + S D]
```

multiply/add-or-multiply operations; the final `S D` term prices the head projector.
All arms have identical literal price.  Dense identity is an executable certificate,
not a compression claim, and receives no simplicity credit.

## Remaining run blockers

There is no newline row freezer, independent row audit, role authority, tokenizer-ID
receipt, program manifest, terminal owner, or receipt-last publisher.  The source in
this commit is therefore a testable scaffold and remains launch-NO-GO.
