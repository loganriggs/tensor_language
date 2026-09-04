# Task 14 subject–verb agreement capability-only FIT preregistration

**Frozen prospectively:** 2026-09-04 07:45 UTC, after approval of the repaired semantic authority and before any
task-14 compiler, producer, model call, or outcome existed. **Execution status:** CPU authority/compiler construction
only. This document does not authorize model or checkpoint access, GPU use, a producer or managed adapter, queue or
enqueue, result/evidence publication, localization, or any later split.

## Question

Does bilin18 assign the correct next-token copula in the repaired subject–verb-agreement FIT panel before we spend
any computation searching for a circuit?

For ordinary A1, A2, and P rows, the answer is ` is` for a singular grammatical subject head and ` are` for a plural
head, regardless of the nearer attractor noun. C has two singular subject conjuncts joined by `and`; its answer is
therefore unequivocally plural ` are` while the attractor number changes. For each prompt $x$, answer token $y$, and
opposite copula $f$, the signed margin is

$$
m(x)=z_y(x)-z_f(x).
$$

Strict correctness means $m(x)>0$. A tie is incorrect. This is a two-candidate grammatical capability screen, not
unrestricted next-token accuracy and not circuit localization.

## Approved sources and frozen FIT authority

This unit is permitted by replacement-authority review commit
`ea7efad782c088ba91a2ce338a9f740563c4e7c1`. The exact source closure is:

| Role | SHA-256 |
|---|---|
| replacement generator | `33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94` |
| generator adversarial tests | `254fe3798efd8a4426f30e054fd8e5646a5bd6635df69815f376311ac2023694` |
| repaired design memo | `3cb4556d1ad2c1564f2708028e5d624c4519fbc4d52a38cac27b9d10d8312f68` |
| replacement-authority review | `7249991dd727f6385d3269cce23b0e5f83c588bcef3488dce33ae19dfd223fd1` |
| full logical authority | `1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1` |
| FIT logical records | `3cf3315a77b3176418739e7a9357c0dbd9b95724d6b276038f53691b873377d1` |
| FIT authority file | `e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f` |

The FIT file contains exactly 128 rows: 32 linked groups with one A1, A2, P, and C row each. Every one of its 256
base/donor prompts is unique. Each base/donor pair has equal prompt-token length and exactly one aligned changed
token. Answer and foil each add exactly one GPT-2 token at the saved continuation boundary.

Only the FIT file may enter this compiler closure. SELECT, TEST, and OOD remain logical hashes in the reviewed design;
no file containing their rows may be created, captured, generated, or opened here.

## Exact calls, order, metrics, and price

The compiler must emit these eight calls in exactly this order:

1. FIT base A1, 32 rows;
2. FIT base A2, 32 rows;
3. FIT base P, 32 rows;
4. FIT base C, 32 rows;
5. FIT donor A1, 32 rows;
6. FIT donor A2, 32 rows;
7. FIT donor P, 32 rows; and
8. FIT donor C, 32 rows.

Within a call, rows are ordered by the frozen compiler's canonical JSON policy. The physical request must bind its 32
row IDs, the selected side's exact prompt IDs, the saved prediction position, and one target and one opposite-copula
foil token ID per row. The first four call lengths are 5, 8, 5, and 8 tokens; donor lengths match their base calls.

Each call retains only one C-contiguous `float32[32]` answer-logit array and one C-contiguous `float32[32]` foil-logit
array. Full logits, activations, gradients, losses, hidden states, attention patterns, component labels, reader/writer
candidates, and localization values are forbidden.

The literal price is:

- 8 forward calls;
- 256 unique row-side evaluations;
- 0 backward calls and 0 model updates; and
- $8\times2\times32\times4=2{,}048$ raw numeric evidence bytes.

Call JSON, array headers, and future result framing are metadata, not learned numeric evidence. A price or coverage
mismatch is an invalid instrument and cannot produce a scientific projection.

## Frozen opposing capability gate

There are eight family/side cells of 32 rows. For A1, A2, and P, each family/side cell also has an exactly balanced
16-row incongruent subset: the grammatical head number differs from the nearer attractor number. The capability
prediction is the conjunction of every condition below:

1. pooled base accuracy is at least `0.85` and pooled donor accuracy is at least `0.85`;
2. every A1/A2/P family × side accuracy is at least `0.85`, and every such cell has strictly positive mean margin;
3. every A1/A2/P family × side incongruent-subset accuracy is at least `0.85`, and every such subset has strictly
   positive mean margin;
4. base C accuracy and donor C accuracy are each at least `0.75`, and each C side has strictly positive mean margin;
5. all A1/A2 rows change the registered answer from base to donor and are scored against the saved donor answer on
   the donor side; and
6. all P/C rows preserve the registered answer exactly across sides.

The integer boundaries are fixed: pooled `0.85` requires at least 109/128 correct; a 32-row `0.85` cell requires at
least 28/32; a 16-row incongruent `0.85` subset requires at least 14/16; and the C `0.75` bar requires at least 24/32.
Positive mean means strictly greater than zero; exactly zero fails.

The capability-failure prediction is the exact logical complement: any one of the conditions above is false. No
family may be dropped or pooled to rescue another, and no threshold, strictness rule, subset definition, answer, or
foil may change after model values exist.

## Fail-closed terminals

The compiler must first validate the complete authority, source, call, metric, array, and price contracts. Evidence
must cover every declared `(call_id, row_id, side, transform_id, incongruent)` key exactly once and contain only finite
numeric scalars for the two retained arrays.

If valid evidence misses the scientific gate, the only terminal is `hard_abort` and every scientific projection field
is null. No component, head, MLP, layer, reader, writer, site, subspace, localization namespace, or later-phase artifact
may be named or created. A pass may only motivate a separately frozen localization proposal; it does not itself select
a circuit or open SELECT, TEST, or OOD.

An invalid authority, closure, request, metric, array, coverage, hash, or price is an invalid instrument and must be
rejected rather than interpreted as capability failure. This document creates no result or evidence namespace and
does not authorize execution.
