# Explanation — 2026-08-30 23:35 UTC: the block-16 arc, the two ledgers, and the retrieval price

Convention everywhere below (§2135): assembly numbers are CE **added above the real model** — lower is better.
This continues explanation_1955.md (which carries the §2135 retraction story and the glossary).

## What the late evening established (§2136–§2157)

**1. The price map of the frontier is finished, and it is concave and additive.** mlp4/5 can shed nothing
(+0.024 for a quarter-trim); c6/c7 keep 576 of 2304 units; c8/c9 keep 288; per-piece pruning costs compose
linearly (three registered additive predictions landed at 0.0016 / 0.0053 / 0.0005). Best full-coverage config:
mlp45-2304 + c6/c7-576 + c8/c9-288 = **2.6662 fresh** (§312: 2.6735) at ~25.9M fewer stored values.

**2. Two ledgers, never conflated again.** In (stored values, damage), every tail-attention dictionary is
dominated by not replacing (all eight marginals ≥ 0, and each costs ~5.3M values). The dictionaries buy
*coverage* — components described — and that is the reverse-engineering goal, not compression. Envelope points,
honestly labeled as coverage retreats: skip-a16 (2.5091), skip-a14+a16 (**2.4230**, sub-additive vs the naive
sum exactly as the interaction rules predict).

**3. The tail price is a retrieval price.** Per-class attribution at both expensive dictionaries gives the same
verdict: ~90–97% of the damage sits on positions whose target token appeared earlier in the document (ind) plus
the open-class remainder (other) — the six mean classes lose nothing. The layers differ only in HOW they read:
attn16 through 3–4 heads (16.3/16.4/16.0 + the window-revealed 16.5), attn14 diffusely through six (one head,
14.2, is *negative* in the deployed context). No per-position grammar can carry this — which is why class-linear
upgrades bought nothing (+0.0037) and the a16L dictionary is +0.038 **worse than deleting the attention
outright**.

**4. Instrument discipline, sharpened by three bugs and two scope failures.** Silent voids (a wrong-loop patch,
a substring-matched splice that deleted the `__main__` guard, an inert hook on an overridden module) are now
guarded by in-script tripwires and line-anchored splices; the rerun-noise floor is measured (0.0003); additive
predictions are licensed only for individual eval-scoped marginals within a kind (joint-throughout removals
carry ~+0.01 interaction terms, §2154); and every reduction claim needs the eight windows first (the three-head
attn16 claim died at window grain; the four-head version costs +0.0097 — a trade, not free).

## In flight

Rung 64 (running): the first constructive **retrieval primitive** — for ind positions, a16's stand-in outputs
(stream at the last occurrence of the target token) @ W_ptr instead of a per-position map. Rung 65 (queued):
is **m16** retrieval-shaped too? An MLP cannot read other positions — if its deletion damage still concentrates
on ind/other, block 16 is a coupled retrieval unit (attention reads, MLP transforms), unifying the m16 remainder
(§2098–§2100, §2127) with the retrieval price. Block 16 is the program's named open object; these two rungs are
the first constructive and the first unifying test against it.
