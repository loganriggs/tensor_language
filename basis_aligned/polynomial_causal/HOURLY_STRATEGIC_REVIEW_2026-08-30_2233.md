# Hourly strategic review — 2026-08-30 22:33 UTC

Convention (§2135): L2 = CE added above the real model; LOWER is better.

## The hour: from price map to the block-16 program

- **Envelope finished for the middles (§2141–§2144):** mlp4/5 sheds nothing (+0.024 for a quarter-trim); c6/c7
  need 576; c8/c9 need 288; per-layer costs compose linearly (two additive predictions landed at 0.0016 and
  0.0053). Best all-dictionary config: mlp45-2304 + c6/c7-576 + c8/c9-288 = 2.6662 FR (§312: 2.6735), 8/8
  windows better, ~25.9M fewer stored values (corrected unit accounting).
- **Tail attribution (§2145):** the +0.35 tail increment is concentrated — a16L +0.157 (45%), a14L +0.073,
  a12L/a17L nearly free.
- **Block 16 is the program's open problem, now sharply characterized:** its residual is damaging per unit
  energy (§2147, ρ 0.786 but both sharp claims failed); class-linear upgrades buy nothing (§2148, +0.0037 worse
  on 8/8); the a16L dictionary is +0.038 WORSE than deleting attn16 outright (§2149) — any new grammar must beat
  zero; and attn16 is THREE HEADS (16.3/16.4/16.0 carry +0.113 of +0.119; six heads sum to −0.001, additive to
  0.0072). Leaving attn16 real strictly dominates in (price, damage) at one component of coverage (§2146).
- **Instrument discipline held:** two crashed runs voided and re-run (rung 54's wrong-loop patch — which
  incidentally measured a 0.0003 rerun-noise floor — and rung 56's double-del, fixed this wake); §312's
  inapplicable sanity band recorded, not retired, in §2146/§2149.

## Strict ledger and gaps
Unchanged: 5.348% / 10.923% / 4.727 nat / 0 of 68. Gaps: block 16 (m16 + attn16 — now the single named target),
attn5's write (cliff; no mechanism yet), coverage credit at the tail spans.

## Ranked next (queue: rung 57 running — three-head attn16 as a registered number; rung 56 re-queued — which
classes pay a16L)
1. Consume 56/57: if attn16 reduces to three heads, the block-16 attention program is 3 heads + per-class damage
   map — enough to DESIGN a targeted stand-in (e.g., per-head low-rank pattern dictionaries for 16.0/3/4 only).
2. Same head-grain attribution at a14 (the second-costliest dictionary) — cheap, same instrument.
3. Eight-window certification of the (skip-a16, three-head) envelope points before they enter the registry as
   more than FR numbers.
4. m16 revisited through the three-head lens: do 16.0/3/4 read m16's two source directions (§2098)? A weights
   pass, no arms.
5. attn5-write stand-in — still parked pending a mechanism.
