# Handoff — Phase V4, the six architectures at DEPTH 3

Written 2026-08-08 18:10 UTC for a reader with no memory of the work.

## The question, in one paragraph

At **depth 2, width 128** the five interpretable architectures (`slots`,
`bandwidth`, `predicate`, `codebook`, `shrink`) did two things the plain model
did not: they used a residual route that carried ~nothing in vanilla
(1.3e−5 nats) and they acquired **induction** at a width where vanilla needed
256 (RESULTS.md FINDING 11). At **depth 3** the plain model does *both by
itself* — it inducts at width 128 (+0.1085 ± 0.0133 over three seeds) and its
later attention blocks transmit 0.16 nats into the next layer's read
(FINDING 14, as corrected by FINDING 16). So: are the architectures merely an
**accelerant** for what depth supplies, do they still **add** something, or do
they **interfere**? Predictions PD1–PD7 were registered in
`tf_d3_variant_predictions.json` **before the first training step**, with an
explicit decision rule.

## State

| thing | where |
|---|---|
| registered predictions, written first | `tf_d3_variant_predictions.json` |
| training + analysis + route-use chain | `tf_d3_variant_chain.sh`, log `tf_d3_variant_chain.log` |
| verdict generator (nothing is transcribed by hand) | `tf_d3_variant_report.py` → `tf_d3_variant_slice.json`, `tf_d3_variant_table.md` |
| vanilla d3 w128 s0/1/2 | **done** — reused from the depth ladder, NOT retrained (identical command, data order and optimizer) |
| the other 5 variants × 3 seeds, plus 5 control cells | **in flight** at the time of writing, roughly 8 minutes a cell |

The chain is **idempotent** (it skips any cell whose `.pt`, `_interp3.json` and
`_routeuse.json` exist), detached, and needs no babysitting. If it died:

```bash
cd /workspace/tensor_language/basis_aligned/tiny_full_interp
source /venv/main/bin/activate
pgrep -af -- 'tf_[d]3_variant_chain\.sh'          # alive?
setsid nohup ./tf_d3_variant_chain.sh > tf_d3_variant_chain.stdout 2>&1 < /dev/null &
python tf_d3_variant_report.py                    # regenerate the verdict
```

**Do not report from `tf_d3_variant_slice.json` until every cell in
`tf_d3_variant_chain.log` has landed** — it is regenerated from whatever
exists, so a partial run yields a partial verdict that looks complete. Both
partial artifacts were deliberately left uncommitted for this reason.

## The forced deviation you must mention in any write-up

The masked-decoder variants (`slots`, `shrink`) put one slot per module, so
depth 3 wants n_slots = 2 × 3 = 6 — and **128 is not divisible by 6**. The only
n_slots that divides 128 and leaves every module a nonempty write mask is
**8**, so `slots` and `shrink` run **8 slots of 16** instead of depth 2's 4 of
32, with two slots written by nothing. The small-decoder variants
(`bandwidth`, `predicate`, `codebook`) are unaffected — they scatter into 6
solved slots, stream 168. Two controls price the change and are in the same
chain:

* `tf_slots_d2_w128_b8192_s{0,1,2}_g8` — the same geometry change at the
  depth-2 cell whose n_slots = 4 answer is already published (CE 4.7414,
  induction +0.0972 ± 0.0275);
* `tf_vanilla_d3_w192_b8192_s0` and `tf_slots_d3_w192_b8192_s0` — depth 3 at a
  width where 6 slots × 32 is exact and the slot size matches depth 2.

If the slice's verdict turns on `slots`/`shrink` alone, the controls decide
whether it is the architecture or the geometry.

## Two rules this slice inherits from the round-4 review

`tf_reviewer_round_4.json` (independent review of the depth ladder) changed
how route and induction numbers may be quoted, and the report script already
complies:

1. **A read-ablation KL is a magnitude, not a route.** Over 243 write/read
   pairs in the ladder, log KL regresses on log of the write's norm share of
   the read it enters with slope 1.99 and r = 0.994, residual 0.26 dex. So
   every route KL must be printed **beside the write's norm share**, and the
   words "open" and "shut" are not available. For the variants this matters
   more, not less: their per-slot norms mean the write-norm-share denominator
   is a different object than in vanilla, and **the O2b regression must be
   re-derived on variant checkpoints before it is applied to them** — that is
   the first analysis to run once the cells land.
2. **Induction is decided over MODEL seeds**, with a t-test, not against the
   probe-noise floor — the floor shrinks as 1/√(probe seeds) and is not a
   property of the model.

## The first thing worth doing after the verdict

The early single-seed signal (do not quote it, it is one seed of one variant)
points at a split result: the variants may **lose** the induction advantage at
depth 3 while **keeping** a genuinely different mechanism — `slots` seed 0 runs
a large share of its induction through its layer-0 attention write, which
vanilla cannot do at any depth because vanilla's layer-0 attention writes
nothing. If the finished slice confirms that, the interesting follow-up is
**depth 4 at width 128 for `slots` and vanilla only** (2 cells × 3 seeds,
~50 minutes): does the variants' own route keep carrying the algorithm as
depth keeps supplying its own, or does it get crowded out?
