# Scale-session draft (width 1152, rented box) — for the original session to fold in

Scale session, started 2026-08-04 ~18:00 UTC. This file is owned by the scale
session per the SCALE_RUN.md protocol; the original session folds it into
RESULTS_l0_mdl.md.

## Box and protocol facts (differ from the handoff spec in three ways)

- The rented cards are 2× **RTX 4080 SUPER 32 GB** (Ada, compute capability
  8.9), not the advertised RTX 5090s. bf16 fine; torch 2.11.0+cu128.
- **Effective batch 32 does not fit in 32 GB** at width 1152: micro-batch 32
  OOMs, so every arm runs micro 16 × 2-step gradient accumulation (preflight
  peak ~25.9 GiB; the grad-accumulation positive control passes at rel
  3.1e-5). The gradient math is identical to real batch 32; step time ~0.95
  s/step, full epoch ≈ 2.5 h/arm plus ~25 min of lr sweep.
- The original cooc corpus is not on this box (never committed).
  `data_fineweb_cooc_tokens.npy` here is a SUBSTITUTE built from fresh34k
  rows [0:6000] (pure-eval docs) so the harness imports; **no old-held cooc
  numbers are produced by the scale session** and none of the qk_s_ JSONs
  contain any.

Data: train prefix = corpus_fresh shards 00..06 concat rows [0:298496]
(9,328 steps × eff batch 32, single pass, epoch_order(0), identical across
every arm). Scale held = shard06 last 1500 rows (never trained; the 4-row
gap [298496:298500] is simply unused). Second held set = fresh34k rows
[33000:34500], the small-scale E-run held set, evaluated per-token on every
arm for direct comparability with qk_e0/e1..e5.

Positive controls pass in every arm process before training: slots model
with identity projections + zeroed writes == vanilla-1152 at init (max
|logit diff| 0.0 fp32); vectorized group penalty == naive loop at slot dim
48 (rel 1.3e-7); accumulated micro-grads == one-shot grads (rel 3.1e-5).

## Arms (qk_s_ files; one per GPU, all on the identical data order)

| arm | architecture | optimizer | status |
|---|---|---|---|
| vanilla | zero-init-write MiniBilin A, d12 w1152 | AdamW, swept | RUNNING |
| slots | partitioned write slots, no lasso | AdamW, swept | RUNNING |
| gc1e4 | slots + group-lasso 1e-4 (recipe base) | AdamW, swept | queued (round 2) |
| muonbase | slots + gc 1e-4, loss-lasso | Muon 2D + AdamW emb, swept | queued (round 2) |
| gc3e5 | slots + group-lasso 3e-5 | AdamW, swept | queued (round 3) |
| E1 slotnorm | slots + per-slot RMSNorm | AdamW, swept | queued (round 3) |

## §S1 The width-1152 gate (partition cost at slot dim 48) — HEADLINE IN

**The slot-partition cost roughly HALVES at width 1152 / slot dim 48:
slots-only minus vanilla = +0.1243 nats (seq-clustered SE 0.0011) on the
scale held set, +0.1259 (SE 0.0012) on fresh34k — vs +0.234 (SE 0.002) at
width 264 / slot dim 11 (qk_e5).** Direction confirmed on both held sets;
the fresh34k number is directly comparable to the small-scale E-runs.

- vanilla: held scale CE 4.11304, f34k CE 4.16121 (lr 0.001 interior winner,
  0 spikes, 9,328 steps).
- slots-only: held scale CE 4.23733, f34k CE 4.28708 (lr 0.002 interior
  winner, 0 spikes). Slots swept HIGHER than vanilla (0.002 vs 0.001), same
  direction as small scale.
- Train-CE curves + held-100 checkpoints every 2000 steps are in
  qk_s_w1152_{vanilla,slots}.json; per-token losses in the _heldloss /
  _f34kloss npys; paired stats in qk_s_w1152_gate.json.
**The full E5-style frontier at width 1152** (deltas on scale held,
seq-clustered SE 0.0010–0.0016; f34k agrees within 0.004 everywhere):

| arm (AdamW unless noted) | held CE | vs vanilla | increment |
|---|---|---|---|
| vanilla | 4.11304 | — | — |
| slots-only | 4.23733 | +0.124 | partition |
| gc3e5 | 4.27903 | +0.166 | lasso 3e-5: +0.042 over slots |
| gc1e4 | 4.39600 | +0.283 | lasso 1e-4: +0.159 over slots |

vs width 264: partition +0.234 → +0.124 (halves), but the IN-LOSS lasso
increment grows (1e-4: +0.107 → +0.159). Under AdamW the penalty, not the
partition, dominates the recipe cost at scale — which is exactly what the
proximal-Muon result in §S2 then removes.

**Seed noise floor:** vanilla seed-1 (init seed only; identical data order)
minus seed-0 = +0.0127 (paired se_seq 0.0008; f34k +0.0141). Init lottery at
this width/budget is ~0.013 nats, so single-seed deltas below ~0.02 are not
decision-grade; seed replications of the gate arms are running (slots_s1,
vanilla_s2 done or in flight).

## §S2 The width-1152 optimizer gate — THE SMALL-SCALE RESULT FLIPS

**Muon WINS on the lasso base at width 1152, even with the loss-lasso
convention: muonbase minus gc1e4 = −0.0850 nats (seq-clustered SE 0.0011)
on scale held, −0.0863 (SE 0.0012) on fresh34k.** At width 264 Muon LOST
this comparison by +0.076 (qk_e0m). Both arms: same architecture
(slots + gc 1e-4), same data order, each with its own interior lr winner
(Muon 0.02 / AdamW 0.002; Muon's embedding-AdamW split at 0.002).

- muonbase: held scale CE 4.31098, f34k CE 4.36100, 0 spikes, ~1.13 s/step
  (Newton-Schulz overhead ~13% over AdamW).
- **Muon vanilla control 3.96451: Muon's vanilla win GROWS with width
  (−0.1485 at w1152 vs −0.094 at w264, SE 0.0016).** Muon is the retrain
  default, full stop.
- **Proximal lasso (E7a rule) at scale: muonprox 4.24084 — beats loss-lasso
  Muon by −0.0701 (SE 0.0012) and the AdamW recipe by −0.1552 (SE 0.0014).
  The full-strength 1e-4 penalty applied proximally under Muon costs +0.0035
  over AdamW slots-only: at width 1152 the penalty is ~free when proximal.**

## §S3 E-series + combo at width 1152

**E1 per-slot RMSNorm replicates and grows: e1 minus gc1e4 = −0.0403
(SE 0.0014; f34k −0.0423), vs −0.026 at w264** — the only small-scale winner
wins again at scale (e1 4.35573, AdamW, gc 1e-4 in loss, lr 0.002 interior).

**Combo (slots + per-slot RMSNorm + proximal 1e-4 + Muon): 4.13125.** The
wins compose — combo minus muonprox = −0.1096 (per-slot norm is worth ~2.7×
more under Muon than under AdamW), combo minus the old AdamW recipe (gc1e4)
= −0.2648.

**Honest recipe premium (CORRECTION to the first combo push):** combo minus
MUON vanilla = **+0.1667 (SE 0.0014; f34k +0.1677)**. The combo-vs-AdamW-
vanilla gap of +0.018 was optimizer subsidy — same trap family as the
memorization subsidy the fresh-data protocol removed. Both referents
improved under Muon; the interpretable-architecture premium at width 1152
is ~+0.17 nats, roughly HALF the width-264 recipe cost (+0.342), with the
1e-4-strength readability penalty now free (proximal) and the remaining
cost dominated by the partition itself (+0.124 of the +0.167).

## §S4 Not yet run at scale

Wiring-Spearman / token-determined probes on the scale checkpoints (probe
eval data on this box is the substitute corpus — runnable, flagged); E4
typed token slot; E2 CP-rank caps; E3 certified-zero anneal; N=6 window;
prox-3e-5 readability point (waits on qk_e8's readability-under-proximal
verdict per the standing directive).

## §S5 E12 funnel family (handed off from local, run at w264 on the scale box)

All four arms, fresh single-epoch protocol, Muon lr 0.02 / AdamW 0.004,
gc 3e-5, 8250 steps x batch 16, 0 spikes and no token starvation anywhere
(held100@2000: E12L 5.735, E12Lv 5.645, E12a 5.789, E12b 5.705 — all far
under the 6.5 flag). Local E9a reference: final held CE 5.0547, body 15.06M
params, 64.75 Mflops/token.

| arm | design | body params | Mflops/tok | held CE | paired delta (seq-SE) | Spearman all/eff | neck→blk11 top-1 |
|---|---|---|---|---|---|---|---|
| E12L | funnel, narrow 286 = 26x11 | 19.18M | 86.0 | 5.0749 | +0.020 vs E9a (pt est) | 0.885 / 0.823 | 0.326 → 0.424 |
| E12Lv | + shared values (P_sv) | 18.39M | 84.4 | 4.9886 | −0.0863 ± 0.0023 vs E12L | 0.887 / 0.804 | 0.301 → 0.430 |
| E12a | true narrowing, 208 = 26x8 | 9.99M | 53.0 | 5.1948 | +0.1199 ± 0.0016 vs E12L | 0.827 / 0.708 | 0.283 → 0.410 |
| E12b | narrowing + shared values | 9.57M | 52.2 | 5.1107 | −0.0841 ± 0.0024 vs E12a | 0.897 / 0.846 | 0.268 → 0.441 |

Findings:
1. **The funnel itself is nearly free**: E12L costs +0.020 vs E9a on point
   estimates (paired SE pending qk_e9_a_heldloss.npy from local).
2. **Shared values are the strong signal** (Logan's exploration ask): they
   win twice, independently — −0.086 at matched bandwidth (E12Lv, which
   beats E9a outright, 4.989 vs 5.055, with +22% body params so capacity-
   confounded) and −0.084 at narrow width (E12b), while *improving*
   readability (E12b posts the family-best Spearman 0.897/0.846 and
   deep-stream token recovery 0.441).
3. **True narrowing is where the cost lives**: +0.12 for 264→208, but it
   buys 2x step time (0.117 vs 0.235 s/step) and half the body params.
   E12b at 2/3 of E9a's body params and 81% of its flops is only +0.056
   over E9a (pt est) — the best CE-per-flop of the family.
4. **Neck diagnostics**: attention's neck read P_a uses about half the
   available dimensions (eff. rank 132–135/286, 92–93/208); the MLP read
   P_m is near full rank (261–266/286, 193–198/208); W_up and P_sv sit
   between. The attention path is the compressed one — narrowing squeezes
   attention, not the MLPs, consistent with E12b (which shares attention
   values from the wide block) recovering most of the narrowing cost.
5. Token-determined R² declines monotonically with depth (mlp0_wide 0.85
   → ~0.55 by mlp9-11 in E12L) — downstream computation is progressively
   less token-determined; the e9a comparison probe needs a local run
   (qk_e9_a.pt absent here).

Ops notes: funnel_light_probe/_ce_with off-by-one fixed (81724d7); E12a's
stale-code crash recovered by idempotent relaunch, no training lost;
qk_e12_a_gpu0.json / qk_e12_b_gpu0.json merged into qk_e12.json after both
processes exited (no key conflicts).

## §S6 Overnight transfer verdicts (2026-08-06): the w264 structural wins DIE at w1152

All arms: combo3e5loss recipe conventions (per-slot norm + Muon 0.02 + in-loss
lasso 3e-5), identical data order, param-matched per Logan's directive, 0
spikes everywhere. Baseline combo3e5loss = 4.10596 (paired seq-clustered SEs).

| arm | design | active body | held CE | vs recipe | Spearman all/eff |
|---|---|---|---|---|---|
| combo3e5sv | shared values (single P_sv) | 273.4M (−4.6%) | 4.1771 | +0.0711 ± 0.0010 | 0.638 / 0.614 |
| combo3e5svpb | per-block P_sv (EXACT param match) | 286.7M | 4.1610 | +0.0550 ± 0.0010 | 0.706 / 0.597 |
| shrink3e5 | E16b shrinking channel, 192-dim floor | +2.7% | 4.1692 | +0.0633 ± 0.0011 | remnant-aware probe pending |
| funnelsv | scale funnel 1536→1118 + sv | 283.1M | 4.1775 | +0.0715 ± 0.0012 | funnel probe pending |
| funnel | scale funnel 1536→1092, no sv | 283.6M | 4.2157 | +0.1097 ± 0.0014 | funnel probe pending |

Findings:
1. **Every w264 structural win flips sign at w1152**: shared values (−0.084 at
   w264 → +0.055 param-matched), shrinking channel (−0.0315 → +0.063), funnel
   family (E12L was +0.020 over E9a at w264-scale; here even the better funnel
   arm is +0.0715). Constant-width combo3e5loss is a far stronger baseline at
   width than its w264 counterpart.
2. **Decompositions survive; absolute wins don't.** Within the funnel, shared
   values still help (−0.0382 ± 0.0013), same directional mechanism as w264.
   Param matching recovered 23% of the sv deficit (−0.016 of +0.071); the
   remaining +0.055 is the architectural constraint.
3. **Readability doesn't rescue them**: combo3e5sv/svpb Spearmans (0.64/0.71
   all-pairs) are in the recipe's own range — the constrained arms don't buy
   extra wiring readability at width.
4. Working hypothesis (matches the E14a census): at 48-dim slots the module
   writes have spare rank to carry token identity, so protected embedding
   bandwidth (E16), tied value content (sv), and wide-detok funnels stop
   paying; at 11-dim slots those same constraints relieved real saturation.
5. Ops: whole night ran on two gated chains with zero interventions; all
   pairs, probes, and JSONs merged race-free.

Morning queue: shrink3e5 remnant-aware wiring probe (needs per-consumer
ablation means), funnel light probes on the two scale funnels, then whatever
the local session's E19 readability dial suggests for the width story.
