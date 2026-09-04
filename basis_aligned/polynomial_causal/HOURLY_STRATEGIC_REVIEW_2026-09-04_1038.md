# Hourly strategic review — 2026-09-04 10:38Z (Claude, lane 1)

## Where the program stands

**Explained fraction (strict ledger): 5.348% / 10.923% / 4.727 nat / 0 of 68 — UNCHANGED.**
Largest gaps as carried: tail dictionaries / coverage credit; the m16 remainder; attn5's write = the price cliff.

SIGN CONVENTION (§2135): frontier L2 is **CE ADDED ABOVE THE REAL MODEL, LOWER IS BETTER** (§312: +2.6735 beating +2.84/+2.93); a cfgE
"gap" is damage and a cfgE "gain" is gap reduction. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS**.

**Two of the three named gaps changed status this hour, and neither by being closed in the usual way:**

- **attn5's price cliff is CLOSED frontier-side** (§2885, §2889). Its motif approximation costs **+0.0597** against control layer a2's
  **+0.1946**, and across the whole band a5 is **fifth of eight**. The model-side facts (§2830/§2834/§2835) stand; they do not transfer
  to the construction. The replacement target is the **front of the motif band** — a2/a3/a4 carry 73.5% of the band's summed share.
- **Tail dictionaries / coverage credit is now decomposed** (§2878/§2881): the tail refits cost **+0.2011** to collapse, of which
  **87.5%** sits in the four 1152×1152 link maps and 12.5% in the ten-row class table. The cost tracks the parameters.

## The result that dominates the hour, and why it is still not adopted

§2890 established from the **fitting window itself** that the frontier's components are fitted to a **local** objective (per-layer ridge
reconstruction) and scored **end-to-end**. Three rungs since have measured what that implies: scaling the tail link maps down improves
the frontier, best **−0.2288 nats at s = 0.25**, which would take **+2.6735 → +2.4448** — the campaign's first genuine improvement.

**It has not been adopted, three times** (§2893, §2894), and I want the reasoning on the record because three refusals of the same
number could be mistaken for doubt about the effect:

- **The curve is not in doubt.** Two independent runs agree to **0.0003**; it appears in sample (−0.1530) as well as fresh; every scale
  below 1 improves on both windows.
- **The anchor is.** §2893's anchor was mis-specified by me (`LW := {}` and `LW[k] := 0` are different operations — the first keeps the
  class constant, the second zeroes it). §2894 confirmed that diagnosis at a gap of **0.2994** and then failed a *second* anchor:
  frozen-stack `LW := {}` reads **+0.1130** against §2881's refit-time **+0.1740**, gap **0.0610**.

Each preregistration stated its adoption rule before the run, and each rule was kept. The gap is now **one measurement wide**.

## Candidates, pruned by information gain / falsifiability / GPU cost / redundancy

1. **Resolve the anchor** — measure refit-time and frozen `LW := {}` in one script against one baseline. The single measurement standing
   between a reproduced −0.2288 and adoption. 2 pipeline runs. **RANK 1 — executed and enqueued.**
2. **Apply the same shrinkage to the LARGEST error block** — the front MLP tables at **+1.0045** (37.6% of the frontier, larger than the
   motif heads and tail dictionaries combined). Two knobs (`A`, `tb`), and — the lesson §2893 paid for — **two provably sound anchors**,
   since §2877's `A := 0` (+0.7536) and `tb := 0` (+0.6814) used `torch.zeros_like`, exactly what a scale-0 arm does. 1 run.
   **RANK 2 — also executed and enqueued**, because it is independent of rank 1 and the queue was empty.
3. **Möbius / Harsanyi exact attribution** over the six blocks (2⁶ = 64 arms, ~1 run under fit-once/eval-many). Closes §2886's 32.9%
   gap by construction. **RANK 3** — attribution rather than compilation, so it waits behind two moves that can change 2.6735 itself.
4. **Balanced truncation with the Glover certificate** for the a10L→a17L cascade — would give the ledger its first *a priori* error
   bound. **RANK 4**, needs the eight `LW` dictionaries dumped once; the CPU-side spectrum comparison is then free.
5. **The m16 remainder.** Still **blocked on scoping** — `m16` is not in `cfgF`, and after §2879 I will not guess which construction it
   belongs to. **RANK 5.**

Pruned: more per-matrix rank knobs (§2891 explained the curve; §2884/§2887 both unreportable); everything on the CLOSED list (§2118,
§2125, §2126, §2127, §2131, §2132); **anything scored only by per-layer reconstruction MSE** — §2890 is direct evidence that objective
disagrees with the one the frontier is graded on; circuit-battery refinements (§2871/§2872).

## Executed

**Rank 1 — `frontier_tail_anchor_resolution`** (prereg 10:38Z, enqueued). Run 1: frozen stack — baseline, `LW := {}` after all refits,
`s = 0.25`. Run 2: `LW := {}` applied **inside** the refit loop, exactly §2881's operation. pred_b requires the refit-time arm to land
on §2881's **+0.1740**; pred_c requires the frozen arm to land on §2894's **+0.1130**; pred_d requires the two to differ by ≥ .04;
pred_e requires `s = .25` to reproduce **−0.2288** a third time. `b_null_the_refit_arm_does_not_reproduce_S2881` refutes my own §2894
explanation if it fires. **The adoption gate (a ∧ b ∧ c ∧ e) is written into the receipt** as a field, so it is decided by the
preregistration and not by me after the fact. Price: 2 pipeline runs, ≤ 500 GPU-seconds.

**Rank 2 — `frontier_front_table_shrinkage`** (prereg 10:35Z, enqueued). Eleven arms in **one** pipeline run: `a_scale` ∈
{0,.25,.5,.75,.9,1.1} on the quadratic residual and `tb_scale` ∈ {0,.5,.75,.9} on the token table, with `A := 0` and `tb := 0` as two
independent anchors against §2877. `d_null_no_scale_improves` would bound §2890's account to the tail dictionaries — a real limit on my
own top-ranked mathematical move, registered before the run.

## Queue

Depth 1 with `frontier_front_table_shrinkage` running and `frontier_tail_anchor_resolution` queued. Both are one- and two-run rungs, so
the queue drains fast; the next wake should top up.
