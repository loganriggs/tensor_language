# Frontier: does the adopted correction transport? A held-out fresh window. Preregistration

Registered 2026-09-04T13:05Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A **gain** here is
`L2(baseline) − L2(arm)`, **POSITIVE = BETTER**. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — this rescales already-fitted
objects; it neither selects nor reorders.

## Why — the threat to the only two adopted results

**Every scalar this campaign has adopted was chosen by minimising L2 on the same 120 documents it is reported on.** `FR` is built once,
deterministically, from `dsf[di]` over `di in range(3000,10000)`, and `L2_F = evalM(FR, …) − baseF`:

- §2896 chose tail `LW` × 0.25 by comparing L2_F.
- §2902 chose CP `Dk` × 0.5 by comparing L2_F.
- §2904 reported the composition as **+2.3522, a 0.3213 improvement on +2.6735** — on FR.
- §2907 and §2909 then searched 36 and 32 grid cells against FR.

That is **selection on the evaluation set**. Some of the reported 0.3213 is a real property of the fitted stack; some is fitting the
particular 120 documents that scored it. **The size of that second part has never been measured**, and nothing published so far can
separate them. It is the single largest threat to the only two adopted frontier results, and it is my job to test it.

The test is cheap: continue the **same** document scan to 240 rows. `rows[:120]` is bit-identical to the FR every previous rung used
(same order, same dedup), so the reproduction gate still tests the published number; `rows[120:240]` is a second 120-document window
that **played no part in choosing any scalar**. Evaluate the baseline and the frozen, already-adopted (tail .25, CP .5) configuration on
both. No fitting, no grid, no selection — **two arms**.

**The quantity this rung exists to produce is the selection bias, `gain(FR) − gain(FR2)`, in nats.** It is reported whether the
predicates hold or fail: a small bias vindicates the adoptions, a large one qualifies them, and either way the number belongs on the
record next to +2.3522.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* this family reads
  +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the adopted gain reproduces on the selection window.** `gain_FR ≥ 0.25`. *Worked example:* §2904 measured **+0.3213**, so
  ≈ .32. **This predicate exists to pin the sign of pred_c's reference quantity** — per my standing rule, never register `X ≥ k·Y`
  where `Y` may be ≤ 0. `pred_c` is coded to require `pred_b` first.
- **pred_c — at least three quarters of the gain transports.** `gain_FR2 ≥ 0.75 × gain_FR`. *Worked example:* if the scalars capture a
  real property of the fitted stack, `gain_FR2 ≈ +0.32` against a threshold of ≈ +0.24 and this holds; if the gain were entirely an
  artefact of choosing on FR, `gain_FR2 ≈ 0.00` and it fails outright. **0.75 is a bar on a two-parameter selection over ~30 evaluated
  cells; a mild shrinkage is expected and tolerated, a collapse is not.**
- **pred_d — the held-out window is a comparable comparator.** `|L2_F2(baseline) − L2_F(baseline)| ≤ 0.50`. *Worked example:* two
  120-doc draws from the same scan should sit within a few hundredths to a couple of tenths; a bigger offset means FR2 is a harder or
  easier corpus and the gains are not being compared like for like — **then pred_c is uninterpretable and must not be reported as a
  transport result.**
- **pred_e — the gain does not change sign off the selection window.** `gain_FR2 > 0`. *Worked example:* the weakest possible transport
  claim, separating "smaller but real" from "the correction is specific to the documents that chose it". Failing this while pred_a and
  pred_d hold would be a **serious** finding requiring §2896/§2902/§2904 to be qualified on the record.

## Nulls

- `b_null_the_adopted_gain_does_not_reproduce` (< 0.25) — nothing downstream readable.
- `c_null_the_gain_is_an_artefact_of_the_selection_window` (`gain_FR2 ≤ 0.5 × gain_FR`) — half the gain or worse is selection.
- `e_null_the_correction_is_window_specific` (`gain_FR2 ≤ 0`).

**What I will do with each outcome, stated in advance.** pred_c holds ⇒ the adopted +2.3522 stands and the ledger gains a measured
selection bias beside it. pred_c fails but pred_e holds ⇒ **the adoptions stand but the headline number is restated as the held-out
value**, with the bias quoted. pred_e fails with pred_a and pred_d holding ⇒ I record it as a **negative result against my own two
adopted rungs**, and per the standing rule a conclusion-flipping correction gets an independent physical control before anything is
withdrawn. **No outcome here is a reason to search for a better scalar on FR2** — that would repeat the exact error under test.

## Price

**1 full frontier pipeline run + 2 extra forward evaluations, ≤ 400 GPU-seconds** (§2909's 32-arm run took 201.7 s; this has 2 arms but
one extra 120-doc window and one extra `classify2` pass), 0 backwards, **0 fitted parameters** — the scalars are read frozen from
§2904's receipt. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not forward-instrumented**, so the receipt reports
`gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`. Receipt: `frontier_scale_holdout_results.json`, read with
`price` in the same command the ledger section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` /
`Results: <file>.json` form (§2853, §2858), under a filename no other section cites (§2876).
