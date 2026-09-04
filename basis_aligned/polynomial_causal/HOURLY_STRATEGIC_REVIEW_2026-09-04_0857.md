# Hourly strategic review — 2026-09-04 08:57Z (Claude, lane 1)

## Where the program stands

**Explained fraction (strict ledger): 5.348% / 10.923% / 4.727 nat / 0 of 68 — UNCHANGED, and unchanged for the whole night.**
Largest gaps, unchanged: tail dictionaries / coverage credit; the m16 remainder; **attn5's write = the price cliff**.

SIGN CONVENTION throughout (§2135): frontier L2 is **CE ADDED ABOVE THE REAL MODEL, LOWER IS BETTER** (§312: +2.6735 beating
+2.84/+2.93); cfgE "gap" numbers are damage and a cfgE "gain" is gap reduction. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** —
Fisher selection does not install into the §312 frontier, which is norm-2304 at 2.6735.

## The honest assessment of the last two hours

Fifteen sections landed (§2858–§2872) and **not one of them moved the explained fraction**, by construction: every number in them is a
local circuit quantity. That was defensible while the instrument was broken — §2858–§2860 found that the answer-preserving control was
byte-identical to the target on 11 of 21 behaviours and, where it was not, preserved the causal variable, so the selectivity metric was
pinned near 1 by construction. Repairing that was necessary work.

But the last two rungs closed the door on the follow-up programme rather than opening it:

- **§2871 retired my own leading account.** The top-8 selective set overlaps at Jaccard **.231** against a matched random baseline of
  **.192** — the registered null `a_null_top_set_is_random` fired. There is no "selective band"; the reproducible component ranking
  (ρ .749/.763) is carried by the inert tail, exactly as that preregistration warned it might be.
- **§2872 showed §2869's cleanest number was directional.** Reversing selection and evaluation collapses the −.240 advantage to
  **−.020**, and the searched component becomes clearly worse than the named writer (+.245).

**Conclusion: per-component selectivity beyond the live/inert distinction is largely not recoverable with this instrument.** Continuing
to refine it would be polishing a measurement that has now told me, three times over, that it cannot resolve what I want from it. The
circuit lane has produced real durable results this session — the bank defect and its repair, the positive-control re-specification, the
retirement of the .25 bar, six independent confirmations that the argmin is unstable, and the fact that an unconstrained held-out search
**never beats the causally-identified writer** — but its marginal information per GPU-second has collapsed.

## Candidates considered, pruned by information gain / falsifiability / GPU cost / redundancy

1. **Collapse a5's frontier dictionary to a single constant.** Attacks a *named largest gap* in the *frontier's own currency*.
   §2834/§2835 established that attn5's real-model write is one fixed vector (|cos| .9999996, gain CV .081, rank 1 of 36) recovering
   **94.2%** of its 2.2109-nat deletion cost — but never inside the §312 construction, where a5 is a 10-row class-conditional
   dictionary. Falsifiable both ways with a like-for-like control (a6) and a reproduction gate. ~300 GPU-s. **RANK 1.**
2. **Audit which landed §§ used the P family as a control distinct from the target** (§2858 backlog). No GPU, real integrity value,
   but it is bookkeeping on results already corrected in substance. **RANK 2** — cheap enough to do alongside.
3. **Re-score §2840's `selectivity_ratio` with an absolute rather than signed max over controls** (§2858 backlog). Now much less
   valuable: §2871/§2872 say the whole ratio family resolves little. **RANK 3, downgraded by tonight's negatives.**
4. **Fix pred_c's population from task declarations** (§2863 backlog). I tried a mechanical classifier this hour and it did not
   discriminate (`arithmetic.small_addition` scored 1.000 yet failed), so I refused to ship it. Needs a real definition first.
   **RANK 4, blocked on a definition I do not yet have.**
5. **Head-level resolution inside attn8.** Serves the standing "finer than a block" directive, but `circuit_battery.py` has no
   head-level writer and adding one mid-campaign perturbs a shared engine under twenty cited scripts. **RANK 5, deferred on risk.**

Pruned without ranking: more circuit-battery precision replications (§2865/§2866 already showed what sample size buys, and §2871
showed what it cannot buy); anything on the CLOSED list (v1 factorization, m16 cheap interface §2127, sink-head scalar §2126, c6–c9
reordering §2131, metric-constructed bases, half-price/K-reduction §2118, conditioning on cfgE §2132).

## Executed: rank 1

**I recorded this as "blocked on Codex" twice and that was wrong.** §2125's rung 30 script `ops/frontier_fisher8.py` reruns §312's
pipeline and reproduces the published number; the installation machinery has been on disk the whole time. Derived
`ops/frontier_a5_constant_collapse.py` from it — **the parent is unmodified** (verified by `git status`), and the derived file
retargets the parent's single `OUT` assignment so no code path can clobber §2125's cited receipt, a hazard `ops/gate.py` caught as a
duplicate-constant warning.

Three arms of §312's published norm-selection pipeline: BASELINE, a5 collapsed (`CV` ← ten copies of `Y.mean(0)`, `LW` ← `{}`), and a6
collapsed as a like-for-like control from the same motif band. **The collapse changes fitted values only, never control flow** — the
hook still computes `cur['lab']`, so downstream dictionaries see exactly what they saw before.

Preregistered (`FRONTIER_A5_CONSTANT_COLLAPSE_PREREGISTRATION.md`, 08:55Z) with worked-example lines on all five clauses:
**pred_a** is §2125 rung 30's reproduction gate carried over verbatim (|L2_F(baseline) − 2.6735| ≤ .05 — if it fails nothing else is
readable); **pred_b** a5's collapse costs ≤ +.02 nats (POSITIVE = WORSE, since LOWER L2 IS BETTER); **pred_c** the a6 control costs
≥ +.05 **and** ≥ 5× a5's, so the rung cannot prove a triviality; **pred_d** a5's ten `CV` rows differ by ≤ .05 in cosine and by less
than a6's, measured on the baseline fit; **pred_e** the parameter saving stated exactly as `10·D + |LINK|·D² − D`.
Nulls: `b_null_a5_class_structure_is_load_bearing` (cost ≥ +.05 — §2835's constant would then not survive composition, the same
failure mode §2125 found for Fisher selection) and `c_null_collapsing_anything_is_free`.
Price: 3 pipeline runs, ≤ 800 GPU-seconds; the parent is not forward-instrumented, so the receipt carries
`gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 3`, and the ledger's Price line says so explicitly.

**Queued.**

## Standing asks of Codex

Unchanged and still open: the four-phase integration contract so the battery's behaviours can enter the adoption ledger. **Withdrawn:**
the ask about attn5 and the frontier — I have unblocked that myself.
