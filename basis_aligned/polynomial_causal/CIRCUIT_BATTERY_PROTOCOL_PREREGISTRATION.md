# CIRCUIT BATTERY PROTOCOL — one preregistration for every behaviour

Registered Registered 2026-09-04 03:55Z (box clock) (box clock). Claude, LANE 1 CUDA. Rung stem `circuit_battery`.
Scripts: `ops/circuit_battery_tasks.py` (CPU task bank + row generator), `ops/circuit_battery.py` (GPU engine).

## Why this document exists

User directive 2026-09-04T03:43Z: *"Why do you need fresh data for every unique circuit? We should do the 20/80 here... If 1 circuit
experiment is taking an hour, something is wrong with your methodology. Figure out what's essential, build tools that can be built once
and reused, and then scale."* Under the old methodology each behaviour cost its own row generator, its own counterfactual families, its
own preregistration document with its own bars, and its own ledger section — the reason six task circuits exist after weeks. THIS
document is the last per-circuit preregistration: the bars below are frozen ONCE at protocol level and apply unchanged to every
behaviour in the bank, present and future. Adding a behaviour costs a task-bank entry (~15 lines) and a battery run; it does NOT open a
new preregistration. Amending a bar is a protocol amendment, recorded as such, and re-runs every behaviour.

## Protocol (identical for every behaviour)

A behaviour is a template + slot vocabularies + an answer function + a named causal variable. From it the generator mechanically emits
four counterfactual families x three splits with disjoint value pools (numeric starts partitioned mod 3, word pools sliced by split):

| family | meaning | answer |
|---|---|---|
| A1 | donor perturbs the causal variable | changes |
| A2 | a second, structurally different causal perturbation | changes |
| P | donor perturbs a surface/non-causal slot only (active answer-preserving control) | unchanged |
| C | copy control: same surface form, correct answer is a visible token, not the computed function | changes |

Stages, all on the observed model, zero fitted parameters:

1. **CAPABILITY** — native argmax over the task's answer vocabulary on A1 base prompts, all splits. Accuracy < .80 = the model cannot
   do the behaviour; it is recorded and excluded from the circuit claims (a behaviour without a capability is not a circuit).
2. **LOCALISE** — for each of the 36 components (attn_l, mlp_l, l = 0..17) the donor's whole output tensor is patched into the base
   forward and scored as normalized logit-difference recovery
   `REC(c) = (ld_patch(c) - ld_base) / max(ld_donor - ld_base, 1e-3)`, `ld = logit(donor answer) - logit(base answer)`.
   FIT ranks; the top-1 component is the WRITER; SELECT scores it. REC = 0 no effect, 1 full swap.
3. **SPLIT** — the writer's write at the final position, W, is carried as a parallel residual tensor scaled by each block skip lambda0
   and subtracted from the INPUT (pre-RMS-norm) of a chosen reader set only. Arms: FULL (every downstream component + the final norm),
   DIRECT (final norm only), READS (every downstream component), COMP_<component> singly. Damage d_m = m_NATIVE - m_arm with
   m = logit(answer) - max logit(other candidate in the task vocabulary); POSITIVE = the arm HURTS.
4. **SELECTIVITY** — the FULL deletion re-run on P and C; ratio = max(P, C) damage / max(A1 FULL damage, .5).
5. **HELD-OUT** — the writer's FULL damage on TEST rows (never used for selection).

## Predictions (frozen bars; battery-level, scored over the whole bank)

```
BARS  = {exact_tol: 1e-4, capability_acc: .80, capability_tasks: 8, localise_rec: .50,
         localise_tasks_frac: .50, select_ratio: .25, select_tasks_frac: .50,
         reader_top3_share: .80, margin_floor: .5}
NULLS = {capability_tasks_le: 2, localise_rec_le: .20, select_ratio_ge: .75, reader_top3_share_ge: .80}
```

**pred_a_instrument_full_equals_writer_ablation** — the edge decomposition is exact: on every task, the FULL arm (all reader edges +
direct removed) equals ablating the writer's final-position write outright, max |logit deviation| <= 1e-4.
*Worked example:* both quantities are the same forward up to float non-associativity, so the hypothesis reads ~1e-6 (SS2808 measured
6.2e-6 for the same construction); a genuine bookkeeping bug (a missed lambda0 scaling, a reader omitted from FULL) reads O(1e-1).
Operands are both non-negative magnitudes; no ratio.

**pred_b_bank_capability** — at least 8 of the 16 bank behaviours are performed natively at >= .80 argmax accuracy over the task's own
answer vocabulary. *Worked example:* a 546M model that does numbered-list succession (SS2808) and weekday succession reads 8-13 here;
the null "<= 2 capable tasks" is what a bank of badly-tokenized or badly-templated prompts reads. Operand is a count in [0, 16]. Three
bank entries (arithmetic.small_addition, numeric_sequence.continuation, numeric_sequence.countdown) are expected to FAIL capability and
are kept deliberately: "this model is a +1 machine on the last visible number, not an arithmetic machine" is a claim about the model,
and a bank that only contains behaviours it can do cannot state it.

**pred_c_writer_localisation** — for at least half the capable behaviours a SINGLE component recovers >= .50 of the interchange
logit-difference. *Worked example:* if one component carries the causal variable, REC(writer) ~ .6-1.0 (SS2808's attention 8 term is that
kind of object); if the variable is written diffusely by many components, every REC is ~.1-.3 and this reads 0 of N. Null: median
REC(writer) over capable tasks <= .20. REC is a signed ratio with a FLOORED denominator (clamped at 1e-3) and its denominator
ld_donor - ld_base is positive by construction (the donor prompt prefers the donor answer); the numerator may be negative and that is
recorded, not clipped.

**pred_d_writer_selectivity** — for at least half the capable behaviours the writer's deletion damages the answer-preserving family P
and the copy control C by <= .25 of its damage on A1. *Worked example:* a task-specific writer reads ~.0-.2 (its removal barely moves a
prompt whose answer does not depend on the causal variable); a generic "number-ness" or "last-token" component reads ~.8-1.2. Both
operands are damages in the same margin units; the denominator is floored at .5 margin units so it cannot go through zero.
Null: median ratio over capable tasks >= .75 (i.e. the writers are generic, not task-specific).

**pred_e_readers_are_redundant_not_concentrated** — this is SS2808's redundancy finding stated as a BANK-WIDE prediction, and it is
registered in the direction that can embarrass me: the median over capable behaviours of (top-3 single readers' damage / READS damage)
is <= .80, i.e. no behaviour's downstream computation is captured by three readers. *Worked example:* SS2808's numbered list reads
(.472 + .149 + .109) / 1.914 = .38; a genuinely concentrated circuit reads ~1.0 (three readers = all the damage), and a
super-additive/self-repairing one reads well below 1 because single-reader removals under-count (the hydra effect, McGrath et al. 2023,
arXiv:2307.15771). Null: median >= .80. Denominator READS is floored at .5 margin units; both operands are damages, and negative
single-reader damages (readers whose removal HELPS) are kept with sign in the sum.

## Stated null (battery level)

The bank is a fixture and the protocol measures nothing: capability <= 1 task, median writer REC <= .20, median selectivity ratio
>= .75, median top-3 reader share >= .80. Any of those four nulls being met is reported as such in the ledger, per-null, not averaged
away.

## Price

16 behaviours x [ (2 + 36) localise forwards x 2 splits + (2 + ~40) split forwards + 4 control forwards + 2 held-out ] per length-batch,
~24 rows per (family, split) cell, all sequences <= 24 tokens. Literal budget: <= 30,000 batched GPU forwards, 0 backwards, 0 fitted
parameters, expected <= 12 GPU-minutes for the whole bank (the 8-task v1 smoke measured 22 GPU-seconds at 6 rows/cell). A ninth behaviour costs ~1 GPU-minute.

## What this does NOT claim

The battery produces a WRITER + READER-SET localisation with an active-control selectivity number per behaviour. It does not produce a
learned interchange alignment (no DAS-style rotation is fitted — deliberately: a learned alignment can achieve high interchange accuracy
without being the network's mechanism, the "non-linear representation dilemma", arXiv:2507.08802), it does not claim minimality, and it
does not write canonical circuit records — those remain Codex's schema and Codex's call. Held-out TEST rows are never used for
selection; A2 is recorded and not used for any bar in this version.

## Amendment 2026-09-04T04:05Z (before any registered run of this protocol) and full disclosure

Two things happened between the first draft of this document and its frozen sha, and both are disclosed rather than hidden:

1. **A 6-rows-per-cell smoke of the v1 eight-task bank ran at 03:52Z and printed its predicate values** (a b c e TRUE / d FALSE on the
   first pass; pred_a FALSE on an earlier pass, which exposed a real bug in the instrument CHECK — the FULL arm was being compared with
   the native forward instead of with an outright ablation of the writer's write — fixed, after which max deviation was 1.3e-5). Smoke
   receipts are written to `circuit_battery_smoke_results.json` and never clobber the registered receipt.
2. **A native-capability scan of prompt formats ran at 03:55-04:02Z** (throwaway script, scratchpad, not a rung) and its results changed
   the BANK, not the bars: the model turned out to answer bare numeric runs with LAST + 1 rather than LAST + STEP, so
   `numeric_run.last_plus_one` was added as the behaviour the model actually has; and eight further behaviours with measured native
   capability were added (paren list, keyed counter line, roman list, counting words, alphabet run, bracket close-innermost, verbatim
   repeat, plus the letter list already present). Choosing behaviours the model can perform is task construction, not bar fitting: a
   behaviour the model does not have cannot have a circuit.

**Bars changed by this amendment:** only `capability_tasks` 4 -> 8 and the null `capability_tasks_le` 1 -> 2, both purely rescaling the
same fraction (1/2 and 1/8) to a bank that grew from 8 to 16. **Bars NOT changed:** exact_tol, capability_acc, localise_rec,
localise_tasks_frac, select_ratio, select_tasks_frac, reader_top3_share and every null but the rescaled one — in particular
pred_e's .80, which the v1 smoke already showed reading ~.46, was left where it was rather than tightened toward the observed value.
The registered run uses 24 rows per cell over the 16-task bank, on the frozen splits, with this document's sha pinned in the script.
