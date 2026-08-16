# Response to Reviewer 2

An independent adversarial review was run over Part A, B2 and `BILIN18_CONNECTION.md`.
It found real errors. This file triages every finding, states the disposition, and records
what was done. `RESULTS.md` and `BILIN18_CONNECTION.md` are amended in place; this file is
the audit trail.

Summary: **3 retractions, 6 substantive corrections, 5 reproducibility fixes, 2 partial
disagreements.** The reviewer also independently found the same gauge bug that
`THEORY.md` T8 derives, from the other direction.

---

## Retracted

### R1. "Grokking is the decay of the memorisation term" — WRONG, retracted

Claimed in A2-4. The reviewer measured that the residual does not decay. I re-measured
independently and confirm it:

| step | circuit logit rms | residual logit rms | off-block fraction | functional residual |
|---|---|---|---|---|
| 3000 | 0.905 | 1.506 | 0.564 | 0.708 |
| 12000 | 1.843 | 1.850 | 0.322 | 0.490 |
| 21000 | 2.589 | **2.343** | 0.298 | 0.441 |
| 39000 | 2.780 | 1.929 | 0.179 | 0.320 |

The memorisation term **grows** from 1.51 to a peak of 2.34, then declines to 1.93 — ending
28% *above* where it started. The circuit grows 3.1× over the same span. The falling
off-block fraction is the numerator growing, not the denominator shrinking.

**Corrected statement:** grokking here is dominated by the growth of the generalising
circuit against a memorisation term that rises and then partially recedes. The residual
does eventually decline ~18% from its peak, so "no decay at all" would also be wrong, but
decay is not the mechanism.

### R2. "The circuit is fully formed at step 1500" — overstated, retracted

At step 1500 the block-projected part accounts for only **25% of the function**
(functional residual 0.749) and is **0.80 aligned** with its own final form in the Λ metric —
and that alignment wanders (0.835, 0.848, 0.817, …) before converging. Test accuracy 1.000
is an argmax statistic on 317 held-out pairs; projected CE is 1.130 versus 0.0006 at the end.

**Corrected statement:** from step ~1500 a 44-dimensional projection carrying a quarter of
the function already gets every held-out pair's *ordering* right. The ranking is correct
long before the magnitudes or the circuit's final shape are. That is still a real and
surprising result — the model itself is at 0.000 — but it is not "the circuit is finished".

Related: "complete 26× before the model uses it" is withdrawn. The model never crosses the
0.999 threshold (`first_step_model_generalises` is `null`), so 26 = 39000/1500 was a ratio
to the last logged step, not to a measured event.

### R3. B2-3's ablation conclusion — retracted, instrument is broken

The ablation replaced a factor with `s[which].mean(-1)`, described as "a constant that
preserves its scale". It is the per-example mean over keys: measured mean −0.0056, and
**negative on 55.9% of examples**. A negative multiplier inverts the attention ordering.

The falsifying evidence was in my own JSON and not in my table: on the *control* tasks,
where a single property suffices and the ceiling is 1.000, the same ablation still drops the
head to 0.27 — indistinguishable from the conjunctive 0.265. A surviving factor that can
demonstrably score 1.000 alone cannot be "doing nothing that matters".

**Retracted:** "the score-level placement is one circuit wearing two hats", and test #4 in
`BILIN18_CONNECTION.md` as motivated. The B2-2 finding (entropy is non-specific) is
independent of this and stands — indeed T8 strengthens it.

---

## Corrections made

### C1. The B1 entropy statistic is gauge-dependent (reviewer §3)

`(W₁, W₂) → (cW₁, W₂/c)` is exactly function-preserving, and softmax entropy is not
invariant under it. The reviewer demonstrated a flip; `THEORY.md` T8 proves it and verifies
that the participation ratio *is* invariant. The entropy numbers for the score-level
placement are withdrawn from B2-2; the conclusion is unchanged and now has a stronger
reason. B2's null 2 only applied `Wq → MWq, Wk → M⁻ᵀWk`, which preserves each `W_i`
exactly — it never tested the gauge that mattered, and its 7-significant-figure agreement
should have been the tell.

Post-softmax has no such gauge (each factor is separately normalised), so B2-4's
specialisation result is unaffected.

### C2. A2-3's "29–35% unidentifiable" needed its baseline (reviewer §7)

The identifiable subspace is 529 of 1081 dimensions, so **isotropic chance is 48.9%**.
Measured: random symmetric 0.4891, random-init layers 0.479/0.500/0.488, trained
0.712/0.646/0.705. The trained model is ~1.4× *more* data-aligned than chance. The heading
read as an indictment; the honest framing is that the trained model concentrates 65–71% of
its mass in a subspace occupying 49% of the space. Corrected in `RESULTS.md` and in
`BILIN18_CONNECTION.md` §2.2.

Also noted: `canonicalise` projects in the Frobenius metric while the program's own rule
says never to use raw Frobenius. The number is metric-dependent. Flagged, not yet fixed.

### C3. A4-3 relabelled from measurement to theorem (reviewer §5)

`err_lin/err_prune = 1/(1+ρ²)²` in A4's design, and the gain cancels identically in general.
Proved and verified in `THEORY.md` T3. The `+0.000` correlation was forced by construction.

I accept the framing criticism too: the plan's actual prediction — that the band moves with
the tolerance — was *confirmed* by `band_vs_budget`, and I substituted a different statistic
before declaring the plan wrong. "What Part A changes about the plan" is amended: the plan's
stated prediction held; what is corrected is the *reason*, and the (still useful) point that
ranking by size answers a different question.

### C4. A5's 0.8374 was a metric mismatch, not a measurement (reviewer §8)

Forms orthonormalised in Frobenius, scored with `lam_cos`. The analytic value is
√(3/4) = 0.8660 in Frobenius; 0.8374 is the Λ-metric cosine of the same pair. I claimed
they "match to three decimals" — they do not, and the gap is entirely the metric mix.
Corrected. Also accepted: in the Λ metric the planted forms are *not* orthogonal
(off-diagonals up to 20% of the diagonal), so the docstring's stated reason for
orthonormalising is not achieved in the metric the program says to use.

### C5. A5-2's generalisation is too strong (reviewer §8)

"The reader shuffle is the wrong instrument for any claim about sharing" holds at R=3
(4.00 → 3.50–4.00) but **breaks at R=8** (9.00 → 4.95, a 45% collapse) — in the same JSON.
Narrowed to: the shuffle is insensitive at small R and only partially sensitive as R grows;
the matched no-sharing control is the instrument to use either way. And per `THEORY.md` T4
the R+1 identity itself requires equal gains, which a real model will not have.

### C6. A3's axis was not resolved (reviewer §9)

A3 varied only `R` at fixed `m=8, d=16`, so its breakdown is equally consistent with a limit
in `R/d` and one in `R/m`. The form family's effective rank is exactly `m = 8` at every K.
`THEORY.md` T6 gives the Kruskal bound `R ≤ (m + 2d − 2)/2` and shows the binding mode is
`k_C ≤ m`. **`BILIN18_CONNECTION.md` §2.1 is rewritten**: the conclusion survives (bilin18's
4608 ≫ the 1727 Kruskal bound at `m = d = 1152`) but the previous "twice the point where
identifiability fails" argument extrapolated along an axis A3 cannot resolve.

---

## Reproducibility fixes

The reviewer verified by hand that every unreproducible number is *correct*, but several
headline numbers existed only in `RESULTS.md`, produced by ad-hoc inline commands:

| numbers | status |
|---|---|
| A2-4's residual-only table (train 0.95 / test 0.00 / logit boost ±) | now in `a2_followups.py` |
| A2-8's six null numbers | now in `a2_followups.py` |
| the symmetry-preserving 7/11 vs symmetry-breaking 3/11 split | now in `a2_followups.py` |
| A2-7's 200k/260k long-run columns | **withdrawn** from the table pending a re-run; the claim "crystallisation arrests" now rests only on 40k |
| `a2_symmetry.json`, `a2_residual_scale*.json` written by no committed script | scripts added |

Also fixed: `RESULTS.md:391` called 0.21 the *raw* off-block while the table header says
canonicalised (raw at 40k is 0.2226).

---

## Partially disputed

### D1. "The keep/prune/linearize family strictly contains keep/prune, so winning is near-tautological" (reviewer §5)

Correct in the surrogate, and I have added the caveat. But it is not empty: the comparison
is made on *re-measured* error, where additivity fails by up to 33%, so a larger family can
and sometimes does lose — and it does, at 3 of 25 budgets. The informative content is the
magnitude, which the reviewer also says; A4-5's random-init baseline (23/25 at 9.3×) is the
honest control and is already reported.

### D2. "A4-1's 99.995% kernel capture is vacuous against a 99.49% chance baseline" (reviewer §5)

Accepted as reported — five significant figures against an unstated baseline is bad
practice, and only the 8 diagonal directions were tested. But the claim is not vacuous: it
is *exact* in the noiseless case by Proposition 1, and the measurement was a check that the
trained model realises it. Restated with the baseline and the coverage limit.

---

## Accepted without dispute, fixed in code

- `sbd_robust`'s draw selection maximises block count before mass, contradicting its comment.
- `bq_sanity.py` check 4b passes unconditionally (a tautological second disjunct), and none
  of `canonicalise`, `jade`, `partition_from_coupling`, `fit_cp`, `fit_dictionary` has a
  sanity check. "12/12 pass, run it before trusting anything below" was oversold.
- A2-6's splice control uses a full `O(4)` rotation, which destroys block-internal structure
  rather than just phase; "same energy, same frequencies, destroyed phase alignment"
  overstates it. A phase control would be `R ⊕ R`.
- A2-6's "any 3 frequencies can be deleted" tested exactly one random 3-subset.
- A1 reports cosines as `min` in one place and `max` in another; nulls should be reported at
  their max (null 3's true max is 0.41, not 0.18) and null 1's signed max hides a −0.249.
- A1's `planted_mass`/`diag_mass` are L1 fractions called "mass"; `lift_kernel_pairs` builds
  off-manifold inputs so the 36–51% figure is off-distribution and depends on `t = 1.5`.
- A1-1's "3 seeds" header mixes a pure-arm number into graded-arm columns.
- A3-3's "form error 0.000 / still exactly the teacher's" hides monotone growth
  1.24e-5 → 5.33e-5 with noise; A3 tables are single-seed under a "2 seeds" header.
- A4-2's "ties at the two endpoints" — there are three ties, one interior (budget 1759).
- A4-1's "√FVU" arithmetic is off by 1.78×.
- B2-4 tabulates `postsoft/conjunctive/s1` as specialised when its JSON shows
  `type_diag ≈ prop_diag ≈ 0.33` for *both* factors, i.e. no specialisation; and
  `factor_readout` compares raw block masses across blocks of different dimension, biasing
  `best` toward the 16-dim payload block.
- B2's DGP uses a second *content* categorical, not the spec's "position-parity or timing"
  feature — the easier, positionally trivial task, which is part of why B2-1 was inevitable.
- Null 3 shuffles labels within each fresh batch, so nothing is memorisable — weaker than
  the spec's "train to matched loss".
- `a4_nulls` counts wins on the surrogate while `a4_quotient` counts on re-measured error,
  so A4-5's "22/25 vs 23/25" compares two different statistics.
- Stale artefacts: `report.html` still says A5/Part B not started; the Queue section listed
  finished work as in flight.

---

## What the reviewer confirmed

Worth recording, since it bounds what the retractions cost:

- `partition_from_coupling`'s binary search is valid — in-block mass is genuinely monotone
  in the threshold.
- `gauge_refactor` is exactly function-preserving and genuinely destroys the hidden basis.
- A2-4's relabelled-group Fourier control is tight (same 44-dim span, different partition)
  and fails as it should — it, not the random-blocks control, is the load-bearing one.
- A2-6's dose-response against equal-dimension random subspaces is fair and clean.
- A1's kernel and blindness machinery is correct and the graded-arm numbers reproduce.
- Every number checked against the JSONs matched to printed precision: **no fabrication,
  only selection and framing.**
- A4-5 and B2-1 are the program reporting against its own interest, and both are correct.
- The bilin18 architecture audit and the argument that B1's entropy census is undefined on
  an unnormalised signed pattern are sound.
