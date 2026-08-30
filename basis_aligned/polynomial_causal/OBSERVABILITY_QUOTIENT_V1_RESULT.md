# Observability quotient v1 — result (self-reviewed)

**Run:** `observability_gramian_v1.py`, 94 s on the RTX 5090, 2026-08-30 17:38 UTC, via `bqrunner`.
Preregistration: `OBSERVABILITY_QUOTIENT_V1_PREREGISTRATION.md` (frozen before any number).
Artifact: `observability_gramian_v1_results.json`. First run died on a shape bug (perturbation
tensor spanned 256 of 257 positions); its exit=1 is preserved in lane 1's `runlogs/`.
**pred_a FAILED | pred_b FAILED | pred_c FAILED.**

## Numbers

| site | r50 / r90 / r99 of the loss-gradient Gramian | r50 / r90 / r99 of activation covariance | A→B transfer of the r90 subspace | top-8 overlap with lm_head rows |
|---|---|---|---|---|
| block 2 | 142 / **737** / 1092 | 1 / 264 / 871 | 0.827 | 0.086 |
| block 5 | 70 / **712** / 1090 | 54 / 521 / 1024 | 0.865 | 0.094 |
| block 9 | 176 / **816** / 1108 | 49 / 586 / 1053 | 0.838 | 0.125 |

Causal test (mean ΔCE over 64 fresh rows, 8 draws; perturbation norm = rel × mean stream norm):

| site | rel | observable r90 subspace | orthogonal complement | random r90-dim subspace | obs / cmp |
|---|---|---|---|---|---|
| 2 | 0.5 | +0.0145 | +0.0062 | +0.0104 | 2.35 |
| 2 | 1.0 | +0.341 | +0.097 | +0.227 | 3.50 |
| 5 | 0.5 | +0.067 | +0.039 | +0.056 | 1.71 |
| 5 | 1.0 | +0.509 | +0.298 | +0.425 | 1.71 |
| 9 | 0.5 | +1.074 | +0.897 | +1.017 | 1.20 |
| 9 | 1.0 | +3.217 | +2.994 | +3.134 | 1.07 |

## Reading

1. **The first-order observable subspace is not small.** The loss is sensitive, at 90 % of gradient
   energy, to 64–71 % of the stream's dimensions at every site tested. The comparison I registered
   (against the activation covariance's r90) was the wrong denominator — activation covariance at
   block 2 has r50 = 1, a single massive direction — but the absolute statement stands against D:
   "factor only the quotient" buys at most a third of the stream at first order. pred_a fails on the
   size clause; its stability clause holds (0.83–0.86 transfer), so what is small-ish is a fixed
   object, just not small.
2. **Direction matters less than magnitude, and more so with depth.** Observable directions cost
   2.3–3.5× the complement at block 2 and only 1.1–1.2× at block 9; a random subspace of the same
   dimension costs nearly as much as the observable one everywhere because the observable one *is*
   most of the space. pred_b and pred_c fail for the same reason.
3. **The stream's price of error rises steeply with depth.** A relative-norm-0.5 perturbation costs
   0.015 nat at block 2, 0.067 at block 5, **1.07 at block 9**; at norm 1.0, 0.34 / 0.51 / 3.22. Lane
   1's assembly carries rel-MSE 1.74 (relative norm ≈ 1.3) at block 6 (§2086). Read against this
   table, a mid-stream error of that size is worth on the order of the whole +2.9-nat frontier gap
   on its own — the "downstream repair" of §2086–§2088 is the model attenuating an error that would
   otherwise cost far more, not an indication that the error is cheap.

## What this closes and what it opens

- Closed: a **linear, first-order, site-local** observability quotient as the object to factor. It
  is document-stable but it is two-thirds of the stream.
- Open, and now better posed: the quotient that matters is the one **relative to the program's own
  error** — which directions a compressed early program actually gets wrong, projected on what
  downstream reads — and the depth at which error is cheapest (block 2 tolerates relative norm 1.0
  for 0.34 nat; block 9 does not tolerate 0.5). The companion depth profile
  (`observability_depth_profile_v1.py`, queued) maps the second; the first needs the assembly's
  error covariance, which lane 1 can export from its §2086 diagnostics.
- The strict ledger is unchanged: 5.348 % / 10.923 % / 4.727 nat / 0 of 68.

## Addendum — depth profile at all 18 sites (`observability_depth_profile_v1.py`, 14 s, third attempt)

Registered against the measured block-2 value (737) after two instrument failures (a per-block
detach that cut the graph; `retain_grad` on a no-grad leaf), both preserved in `runlogs/`.
**pred_b HELD (ρ = +0.52) | pred_a FAILED | pred_c FAILED (one site).**

| site | r90 obs | r90 act | ratio | gradient trace | A→B transfer |
|---|---:|---:|---:|---:|---:|
| 0 | 736 | 546 | 1.35 | 6.1e-1 | 0.836 |
| 1 | 677 | 425 | 1.59 | 1.6e-7 | **0.777** |
| 2 | 737 | 296 | 2.49 | 1.4e-5 | 0.827 |
| 3 | 732 | 472 | 1.55 | 5.7e-7 | 0.838 |
| 4 | 712 | 483 | 1.47 | 3.5e-7 | 0.854 |
| 5 | 712 | 528 | 1.35 | 5.5e-7 | 0.865 |
| 6 | 752 | 505 | 1.49 | **4.6e-5** | 0.862 |
| 7 | 825 | 515 | 1.60 | **7.6e-5** | 0.844 |
| 8 | 821 | 562 | 1.46 | 6.9e-5 | 0.841 |
| 9 | 816 | 595 | 1.37 | 2.3e-5 | 0.838 |
| 10–16 | 826→771 | 619→669 | 1.33→1.15 | 1.9e-5→2.6e-6 | 0.82–0.83 |
| 17 | 739 | 544 | 1.36 | 1.7e-6 | 0.837 |

The first-order observable subspace is **59–72 % of the stream at every depth** — it never
shrinks toward the readout (site 17: 739, pred_a fails outright), it is always larger than the
activation covariance's r90, and it transfers at 0.78–0.87. The linear quotient is closed at every
site, not just three. The one number here that carries the cliff is the **gradient trace**: the
loss's total first-order sensitivity to the stream jumps two orders of magnitude between block 5
(5.5e-7) and blocks 6–8 (4.6e-5 to 7.6e-5), the same place `STREAM_ERROR_PRICE_V1_RESULT.md`
finds the 25× price jump. (Site 0's 0.61 is the un-massive embedding stream, norm ≈ 34 against
5,000–32,000 later; trace is not norm-normalised.)
