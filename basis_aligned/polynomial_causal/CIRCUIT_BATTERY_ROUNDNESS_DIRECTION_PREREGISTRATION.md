# CIRCUIT BATTERY — ROUNDNESS DIRECTION (preregistration)

Registered 2026-09-04 06:19Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_roundness_direction`. Script: `ops/circuit_battery_roundness_direction.py`.
Input receipt: `circuit_battery_roundness_head_split_results.json` (§2843, sha 5e076b400295c62d9936bb69f433d207d03111e979253d677a0ebb187109d89e).
IMMUTABLE: any change gets a new document, not an edit.

## Object

§2842 and §2843 located the step-versus-plus-one switch in attention 8 and then in heads {3, 7} — the same pair that writes the last
item's identity (§2820, Codex's R576) — with head 3 leading both formats and the pair holding .925 of the positive head recovery. That
says WHERE the feature is, not WHAT it is. §2826 asked the analogous question on the reader side and got the campaign's first positive
sub-block result: one unfitted direction carried a fifth of the damage at 2.4× the block's specificity while holding .0021 of the
energy. The write side has never been asked.

Here the per-row difference between the donor's and the base's head-3 output is the roundness delta. A single MEAN direction is fitted
on half the pairs, and the switch is then driven by projecting each held-out row's delta onto that one fixed direction. If it works, the
feature is one vector in a 128-dimensional head output and is compilable; if it does not, roundness is carried by a row-dependent
pattern and §2843's head localisation is as far as this instrument goes.

Sign convention: `ld = logit(plus-one) − logit(step)`, `REC = (ld_patch − ld_base)/max(ld_donor − ld_base, 1e-3)`; 0 = no effect,
1 = full switch. **No CE, no §312 L2, nothing installs. 128 fitted parameters per format, fitted on the fit half only.**

## Predictions

```
BARS  = {heldout_frac: .50, random_rec: .10, transport_cos: .50, bulk_cos: .80, exact_tol: .02}
NULLS = {heldout_frac_le: .15, random_rec_ge: .40, transport_cos_le: .10}
```

**pred_a_one_fitted_direction_carries_the_switch** — median over the two formats of
`REC(fitted direction) / REC(whole head 3 swapped)` ≥ .50 on HELD-OUT pairs. *Worked example:* §2843 measured head 3's whole-slice
recovery at .450 (percent) and .311 (bare); if the roundness feature is one direction, the projection recovers .2–.4 of the switch and
therefore .5–.9 of the head's own effect. If roundness is a row-dependent pattern, the fitted mean direction transports poorly and this
reads .0–.2. A ratio of two recoveries in the same units — and the denominator is a measured, non-degenerate quantity (§2843 reported
both values), not something that can approach zero. Null: ≤ .15.

**pred_b_random_direction_is_inert** — median over formats of `REC(random unit direction in the head's 128 dims)` ≤ .10.
*Worked example:* a random direction should capture ≈ 1/128 of the delta and essentially none of the switch, ~.00–.03. This is the
control that keeps pred_a from being satisfied by "projecting onto anything works". Seeded (2843). Null: ≥ .40.

**pred_c_the_direction_transports_across_formats** — |cos| between the direction fitted on the percent pairs and the one fitted on the
bare pairs ≥ .50. *Worked example:* roundness is a property of the number rather than the "%" surface, so a genuine feature direction
agrees across formats at .6–.95; two random directions in R^128 sit at |cos| ≈ .09. This is the clause that distinguishes a feature from
a format-specific artefact. Null: ≤ .10.

**pred_d_it_is_not_the_bulk_output** — median over formats of |cos(fitted direction, mean head-3 output)| ≤ .80.
*Worked example:* §2835 found attention 5's dominant direction IS its mean write (|cos| .9999996), so "the feature is just the bulk" is
a live possibility that has already occurred once in this campaign. If roundness is a genuine feature rather than the head's bulk
output, .0–.5. If it reads ≥ .95 then head 3's roundness delta points where head 3 always points and the "feature" is a magnitude
change, which would be a different and simpler mechanism — worth knowing either way.

**pred_e_full_delta_reproduces_the_head_patch** — head 3's whole-slice recovery is non-zero in both formats, i.e. the arm this rung
normalises against is live. *Worked example:* §2843 measured .450 and .311; a zero here would mean the hook-based head swap is not
reproducing that rung and no ratio in this document could be read. Instrument check.

## Stated null

The fitted direction does not transport (≤ .15 of the head's effect), a random direction does as well, or the directions disagree across
formats. Then roundness is not a single direction in head 3's output and §2843's head localisation is the finest statement this
instrument supports — which, given §2822–§2824 found exactly that on the reader side, would be the unsurprising outcome and is registered
as such.

## Price

2 formats × (≈12 fit pairs + ≈12 held-out pairs) × (2 native + 3 patched) forwards, batched by token length.
Literal budget: ≤ 300 GPU forwards, 0 backwards, **256 declared fitted parameters** (one 128-dimensional direction per format).
< 60 GPU-seconds.

## What this does NOT claim

One head (3) of one component, one mean direction, no rank > 1 and no per-row adaptivity. No selectivity control exists in a
two-behaviour minimal pair. One step size and one digit range, inherited from §2841. Pairs are §2842's construction, not the bank's
frozen splits. Does not satisfy Codex's four-phase integration contract; updates no circuit record.
