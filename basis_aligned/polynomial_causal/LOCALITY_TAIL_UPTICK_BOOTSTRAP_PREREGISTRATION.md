# Preregistration — is the §2640 locality-tail uptick real or estimator noise? (CPU companion; parallel lane)

Date: 2026-09-02 23:58 UTC
Owner: Claude (parallel probe lane)
Status: frozen before any bootstrap outcome; zero model forwards, CPU only.

## Question

§2640 measured f(1/128) > f(1/64) in all four branch×half cells (aligned
recovered fraction on edited positions; higher = more restored), breaking
monotonicity at the locality curve's tail. I preserved two un-promoted
readings: (a) small-mask estimator bias — the ratio f = <recovery,x>/||x||²
computed on ~490 sparse positions is high-variance; (b) a real isolated-edit
advantage. This resolves them by document-level bootstrap on §2640's OWN frozen
per-token bundle. No new forwards; the bundle already holds per-token CE for
native, both branch-absent trajectories, and all six density arms plus the six
nested masks.

## Computation (exact, deterministic seed)

Load the §2640 bundle (native, absent[2], arms[2,6], masks{6}). For each branch
b ∈ {T,I} and half h ∈ {0:250, 250:500}, and the two tail densities
d ∈ {1/128, 1/64}: x = absent_b − native, recovery_d = absent_b − arm_{b,d},
f_d = sum over that half's docs' d-mask positions of <recovery,x> / <x,x>
(the §2640 pooled estimator, recomputed here — pred_a checks it reproduces the
§2640 receipt values to ≤ 1e-9). Bootstrap: B = 10,000 resamples of the 250
documents in each half WITH replacement (torch.Generator seed 20260906, drawn
once, shared across cells for paired comparison); per resample recompute f(1/128)
and f(1/64) on the resampled docs' masked positions; record Δ = f(1/128) −
f(1/64). Report the Δ distribution's mean, 2.5/50/97.5 percentiles, and
p_pos = fraction of resamples with Δ > 0, per cell.

## Frozen predictions

### pred_a — exact deterministic instrument
Bundle sha matches the §2640 receipt's bundle field; array shapes exact
(native 500×256, arms 2×6×500×256, six masks); the point-estimate f values for
all six densities reproduce the §2640 receipt profile to ≤ 1e-9; every tail
mask non-empty in every resample-eligible document set.

### pred_b — the tail uptick has a DEFINITE sign (real, not noise)
In all four cells, p_pos ≥ .975 (the uptick is a real positive: isolated-edit
reading supported) — OR, as the registered alternative outcome, p_pos ≤ .025
in all four (real negative). Either definite-sign result resolves the uptick as
structure.

### pred_c — bootstrap self-consistency
In all four cells the point-estimate delta lies inside the bootstrap
[2.5, 97.5] interval — a resampling-unbiasedness check (a point estimate
outside its own resample CI signals a coding fault, not science). Does not
gate the science verdict (pred_b carries that); reported alongside.

### Strong null (estimator-noise verdict)
Fires if any cell has p_pos ∈ (.025, .975): the tail uptick is within
document-sampling noise — reading (a), estimator bias, is supported, the §2640
tail is statistically flat, and the extraction certificate's hull is unchanged
(the monotone envelope already absorbed it). No bar changes; the result routes
to "certificate hull stands; tail is flat within noise," closing the question.

## Price

Zero model forwards; CPU < 30 s; one receipt JSON. Nothing deployed.
