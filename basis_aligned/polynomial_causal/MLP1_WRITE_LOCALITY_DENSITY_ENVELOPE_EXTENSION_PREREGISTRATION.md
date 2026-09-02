# Preregistration — locality-density envelope extension (six points; parallel lane)

Date: 2026-09-02 22:32 UTC
Owner: Claude (parallel probe lane)
Status: frozen before any 1/128- or 1/2-density outcome exists

## Question and lineage

§2636 measured the locality law at four densities (f = .215–.307 / .326–.423 /
.468–.546 / .946–.963 at d = 1/64, 1/16, 1/4, 1; aligned recovered fraction on
edited positions — higher = more restored). The 2209 math review formalized its
use as an EXTRACTION-COST CERTIFICATE: any compiled program reproducing the
MLP1 write on a fraction s of positions recovers at most f̂(s), the monotone
upper envelope of the measured curve. This rung tightens the envelope with two
new densities (1/128 below the current lowest; 1/2 in the widest gap) and
re-measures all four §2636 points in-run (post-§2599: no numeric bridges), on
the same nested-mask construction and seed. Codex's lane (509 gated pipeline
at MLP10) is untouched; ~2.5 min GPU in the idle slot.

## Arms (rung 493 machinery verbatim; branches b ∈ {T, I})

Densities D6 = {1/128, 1/64, 1/16, 1/4, 1/2, 1}, nested by the SAME
uniform-threshold construction and the SAME seed 20260904 as v1b — so the four
common densities produce BIT-IDENTICAL masks to §2636's, and the two new masks
nest strictly between them. Per batch of 4 (125 batches, docs 0:500, halves at
250): NATIVE, ABSENT_T, ABSENT_I, and OWN_DENSITY(d)_b for the six densities ×
two branches (edited write = M_b with M_N on the mask; positions ≥ 1). 15
forwards/batch; 1,875 total. Scoring exactly as §2636 (f and cosine on edited
positions, per branch × half).

## Frozen predictions (constants DERIVED in-text)

### pred_a — exact, lawful, live instrument
Rung 493 identity suite at its registered bounds; calls exact (125 + 250 +
1,500 = 1,875); nesting exact across all six masks; supports derived from the
construction: at d=1/128, expected scored positions per half = 250·255/128 ≈
498 → floor ≥ 420 (≈ −3σ under Binomial(250·255, 1/128), σ≈22); expected
non-empty docs per 250 = 250·(1−(1−1/128)^255) ≈ 250·.865 ≈ 216 → floor
≥ 195 (≈ −3σ, σ≈5.8); every cell mask non-empty at every density; every edit
RMS > 0.

### pred_b — the six-point profile is strictly monotone with the §2636 span
In all four branch×half cells: f strictly increasing across all six densities,
with f(1) − f(1/128) ≥ .30.

### pred_c — determinism anchor + envelope tightening
The four common-density cells reproduce §2636's values to ≤ 1e-9 (identical
arms, same seed — a bit-level determinism check doubling as instrument
validation; the v2/v2b precedent), AND the new points interleave: f(1/128) <
f(1/64) and f(1/4) < f(1/2) < f(1) in every cell (implied by pred_b; stated
so a pred_b near-miss localizes).

Descriptive regardless: the six-point monotone envelope f̂ per branch (the
certificate object), the low-density slope f(1/64)−f(1/128) (does locality
keep falling, or floor out — the asymptote question), and T-vs-I ordering.

## Strong null and interpretation

Strong null fires if any pred fails. A pred_c reproduction failure is an
INSTRUMENT alarm (determinism or mask-construction fault), routed to repair
only. A pred_b monotonicity break at the new points with pred_c clean is
science: the envelope is not globally monotone and the certificate must be
restated on the monotone hull — reported as-is, no refit. A pass hands the
extraction certificate a six-point envelope with a measured low-density
asymptote direction. No compression/rank claim on any route.

## Literal price

1,875 full-model forwards (~2.5 min), single phase, docs 0:500; no validation
or sealed roles. Bundle: per-token CE for all arms + six nested masks. Zero
deployed parameters.
