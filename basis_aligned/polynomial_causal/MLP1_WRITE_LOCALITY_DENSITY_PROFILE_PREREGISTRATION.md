# Preregistration — locality-density profile of the MLP1 write restoration (parallel lane)

Date: 2026-09-02 20:33 UTC
Owner: Claude (parallel probe lane)
Status: frozen before any density-mask outcome exists

## Question and lineage

§2618 measured that restoring the FULL MLP1 write recovers .956–.963 of the
T/I branch effect (aligned recovered fraction — higher = more restored).
§2620/§2626 found the sleeper: OWN restoration restricted to sparse
token-match masks (~2–7 positions/doc) recovers only .15–.53 of the effect AT
those very positions — the write's effect at position p is substantially
carried by writes at OTHER positions propagating through attention. This rung
turns that accident into a measured LOCALITY PROFILE: restore the native MLP1
write on seeded random masks of four densities and score recovery ON the
edited positions. If the write's effect were position-local, the profile would
be flat at ≈.956; the sleeper predicts a rising profile with a materially
non-local low-density limit. Codex's main line (505, five-site program on the
gauge) is untouched — different site, my probes' rung-493 machinery, imported
hash-pinned; at 1,375 forwards this slots into the idle slot without
materially delaying 505's 17,875-forward run whenever it enqueues (FIFO
serializes).

## Arms (named; branches b ∈ {T, I}; rung 493 `_merge_forward` M_ONLY; layers 2–17 recompute)

Per batch of 4 documents (125 batches, docs 0:500, halves at 250):

- NATIVE; ABSENT_b (rung 493 exact branch removal), b ∈ {T, I};
- OWN_DENSITY(d)_b for d ∈ {1/64, 1/16, 1/4, 1}: edited write = M_b
  everywhere, M_N on a seeded random position mask of density d (positions
  ≥ 1; per-document Bernoulli masks from torch.Generator seed 20260904,
  drawn once per document per density, nested so the d=1/64 mask ⊂ d=1/16
  ⊂ d=1/4 ⊂ all — nesting removes mask-resampling noise from the
  monotonicity comparison; d=1 is the §2618 full-write arm re-measured
  in-run).

11 forwards per batch (1 + 2 + 2×4); 1,375 total.

## Scoring

Per branch × half × density: x = CE(ABSENT_b) − CE(NATIVE) on the density-d
mask positions; recovery = CE(ABSENT_b) − CE(arm) there; aligned recovered
fraction f(d) = <recovery,x>/||x||² and cosine, scored ON the edited
positions only (the §2620 convention).

## Frozen predictions

### pred_a — exact, lawful, live instrument
Hashes match (rung 493 source/result, §2618 probe result, §2626 v2b result,
this prereg); rung 493 identity suite at its registered bounds; calls exact
(125 + 250 + 1,000 = 1,375); every mask non-empty in every document at every
density (d=1/64 gives ≈4 positions/doc over 255; require ≥1/doc and ≥3,000
scored positions per cell at the lowest density); every edit RMS > 0; nesting
verified exactly (each mask a subset of the next).

### pred_b — the profile rises monotonically and materially
In all four branch×half cells: f(1/64) < f(1/16) < f(1/4) < f(1) strictly,
with total span f(1) − f(1/64) ≥ .30 (measured anchors: sparse ≈.15–.53,
full ≈.956).

### pred_c — non-locality is material and the full anchor reproduces
In all four cells: f(1/64) ≤ .60 (the write's effect at a position is NOT
recoverable from that position's write alone), AND f(1) within .03 of the
§2618 full-write value for that branch (re-measured in-run: .956–.963
neighborhood; bar |f(1) − .96| ≤ .04 stated as an absolute window [.92, 1.00]).

Descriptive regardless: full f(d) and cosine tables, per-density scored-position
counts, T-vs-I profile comparison (the arc's T-most-token-specific theme
predicts T's profile BELOW I's at low density — stated, not scored), and the
implied propagation share 1 − f(1/64)/f(1).

## Strong null and interpretation

Strong null fires if any pred fails. Null routes: a FLAT high profile
(f(1/64) > .60 everywhere) would mean the write's effect is position-local
after all and §2620's low numbers were mask-selection artifacts — a genuine
reversal candidate that would then need an independent control before any
§2618/§2620 reinterpretation (correction discipline stated in advance);
non-monotonicity means density is the wrong variable — profile closed, no
refit. A pass gives the program its first measured LOCALITY LAW at a write
site: the fraction of a write's effect recoverable locally as a function of
restoration density — a required input for any future extraction pricing
(what granularity of write must a compiled program reproduce). No
compression/rank claim on any route.

## Literal price

1,375 full-model forwards (~2 min), single phase, docs 0:500 only; no
validation or sealed roles. Per-token CE for all arms + the four nested masks
(bool) in the bundle. Zero deployed parameters.
