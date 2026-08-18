# Pre-registered predictions for the next bilinear checkpoint

Committed 2026-08-18, before any larger checkpoint is available. Every
prediction below is frozen here with its numeric bar and instrument so the
registration provably precedes the data. Instruments are named by file;
each has been validated on bilin18 (18L, D=1152) and bilin12 (12L, D=768).

## Standing rules (bought by ledger entries 15-18)

1. **Register the metric, not just the number** (ledger: the §208 near-miss).
   Every bar below names its instrument file.
2. **Measure the random floor before applying any bar** (ledger #17). Where a
   bar says "above floor," the floor is the measured random-basis /
   random-projection median from the same run, never an assumed constant.
3. **Before deferring a fork off-box, write the orthogonalization that would
   decide it** (ledger #18).
4. Honest parameter accounting: count warm bases (ledger #15).

## The bundle

For a bilinear-MLP transformer checkpoint with L layers (L > 18 preferred),
same training family:

**P1 — Slack scales with size** (family_size_scan.py; §§203-205).
Number of rank-0-licensed tail layers (CE cost <= 0.05) is >= the 18L count
(4). The rank-0/rank-4 scan is the mandatory first measurement (BENCHMARK.md).

**P2 — Slack-regularizer identity** (family_regularizer_scan.py; §§206-207).
The deletion-improves span set (top-8 span deletion <= -0.01 on TWO
independent span fits from disjoint stats rows -- boundary flips on a single
fit do not count; §228) is a subset of the constant-licensed layer set. Zero
tolerance on replicated violations.

**P3 — The private writer** (bilin18_behavioral_writers.py + landscape
scripts; §§209-228, REVISED after ledger #19). Aggregation is POOLED median
(never fold-median), and the profile is measured under >= 3 disjoint reader
ensembles. The bar is RELATIVE: exactly one writer whose pooled LORO is
<= 0.5x the in-model control median in EVERY ensemble, at depth fraction
0.33 +/- 0.08, width one layer, with privacy concentrated in the top-8
output span: tail coords 9-48 share at >= 0.35 pooled in EVERY ensemble
(the ensemble-invariant form, §229 -- the ratio-to-full form is not used
because the full-coords denominator is ensemble-sensitive).
Absolute floor-crossing is reported but is NOT the bar (it is
ensemble-sensitive; §228). MLP-side only: the attention writer at the
notch either recovers under span-orthogonalization (18L carrier pattern) or
shows no notch (12L pattern) -- either is consistent; an attention notch
that survives orthogonalization is NOT.

**P4 — MLP sharing declines with depth** (same instrument; §227).
Spearman(depth, LORO) <= -0.3 over >= 8 MLP writers.

**P5 — Last-reader secession strengthens with depth** (bilin12_solitary_
reader.py pattern; §§210, 218). The deepest reader is the worst LORO fold
for a majority of writers; its median fold at L >= 18 layers is <= 0.25
(secession), at L < 18 merely lowest (ordinal).

**P6 — Depth-fraction placement law** (family_edge_kinship.py instruments;
§§180-190 arc). Component best-match placement across the family tracks
depth fraction with zero anomalies at the direct-best-match instrument.

**P7 — Whisper structure** (bilin18_dialects.py; §§220-221, 230). Over the
new model's private span: per-reader self-basis behavioral R^2 exceeds the
population cross-basis by >= 0.3 at matched rank, with rank <= HALF the
form-space ambient dimension (the discriminative regime; at 2/3 ambient the
instrument saturates by construction -- §230), and no reader group
(early/deep) transfers above the measured random floor. I.e. structure
without agreement, idiosyncrasy total.

**P8 — Partial conservation** (family_private_fingerprint.py; §217). The new
private span's per-token damage fingerprint correlates with bilin18's at
rho >= 0.10 on shared text AND above the 95th percentile of random-span
pairs, while remaining BELOW the matched shared-span pair correlation.

## Falsification stance

P2 and P3 are the sharp ones: single-violation kills. P1/P4/P5 are trend
claims with stated bars. If P3 fails -- no notch, or wrong fraction, or
width > 1 -- the private-writer regularity was a two-model coincidence and
§§215-226's universality claims must be withdrawn to bilin18+bilin12 scope.
That sentence is part of this registration.
