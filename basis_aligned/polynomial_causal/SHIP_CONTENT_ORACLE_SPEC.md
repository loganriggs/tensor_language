# Optimizer-free live-ship content oracle screen

## Motivation

The MLP0-2 factorial treatment is an engineering localization, not an intrinsic
causal decomposition. It bundles three planks and the incumbent CE-trained MLP2
glue; its attention interaction is large and partly induced because early planks
were fit under the attention composite. A newly trained CE correction could also
compensate for arbitrary downstream errors without recovering the missing MLP
computation.

Before training another patch, this screen asks a narrower falsifiable question:
at a live full-ship state, does restoring the original MLP computation repair CE,
and is the useful part unusually aligned with the previously frozen content
subspace?

## Frozen intervention

For each singleton site `l in {0,1,2}`, compute on the live ship input

```text
e_l(z_l) = MLP_original,l(z_l) - MLP_plank,l(z_l)
```

with every other replacement and the incumbent MLP2 glue active. Inject one of:

1. the full `e_l`, which is the conditional recoverable ceiling;
2. `P_content e_l`, using the frozen 64D deep-content basis;
3. `P_local e_l`, using the top 64 through-origin residual PCs fit on 96 disjoint
   rows at skip 1200;
4. 20 seeded null projections. Each null is a Haar-random 64D subspace inside
   that site's top-256 residual support and is scaled to the content arm's fit-set
   correction RMS.

With exactly 20 nulls, the authoritative v2 gate uses the exact one-sided Monte
Carlo rule: content must beat all 20, giving
`p=(1 + #null >= content)/21 = 1/21`. The earlier interpolated 95th-percentile
rule is retained only in the preliminary result and cannot license OOD work.

There is no optimizer and no learned decoder. Thus a benefit cannot arise from a
new patch learning a ship-specific compensation. All arms are singleton slot
interventions.

The saved content factors have `5.65e-4` maximum Gram drift from serialization.
The screen QR-orthonormalizes their 64D column span before constructing
`P_content`; QR preserves the frozen subspace exactly and prevents a non-idempotent
"projection" from changing the intervention.

Discovery and held-out each use 192 rows at skips 7000 and 11000. The rare-token
vocabulary is frozen from discovery and reused on held-out. Copy means recurrence
at context distance 1 through 64; this fixes the off-by-one mask in the first
factorial stage. Paired row-bootstrap 95% intervals use 2,000 draws.

## Registered decisions

A site licenses learned content prediction only if:

1. the full-oracle held-out gain has bootstrap lower bound above zero;
2. the content projection improves global CE on both splits; and
3. its held-out gain beats all 20 covariance/energy-matched null gains.

Report the content fraction of full-oracle gain, correction RMS, all cell gains,
and the local-PCA result without clipping negative values. A full oracle that does
not help means later CE training at that slot would be compensatory glue. A content
arm that loses to matched nulls rejects this fixed content coordinate system even
if the full or local-PCA oracle succeeds.

Passing is only a license to fit the projected missing computation. It does not
establish semantic content or a causal program; output-slice, intervention-family,
alternate-background transfer, and OOD tests remain required.
