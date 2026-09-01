# Hourly strategic review — 2026-09-01 06:30 UTC

## Goal and standard

Recover a substantially smaller executable tensor program for bilin18 that is:

- predictive on census, fresh documents, and shifted corpora;
- composable across independently useful replacements;
- manipulable under direct signed interventions;
- literally simple, with every tensor, index set, fallback, scalar, and byte priced.

Low local reconstruction error, weight energy, and validation R2 are screens, not adoption evidence.

## What changed since 05:30

1. Context-covariance reduced-rank regression transferred from MLP input maps to all 440 Q/K maps.
2. Independent-fit context-QK96 reached +.001415 census damage, 61/62 certificates, and near-zero shifted damage
   at 535,089,462 scalars, then passed signed a16 at cosine .997617 and collateral rho .998265.
3. Context-QK96 + context-MLP0-p448 composed at +.009585/49 and 529,117,494 scalars. Its ratio to the frozen
   additive prediction was 1.02135x, not the expected ~1.3x, and it passed signed a16 at .994186/.997653.
4. Cross-family composition repeated at MLP0 p512 and p640 with ratios 1.0241x and 1.0199x. Their census/certs
   are +.007630/51 and +.005087/54; all new WikiText tails held.
5. Context-QK88 removed another 4,505,600 scalars and landed at +.002196/58, only +.000781 above rank96, with
   shifted mean/p95/max -.003865/.006220/.007768. Its signed gate is running; rank80 is queued.

The fully gated frontier currently has two points pending the rank88 gate:

| Scalars | Census damage | Certificates | Status |
|---:|---:|---:|---|
| 535,089,462 | +.001415 | 61/62 | adopted context-QK96 |
| 529,117,494 | +.009585 | 49/62 | adopted QK96 + MLP0-p448 |

## Updated laws

### 1. Functional rank is metric-dependent

For a map `W` receiving `x` with covariance `C`, the correct local objective is

`E ||(W-W_r)x||^2 = ||(W-W_r) C^(1/2)||_F^2`,

not unweighted Frobenius error. This single correction now improves both MLP shared-input maps and Q/K maps. The
old hand-selected Q/K fine band and the apparent full-rank late-MLP cliff were both symptoms of weight-space
misranking.

### 2. The composition tax is provisionally intra-family

Within-MLP compositions repeatedly cost about 1.30–1.34x their additive solo damage, and sequential refitting did
not help. Three cross-family QK96 × MLP0 compositions now cost only 1.0199–1.0241x. The evidence now rejects a
universal tax. Working hypothesis: residuals within a subsystem align or amplify, while attention-map and MLP0
input residuals are close to functionally orthogonal.

This is still an empirical law, not a theorem. It needs another module family or a deliberately aligned
cross-family control before being used outside QK × MLP0.

### 3. The context-QK rank curve is gentle at 96→88

Removing eight ranks across 440 maps costs only +.000781 census and three certificates. This is unlike the
late-layer weight-SVD cliff and makes the rank ladder the highest expected-value next experiment.

## Confound audit

- Negative shifted means at QK96/QK88 are within row variance and mean “indistinguishable from zero,” not model
  improvement.
- QK ranks share one frozen split-B covariance. This is intentional for a literal ladder, but a final low-rank
  point still needs an independent-fit reproduction if the curve changes qualitatively.
- WikiText segments are advanced monotonically (140k, 160k, 180k, 200k, 220k) so no arm is selected on a reused
  tail. They are one corpus, so a second shifted corpus remains valuable robustness expansion.
- The p512/p640 composition variants share one physical run, native baseline, and QK map. This improves paired
  comparison but creates correlated evidence; both still need direct signed effects against their own CEVs.
- Certificate counts remain strongly one-dimensional with census damage. They are behaviorally meaningful gates,
  but not 62 independent statistical samples.
- The embedding-folded MLP0 structure result remains non-identifiable under R2 because bilinear gauges admit wrong
  priors with equal fit. No support/hierarchy claim should be revived without an intervention, OOD, or price
  discriminator.

## Ranked next actions

1. Finish signed QK88; physically evaluate QK80 on untouched WikiText. If both pass, continue to rank72/64 until a
   preregistered census, certificate, or tail cliff appears.
2. Run a common signed a16 gate for QK96 + MLP0 p512/p640. This would add two causal Pareto points and harden the
   cross-family law.
3. Combine the best lower QK rank with one MLP0 point only after its own signed adoption; predict by measured
   cross-family additivity, not the obsolete universal 1.3x multiplier.
4. Develop a downstream/Fisher or joint residual metric for within-family composition. Stale-context refitting is
   already falsified; the objective must change.
5. Add tail-robust covariance or leverage weighting where lower MLP0 ranks fail rare-row maxima.
6. Retain vocabulary sharing plus sparse exceptions as the largest independent storage-upside route, but require a
   larger held-out tail and physical compiler gate before returning GPU priority to it.

## Stop rules

- Do not relax rank-specific OOD maxima after observing a miss.
- Do not call a physical point adopted before signed intervention transfer.
- Do not interpolate a new rank after a tail failure without a new preregistration.
- Do not generalize cross-family near-additivity beyond QK × MLP0 until a distinct family pair tests it.
