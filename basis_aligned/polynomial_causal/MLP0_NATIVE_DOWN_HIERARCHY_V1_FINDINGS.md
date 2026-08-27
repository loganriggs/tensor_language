# MLP0 native-Down hierarchy v1: executable result

## Verdict

The registered lexical write hierarchy is rejected as a simpler executable
replacement for MLP0's native `Down` map at either tested price rung. All ten
candidate programs fail the absolute interface gate in each independent
192-source-document wave and in the pooled analysis. The best arm, the
unstructured continuous `C512` control, still has pooled family-wise lower bound
3.389 times the registered tolerance.

The lexical partitions are not meaningless: `Q` and `A` beat their
assignment-preserving deranged-centroid nulls in the family maximum. But they
lose to the matched-byte continuous controls and fail the registered pointwise
no-free-rider tests against both the controls and their nulls. Thus the
experiment detects lexical organization without earning lexical simplicity
credit.

This is a result about replacing `Down` while retaining and explicitly pricing
the exact native Left/Right product state. Even a pass would not have simplified
all of MLP0. Because there is no pass, it adds no executable whole-model recovery
credit.

## Artifact and integrity

- Result: `basis_aligned/bilinear_quotient/mlp0_native_down_hierarchy_v1_results.json`
- SHA-256: `81ef485a0b4a734c0ca63747c854d4e1a34f78d1fa55eb759adadd8bb25ef71f`
- Rows: 384 unseen FineWeb source documents, split before evaluation into two
  disjoint 192-document waves; 607 chunks and 310,784 raw prediction positions.
- Evaluated coverage: 0.93513 in wave A, 0.93903 in wave B, and 0.93711 pooled.
- The minimum pooled cell support is 60 independent source documents; every
  registered support gate passes.
- The cloned-native control reproduces logits, CE, MLP0, attention-1, and MLP1
  exactly. The poison canary raises once. Candidate evaluation makes zero calls
  to the original `Down` and 304 proxy calls per arm. The native `Down` weight
  hash is unchanged.
- The saved sufficient-statistics ledger exactly replays the frozen inference
  result with 20,000 source-document family-wise bootstrap draws.

## Absolute result

The table reports the pooled maximum standardized effect and its family-wise
95% upper and lower confidence bounds. A candidate needed both waves below one,
pooled UCB below 0.8, and all other registered gates.

| arm | physical bytes | point max | UCB | LCB | absolute credit |
|---|---:|---:|---:|---:|---|
| C256 | 2,955,520 | 7.9961 | 9.3230 | 6.7396 | no |
| Q248 | 2,952,371 | 8.7338 | 10.0608 | 7.4774 | no |
| A247 | 2,954,675 | 8.7141 | 10.0410 | 7.4576 | no |
| C512 | 5,904,640 | 4.6456 | 5.9725 | 3.3891 | no |
| Q504 | 5,901,491 | 6.0383 | 7.3652 | 4.7818 | no |
| A503 | 5,903,795 | 6.0226 | 7.3495 | 4.7661 | no |

The null arms also fail, with pooled point maxima from 6.6494 to 9.4663. The
binding cell for every main arm is the MLP1 state at
`pos0_freq0_prev0_dev1`: early positions, low fit-frequency tokens,
non-punctuation predecessors, and high pre-MLP0 norm. The raw MLP1 nRMSE there
is 0.39980 for C256 and 0.23228 for C512 against a 0.05 tolerance. The failure is
stable rather than a wave accident: C512's point maximum is 4.6554 in wave A
and 4.6361 in wave B.

## What the C512 mismatch does and does not say

C512 is the best tested program. In the pooled cells, its worst final KL is
0.005326 against the 0.01 point margin and its worst CE harm is 0.005492 against
0.0075. Those point estimates are inside their behavioral margins. However,
attention-1 reaches 0.05437 and MLP1 reaches 0.23228, so the direct interface
gate fails decisively. The registered family-wise decision therefore remains a
failure.

This gap is scientifically useful. It localizes two incompatible possibilities:

1. C512 discards directions that are large in activation norm but lie mostly in
   a downstream behavioral null space. In that case it may be a useful
   behavior-preserving compression, but not yet a causal or modular interface.
2. The downstream network compensates on the observational distribution, while
   the missing directions are required under interventions, composition, or
   distribution shift. In that case the small final KL/CE is an observational
   coincidence and cannot support a manipulable program.

The current result cannot choose between them. It also cannot justify relaxing
the direct-interface gate after seeing C512. A new claim needs fresh rows and a
separately frozen causal/compositional test.

## Consequences for the simplicity program

This experiment separates three notions that had been conflated:

- **lexical organization:** supported relative to structured derangements;
- **description length at fixed observed fidelity:** not supported, because the
  lexical arms are worse than matched-byte continuous maps;
- **causally composable sufficiency:** not supported by any tested arm.

Therefore a useful simplicity objective cannot reward clusters merely because
tokens share a centroid. It must jointly charge serialized program size and
the residual information required by downstream consumers, and it must evaluate
that residual under interventions and composition. The observed MLP0/MLP1/MLP2
superadditivity makes a purely site-local decomposition especially suspect.

## Pruned next directions

1. **Fresh C512 downstream-null versus causal-interface discriminator.** Measure
   final and internal response operators under frozen, bounded interventions at
   the MLP0 output, with native and C512 paths receiving identical perturbations.
   This directly decides whether the large MLP1 mismatch is behaviorally null or
   merely compensated on-distribution. Add code/OOD rows where available.
2. **Joint conditional MLP0-to-MLP1 compiler.** If the mismatch is predictable,
   price a small transport/adaptor jointly with C512 and compare it against both
   a higher-rank continuous `Down` map and an equal-byte direct control. This is
   the minimal test of the proposed joint simplicity objective.
3. **State-complete MLP2 composition.** Revisit the exact early MLP cube only
   after the MLP0-to-MLP1 interface is identified; singleton scores are invalid
   currencies in the observed superadditive regime.
4. **Same-row executable residual census.** Attribute remaining final CE/KL to
   interfaces on one frozen row population, without adding incompatible module
   denominators.
5. **OOD/edit/collateral certification.** Require any promoted component to keep
   its claimed behavior under held-out domains and targeted edits while bounding
   damage to non-target behavior.

More hard clustering, an unregistered rank sweep, donor swaps, and oracle-only
PCA are lower priority. They either repeat this negative, optimize the wrong
interface, or cannot yield an executable program.
