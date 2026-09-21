# The linear-sharing regression is already present at early positions

21 September 2026, 02:49 UTC.

The position diagnostic rejects a simple explanation for the failed full-contribution preservation: the regression is not confined to positions beyond the64-position calibration window. It is already substantial within that range and does not consistently grow later.

We kept the same programs, documents and donor mapping, added recipient-position bins to the shared native evaluator, and performed no fitting. Summing bin energies reproduces the unbinned results to relative error below $3.7\times10^{-16}$.

## Whole-contribution swap errors

| Recipient positions | FineWeb: exact linear → shared separate256 | Code: exact linear → shared separate256 |
|---|---:|---:|
| 16–63 | 23.23% → 29.59% | 20.60% → 26.88% |
| 64–127 | 23.37% → 29.70% | 20.88% → 26.17% |
| 128–255 | 23.27% → 30.43% | 19.62% → 24.75% |

The registered predictions both fail: the late compression penalty is not at least10% larger than the early penalty in both domains, and the early shared program does not preserve error within5% of the exact-linear program.

A successor CPU ratio audit finds that the late-to-early compression-penalty ratio is1.027 for FineWeb and0.967 for code. The source-only preservation advantage remains similar across positions, while context-only results remain identical by algebraic construction. The failure is specific to the broader intervention families, not a binning or baseline-replay discrepancy.

These bins describe recipient position; same-token donors may occupy other positions. Thus this is not a causal manipulation of sequence length and does not rule out all input-distribution shift. It does rule out the proposed explanation that the observed regression appears only beyond calibration's position range.

## Consequence for the next step

Collecting longer contexts alone is not justified as the primary repair. We should examine how error from the linear approximation combines with the already-imperfect bilinear approximation, and whether jointly refitting their shared output coefficients improves the complete allowed intervention family.

A lower error on either component by itself need not minimize their combined error. Any joint refit must be scored on source-only changes and whole-contribution changes together, with calibration-only fitting and subsequent native validation. It must not sacrifice a failed family to obtain a narrower passing gate.

The current shared-linear graph remains an unadopted tradeoff. Stable semantic identities, selective behavioral manipulation, extraction, composition and broader OOD prediction remain unresolved.

Evidence: `MIDPOINT_POSITION_AUDIT_V1.json`, its raw nested position-bin records, and `MIDPOINT_POSITION_PENALTY_SUMMARY_V1.json` under `direct_tensor_match`. The optional binning utility lives in `logit_effect_partition.py` and is exercised through the shared native executor.
