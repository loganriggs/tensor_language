# Rank512 cross-task validation: predictive pass, fresh-context failure

Date: 2026-08-28

Status: measured gate failure. Cross-task predictive validation passes, but the new
deterministic prefix fixture narrowly misses both causal thresholds. Rank512 is not
promoted as a robustly interchangeable compressed program.

## Cross-task FineWeb result

The roles at skip31000 and skip35000 were never used for attention fitting or rank
selection and pass their prospective serialized/raw hash authority checks.

| role | all-position CE harm | covered harm | unseen-current harm |
|---|---:|---:|---:|
| skip31000 | +0.010453 | +0.009377 | +0.013426 |
| skip35000 | +0.009423 | +0.009837 | +0.008149 |

All predictive, unseen-support, no-degradation, ownership, cost, and replication gates
pass. The approximate 0.01-nat harm from the opened roles transfers cleanly.

## Fresh context fixture

- native/program maximum downstream changes: 3.443508 / 3.568322;
- delta norm ratio: 0.973423;
- context-delta recovery: 0.892899, below 0.90;
- delta cosine: 0.945350, below 0.95.

The miss is small but preregistered and counts. Together with the opened fixture's
0.914855/0.956512 pass, it shows that the causal boundary at rank512 is intervention-
sensitive. A single selected poke is not a stable certificate.

## Consequence

Rank512 remains an excellent predictive compression point and a conditional opened-role
causal pass. It is not yet a robust causal abstraction. The next experiment should
replace one-poke decisions with a frozen bank of prefix positions, token fixtures, and
perturbation magnitudes, reporting the distribution and lower confidence bound of
context recovery. Rank640 and the causally weighted shared basis should then be compared
on that bank; simply lowering the 0.90/0.95 gates is not licensed.
