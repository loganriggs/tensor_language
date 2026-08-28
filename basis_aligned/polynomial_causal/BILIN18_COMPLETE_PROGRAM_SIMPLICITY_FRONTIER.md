# Bilin18 complete-program simplicity frontier

Date: 2026-08-28

Simplicity is not represented by one scalar. A candidate first passes executable
ownership and total-support gates; surviving candidates are compared on stored values,
operations, predictive CE, and causal transport. This makes the definition operational:
a simpler program must enable lower executable cost without losing the behavior needed
for prediction and intervention.

| complete program | stored values | saving | all-position CE harm, 7000 / 11000 | context recovery | delta cosine | status |
|---|---:|---:|---:|---:|---:|---|
| dense exact | 545,904,054 | 0% | 0 / 0 | 1.0000 | 1.0000 | exact reference |
| shared-QK-640 | 516,707,766 | 5.3481% | pending | 0.9444 mean, 0.9273 LCB | 0.9724 mean, 0.9637 LCB | 16-fixture causal pass; predictive gate pending |
| shared-QK-512 | 503,436,726 | 7.7793% | 0.00866 / 0.00975 | 0.9149 | 0.9565 | opened fixture pass; fresh fixture fails |
| shared-QK-384 | 490,165,686 | 10.2103% | 0.01843 / 0.01991 | 0.8468 | 0.9211 | causal gate failure |

The rank384 result shows why storage and CE alone are insufficient: the most compressed
candidate predicts nearly as well but rotates the downstream prefix effect too much.
Rank512 is therefore simpler than dense under the current consequence-constrained
definition, while rank384 is only a predictive compression point.

This frontier is provisional in two ways. First, the roles and original prefix fixture
are not fresh after rank selection, so rank512 requires untouched-row and intervention-
bank validation. Second, dense MLPs still account for 286,675,200 values (52.51% of the
dense model); attention compression alone cannot approach the eventual program-size
frontier.

Cross-task FineWeb validation subsequently reproduces rank512's predictive harm at
0.0094--0.0105 nat, but a new deterministic prefix fixture obtains only 0.8929 context
recovery and 0.9454 cosine. Both miss the frozen 0.90/0.95 gates. Rank512 is therefore a
robust predictive point but not yet a robust causal abstraction; a multi-intervention
distribution replaces single-fixture admission next.

The prospective 16-fixture bank rejects rank512 (0.8982 mean recovery, 0.8634 lower
bound, 8/16 individual passes) and admits rank640 causally (0.9444 mean recovery,
0.9273 lower bound, 0.9724 mean cosine, 0.9637 lower bound, 14/16 passes). Rank640
improves every paired fixture. It remains a causal candidate until predictive CE is
measured at the same rank; complementary passes from different candidates are not
composed into one certificate.
