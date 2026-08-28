# Bilin18 complete-program simplicity frontier

Date: 2026-08-28

Simplicity is not represented by one scalar. A candidate first passes executable
ownership and total-support gates; surviving candidates are compared on stored values,
operations, predictive CE, and causal transport. This makes the definition operational:
a simpler program must enable lower executable cost without losing the behavior needed
for prediction and intervention.

| complete program | stored values | saving | validation all-position CE harm | context recovery | delta cosine | status |
|---|---:|---:|---:|---:|---:|---|
| dense exact | 545,904,054 | 0% | 0 / 0 | 1.0000 | 1.0000 | exact reference |
| shared-QK-640 | 516,707,766 | 5.3481% | 0.00553 / 0.00445 (31k/35k) | 0.9444 mean, 0.9273 LCB | 0.9724 mean, 0.9637 LCB | prospectively admitted |
| shared-QK-512 | 503,436,726 | 7.7793% | 0.01045 / 0.00942 (31k/35k) | 0.8982 mean, 0.8634 LCB | 0.9498 mean, 0.9333 LCB | predictive pass; causal-bank failure |
| shared-QK-384 | 490,165,686 | 10.2103% | 0.01843 / 0.01991 (7k/11k) | 0.8468 single | 0.9211 single | causal gate failure |

The rank384 result shows why storage and CE alone are insufficient: the most compressed
candidate predicts nearly as well but rotates the downstream prefix effect too much.
Rank512 makes the same failure under a prospective distribution despite passing one
opened fixture. Rank640 is the first point simpler than dense under the current
consequence-constrained definition; lower ranks remain predictive-only compression.

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

Rank640 subsequently passes that missing predictive gate at +0.00553/+0.00445 nat
all-position harm and +0.00766/+0.00391 unseen-current harm. All measurements belong to
the same 516,707,766-value executable candidate. The strict whole-model frontier is no
longer empty: 5.3481% storage removal is prospectively certified, although it remains a
compression result rather than a semantic explanation of every retained coordinate.
