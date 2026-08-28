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
| shared-QK-512 | 503,436,726 | 7.7793% | 0.00866 / 0.00975 | 0.9149 | 0.9565 | passes opened-role gates |
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
