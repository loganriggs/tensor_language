# Hourly strategic review — 2026-08-30 13:25 UTC

## Bottom line

The first complete real shared/private causal-response factorization grid is finished.
All 51 FIT-only cells (17 rank pairs × 3 seeds) completed healthily in 1,052.76 summed
optimizer-seconds, with no numerical failures and no validation or EVAL access.

The result is narrower than “we found the hierarchy.” There is strong shared low-rank
structure across the 49 × 49 intervention library, but no training evidence that the
six proposed source-owner-private branches are the right hierarchy. All private-only
points are dominated. Three joint points remain nondominated only because they exchange
fewer persistent values for more per-document coordinates; they do not beat the
shared-only curve on fit at comparable execution state.

Nothing moves the strict whole-model ledger:

- certified removable storage: 29,196,288 / 545,904,054 = **5.348245316%**;
- named deletion cross-entropy: 0.57968 / 5.30682 = **10.923302467%**;
- unexplained deletion cross-entropy: 4.72714 nat = **89.076697533%**; and
- complete prediction/extraction/removal/OOD circuits: **0/68**.

This grid compresses a measured causal interface. It has not yet predicted a held-out
document, composed through RMSNorm/residual interfaces, or enabled a selective edit.

## Computation and definitions

The measured object is a signed response tensor

$$
R_{pstd},
$$

where $p$ is one of two intervention phases, $s$ is one of 49 source circuits,
$t$ is one of 49 target circuits, and $d$ is one of 229 FIT documents. The program is

$$
\widehat R_{pstd}
=\sum_{k=1}^{K_0}a_{pk}b_{sk}c_{tk}h_{dk}
+\sum_g \mathbf 1[s\in g]
\sum_{j=1}^{K_g}a^{(g)}_{pj}b^{(g)}_{sj}c^{(g)}_{tj}h^{(g)}_{dj}.
$$

The first sum is shared by every source owner. The second gives each of six source
owners an optional private branch. $h_d$ is the small vector of coordinates required
for one document. This is multilinear and has no top-k router.

Two simplicity prices remain separate:

$$
P=100K_0+355K_g
$$

persistent stored values, and

$$
C=K_0+6K_g
$$

coordinates inferred per document. A point is dominated only if another point is no
worse in both prices and all compared errors, and strictly better somewhere. The
frontier uses the median of all three seeds for one rank pair. It never treats a lucky
seed as a separate candidate.

## Numerical result

The zero-program training MSE is 0.0460729043. The observation-wise training mean has
MSE 0.0429483798, costs 4,802 persistent values, and carries no document state.

| program | $P$ | $C$ | median MSE | response energy explained | worst owner-pair NRMSE |
|---|---:|---:|---:|---:|---:|
| shared rank 1 | 100 | 1 | 0.041553247 | 9.81% | 2.9802 |
| shared rank 8 | 800 | 8 | 0.031415999 | 31.82% | 2.5974 |
| shared rank 16 | 1,600 | 16 | 0.024361129 | 47.12% | 2.1418 |
| shared rank 32 | 3,200 | 32 | 0.016047438 | **65.17%** | **1.6219** |
| joint $(K_0,K_g)=(4,1)$ | 755 | 10 | 0.034219853 | 25.73% | 2.7349 |
| joint $(8,2)$ | 1,510 | 20 | 0.028519744 | 38.10% | 2.4997 |
| joint $(16,4)$ | 3,020 | 40 | 0.020332029 | 55.87% | 1.9572 |

“Response energy explained” is $1-\mathrm{MSE}/0.0460729043$ on FIT. It is not a
held-out prediction score. The shared rank-32 point also removes 62.64% of the MSE left
by the observation-wise mean, while using fewer persistent values but 32 coordinates
per document.

The complete training frontiers are identical whether the error coordinates contain
pooled MSE alone or pooled MSE plus worst-owner-pair NRMSE:

- shared ranks 1, 2, 4, 8, 16, and 32;
- joint ranks $(4,1)$, $(8,2)$, and $(16,4)$.

No private-only point survives. The joint points survive because $(P,C)$ is a genuine
two-coordinate order: compared with the next shared point they use slightly fewer
persistent values but more document coordinates and fit worse. This is a legitimate
tradeoff, not evidence that the private branches discovered semantic subprograms.

## The important confusing result

Every one of the 17 rank pairs has the same hardest block: source owner `m16` to target
owner `m16`. For shared rank 32, pooled MSE is 0.01605 but source-`m16` MSE is 0.10259.
Its worst-block NRMSE remains 1.62. The residual intervention phase is also harder than
the full phase (MSE 0.02018 versus 0.01192).

This means pooled compression is hiding a specific missing interface. It is not enough
to increase a universal CP rank and call the library understood. A plausible reason is
that `m16` interactions require a different structured branch, but that is a hypothesis,
not yet a result; the registered owner-private branch did not solve it efficiently.

Seed behavior is reassuring but not dispositive. All cells are healthy. Shared rank 32
has a 1.06% max-minus-min MSE spread relative to its median; joint $(16,4)$ has 2.98%.
The effect is therefore not carried by one optimizer seed.

## What fraction is actually explained

Three statements must stay separate:

1. **65.17% of FIT response energy** is reconstructed by one 32-coordinate shared
   response program. This is a real compression result for the measured library.
2. **0% new held-out response behavior** is established, because the 114 validation
   documents remain sealed and training codes were fit using their documents' cells.
3. **0% new whole-model ledger credit** is established. The strict named behavioral
   fraction remains 10.9233%, and no terminal circuit exists.

## Pruned actions

- More private-only ranks are pruned: every tested private-only point is dominated.
- A semantic story for the owner hierarchy is pruned until validation; the training
  frontier does not support it.
- More pooled-MSE-only fitting is low value because `m16 → m16` is the visible failure.
- A global full-model tensor decomposition remains premature; the smaller causal
  interface has not transported yet.
- SAE/HOSVD or top-k routing on local activations remains lower priority because it
  does not address held-out causal prediction or the broken owner interface.

## Ranked next five actions

1. **Freeze the nine nondominated rank-pair candidates and all three seed programs for
   each.** This is deterministic CPU work, prevents validation-shaped selection, and
   is the prerequisite for lawful held-out scoring.
2. **Audit and run the existing held-out scorer on all frozen candidates.** Report
   unconditional mean-code transport and calibrated non-anchor prediction at 2, 4, 8,
   and 16 physical source arms, including all 36 owner pairs. This has the highest next
   information gain and directly tests whether 32 coordinates are a useful program or
   merely a training code.
3. **Use the validation result to choose between topology repair and compression.** If
   failure remains concentrated at `m16 → m16`, fit a prospectively specified
   `m16`-block structure rather than more universal rank. If all blocks fail, reject
   the v1 response-program entry point.
4. **Apply quotient-Jacobian, tree-cut-rank, and sparse-code certificates only to
   validation survivors.** These decide whether atoms are stable/minimal/editable;
   they cannot rescue a nontransporting predictor.
5. **Turn survivors into whole-model causal tests.** Predict fresh interventions,
   compose across early MLP/RMSNorm/residual interfaces, then test extraction,
   selective removal with unrelated-target controls, and a second domain.

## Action executed this review

The exact 51-cell GPU grid completed. A source-closed FIT-only analyzer was implemented,
red-teamed through three NO-GO rounds, and repaired against best-seed selection,
malformed terminal, forged source/input/protocol, and forged-health attacks. Five
focused tests pass. Independent review returned GO on analyzer source
`2db6e7c1…`, tests `7a736455…`, and analysis closure `ccdccd7b…`. The immutable
analysis receipt is
`causal_response_factorization_v1_training_analysis.json`.

No validation or EVAL value was read, no candidate was selected or frozen, and no
scientific failure was discarded.
