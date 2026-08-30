# Causal-response factorization v1 — prospective FIT-only analysis

Status: frozen before the causal-response FIT bundle exists. This document opens no
EVAL artifact and authorizes no circuit or strict-ledger claim.

## Question and exact object

For phase $p$, source circuit $s$, target circuit $t$, and source document $d$, the
collector stores additive signed CE sums and counts. The fitted response is

$$
R_{pstd}=
\frac{S^{\mathrm{member}}_{pstd}}{N^{\mathrm{member}}_{td}}
-
\frac{S^{\mathrm{off}}_{pstd}}{N^{\mathrm{off}}_{td}}.
$$

A cell is valid only when both denominators are positive. Positive response means
that deleting the source direction worsened prediction more on the target circuit
than off the target's registered slice. The absolute-response sums remain diagnostics
and are never fitted.

The question is whether the signed array is predicted more efficiently by reusable
global response atoms plus component-private atoms than by either an unstructured
low-rank basis or independent component models.

## A necessary distinction: reconstruction is not new-document prediction

The formal block-term model is

$$
R_{pstd}\approx
\sum_{k=1}^{K_0} A_{pk}B_{sk}C_{tk}H_{dk}
+\sum_{g=1}^{6}\mathbf 1[g(s)=g]
\sum_{k=1}^{K_g} A^{(g)}_{pk}B^{(g)}_{sk}C^{(g)}_{tk}H^{(g)}_{dk}.
$$

The first term is a shared parent library. Each of the six registered owner
components is a private child. All operations are sums, products, and fixed masking,
so this remains a tensor-network program; there is no input-dependent top-k branch.

But $H_d$ is a free coordinate for document $d$. It cannot be known on a wholly new
document from this response artifact alone. We therefore keep two claims separate:

1. **Unconditional transport:** freeze the library on training documents and predict
   the mean signed response tensor of held-out documents using only the training mean
   code. This tests transport without looking at any held-out intervention response.
2. **Calibrated missing-cell prediction:** on a held-out document, infer only its
   low-dimensional code from a frozen anchor panel, then predict every non-anchor
   source-target response. This is response tomography, not zero-shot OOD prediction.

Neither claim is called semantic extraction, selective removal, or OOD circuit
transport. Those require a later fresh intervention and domain.

## FIT-only splits

The 343 FIT source-document IDs are ordered by SHA-256 of
`causal-response-factorization-v1-doc|<decimal document id>`. The first 229 are
library-training documents and the remaining 114 are internal validation documents.
No row may cross roles because the source document is the unit.

The anchor panel is chosen without responses. Flatten $(p,s,t)$ in row-major order
and order the 4,802 cells by SHA-256 of
`causal-response-factorization-v1-anchor|p|s|t`. The first 384 cells are anchors.
All valid non-anchor cells are scored. A validation document with fewer than twice the
candidate code dimension valid anchors is unsupported for that candidate and cannot
silently enter a pooled score. At least 90% of the 114 validation documents must be
supported or the candidate is ineligible for the Pareto frontier; the exact supported
document IDs and fraction are retained.

## Frozen candidate families

Every learned candidate is run at seeds `2026083001`, `2026083002`, and
`2026083003`. The reported point is the median validation loss; the full range is
retained. A failed/nonfinite optimizer or a seed whose final loss did not improve by
at least $10^{-4}$ relative to initialization is unhealthy.

- **Global CP:** $K_0\in\{1,2,4,8,16,32\}$ and every $K_g=0$.
- **Independent private:** $K_0=0$ and a common
  $K_g\in\{1,2,4,8\}$ for the six fixed owner groups.
- **Shared plus private:**
  $(K_0,K_g)\in\{(1,1),(2,1),(4,1),(4,2),(8,2),(8,4),(16,4)\}$.
- **Unstructured SVD control:** a dense observation-by-code basis with ranks chosen
  after FIT only by the exact stored-price matching rule below. It receives no CP or
  component topology advantage.
- **Per-cell mean control:** one training mean for each of the 4,802 observation
  cells. It has no per-document code.

The six owner groups and all source membership are inherited from the sealed circuit
order. The post-hoc a8 geometric clusters are not candidate topology in v1.

## Price and computation

Persistent stored values for a shared/private candidate are

$$
P=K_0(2+49+49)+\sum_g K_g(2+|S_g|+49).
$$

The per-document calibration state costs

$$
C=K_0+\sum_g K_g.
$$

Both $P$ and $C$ are reported; they are never collapsed into a favorable scalar.
Prediction multiply-adds and the 384-cell calibration solve are reported separately.
The unstructured control at a given structured price uses the largest rank whose
dense basis plus code does not exceed that persistent and per-document price.

Known continuous scale gauges and discrete factor permutations do not count as
distinct programs. Literal storage still counts every serialized value. After fitting,
the quotient-Jacobian test must explain all local null directions by registered CP or
block gauges before individual atoms can be described as separately editable.

## Scores and selection

For each candidate and seed report:

- masked training signed-response MSE;
- unconditional held-out-document MSE and signed correlation;
- calibrated non-anchor MSE and signed correlation;
- error separately for full/residual phase, six source owners, and every target owner;
- worst owner-pair normalized RMSE, not only a pooled average;
- persistent values, per-document values, calibration cells, and multiply-adds;
- optimizer health and seed range.

All scores are normalized only by the training-role response RMS fixed before
validation scoring. The output is the complete Pareto frontier under
`(persistent values, per-document values, calibrated validation MSE,
worst-owner-pair NRMSE)`. No knee is selected by eye. All nondominated candidates are
frozen before an EVAL lifecycle is implemented.

The hierarchy earns support only if at least one healthy shared/private point strictly
dominates both a global-only and an independent-only point in both price coordinates
and both error coordinates. Ties do not pass. This is a structure result, not a
terminal circuit result.

## Cheapest falsifiers and later consequence tests

The hierarchy is rejected as a useful v1 simplification if it fails the strict
dominance rule, if its apparent gain is carried by one optimizer seed, if any owner
pair fails while the pooled mean looks good, or if the quotient Jacobian has excess
nullity. A surviving frontier is then evaluated once on the sealed EVAL role without
rank or topology changes.

Even EVAL response prediction is insufficient for the project goal. A surviving atom
must next predict a fresh amplitude or direction intervention, enable extraction or
selective removal with less unrelated-target damage, and transport to a second domain.
These consequences, not local reconstruction alone, validate the simplicity measure.
