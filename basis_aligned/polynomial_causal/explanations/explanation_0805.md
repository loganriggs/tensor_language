# Current bilin18 explanation — 08:05 UTC

**Date:** 2026-08-30  
**What is new here:** sections 3–7 explain the signed causal-response experiment,
the integrity work completed this hour, the completed M16 seed test, and the next
decision point. This is an implementation update, not a new whole-model result.

## 1. The honest short answer

The strict whole-model ledger has not changed:

| Quantity | Current value | Plain meaning |
|---|---:|---|
| Certified removable stored values | 29,196,288 / 545,904,054 = **5.348%** | Values already removed inside a tested executable replacement |
| Deletion-based CE assigned to named mechanisms | 0.57968 / 5.30682 = **10.923%** | Measured behavioral damage for which we have a named causal account |
| Unexplained deletion-based CE | **4.72714 nat = 89.077%** | The largest quantitative gap |
| Terminal circuits passing extraction, selective removal, and OOD together | **0 / 68** | No circuit yet passes every desired test |

The highest-return current move is still to build a signed causal coordinate system
over many known circuits. The mathematical backend and most FIT artifact validation
now exist, but the real model experiment has deliberately not run yet. An independent
audit is checking the exact committed authority boundary before we connect the model
executor.

## 2. What object are we trying to measure?

Suppose circuit $i$ appears to write through model component $c_i$, such as attention
8 or MLP 16. On FIT documents we estimate a direction $u_i$ in that component's
1,152-dimensional output space. It is the normalized contrast

$$
u_i =
\frac{\mathbb E[w_{c_i}\mid\text{member positions of }i]
-\mathbb E[w_{c_i}\mid\text{off-slice positions of }i]}
{\left\|\mathbb E[w_{c_i}\mid i]
-\mathbb E[w_{c_i}\mid\text{off-slice positions of }i]\right\|_2},
$$

where $w_{c_i}$ is that component's residual-stream write. At a model position, the
intervention removes only the rank-one projection along $u_i$:

$$
w'_{c_i}=w_{c_i}-(w_{c_i}^{T}u_i)u_i.
$$

This is a fine-grained deletion. It does not zero the whole attention or MLP module.

For every source circuit $i$, target circuit $j$, document $d$, and direction phase
$p$, we then measure

$$
R_{pijd}=
\sum_{x\in\text{target }j\text{ in document }d}
\left[CE_x(\text{delete }u_{p,i})-CE_x(\text{native})\right].
$$

Here $CE_x$ is next-token cross-entropy at position $x$. Positive $R_{pijd}$ means
deleting source direction $i$ hurts target $j$ on document $d$; a negative value means
the deletion helps. We retain signed sums, absolute sums, and counts, rather than a
single concentration ratio, because sign, magnitude, document variation, and
additivity are all needed for causal factorization.

There are two phases:

1. **full:** use the entire fitted direction $u_i$;
2. **residual:** subtract the leading direction shared by circuits at the same
   component, then normalize what remains.

With 49 registered sources and 49 targets, this creates a $49\times49$ causal
interaction table for each phase, with document-level evidence underneath it. The
FIT side uses 496 rows from exactly 343 source documents. EVAL will later use 504 rows
from 345 disjoint documents, but EVAL is not opened while choosing a factorization.

## 3. What is this meant to accomplish?

Weight cosine and HOSVD can say that two directions look alike. Recent measurements
show they do not reliably say whether the directions *do the same thing*: across six
components, geometry-versus-causal correlations change sign, and after removing a
shared geometric direction their magnitude is at most 0.1773.

The response tensor instead lets us ask operational questions:

- Does one shared source factor predict effects on many target circuits?
- Does each circuit need a private branch after the shared factor?
- Is there a sparse hierarchy or DAG whose parents predict several children?
- Does a factor fitted on FIT documents predict signed effects on untouched EVAL
  documents?
- Can deleting one fitted factor remove its intended targets without collateral
  damage to other targets?

The proposed comparison is between independent factors, shared-plus-private block
terms, and parameter-matched dense controls. A structure is not accepted merely
because it reconstructs FIT. It has to predict held-out signed causal effects per
literal storage/compute price and support selective interventions.

## 4. What was completed this hour?

The canonical FIT inputs are now reconstructed from frozen parents rather than
accepted from a caller. Exact identities include:

| Bound object | Shape | SHA-256 role |
|---|---:|---|
| model rows | `[1000,257]` int64 | fixes every token passed to the model |
| FIT row indices | `[496]` int64 | fixes the training rows |
| FIT document IDs | `[343]` int64 | prevents silently substituting another document set |
| circuit order | 49 `(component, tag)` pairs | fixes tensor axes |
| member/slice masks | 49 pairs of 256,000-position masks | fixes every target definition |

The original census rows contain 513 tokens, but every registered mask covers 256
next-token positions. Therefore this collector must pass columns 0 through 256: 257
tokens produce exactly 256 predictions. Passing all 513 would compute an unregistered
second half.

All direction means, singular-value decompositions, and residual projections are now
computed on CPU in float64. Each final unit direction is cast once to the model's
float32 dtype. This avoids fitting a shared direction to already-rounded vectors.

The event ledger records exactly one count for every

$$
(\text{phase},\text{source},\text{batch})
\in 2\times49\times124
$$

projection event and every

$$
(\text{captured component},\text{batch})\in6\times124
$$

write-capture event. Together with native attention/MLP call counts, this detects a
missing, duplicated, or misrouted intervention.

The first attempt used a one-use Python “capability.” Independent attack showed that
it could be forged with low-level Python object construction. That abstraction is now
removed. The simpler rule is: the source-closed lifecycle is the only production
executor; its private input reconstruction is guarded before and after parent access;
and published artifacts are joined by exact hashes. The FIT publisher returns only
the artifact digest, not directions or an EVAL-capable program.

Commit `b117acbb` introduced this boundary; `ca31055c` canonicalized short and full
commit names; `1c900444` closed four independently reproduced identity/numerical/race
defects; and `4b0cd7fa` made the FIT-only claim, CE sign, and off-mask definition exact
semantic contract fields. The current wider closed backend suite passes **57/57**.
The resulting 21-file committed source closure replays to
`1dc3f438f1425ef4ed2799f08eae4da2940bcec66332d2653c758a9186e611bd`.
This is concrete integrity progress, but it has not increased the explained fraction.

## 5. M16 seed result

The M16 rank-one DAS test was repeated over three optimizer seeds and held-out rows.
“Selectivity margin” means damage to the circuit's own member positions divided by
average damage to the other registered M16 circuits; the preregistered bar is 1.10.

| Circuit | Mean margin | Standard deviation | Interpretation |
|---|---:|---:|---|
| `r.1.1.1` | 1.151 | 0.046 | above the bar in all three seeds, but modest margin |
| `r.1.1.2` | 0.969 | 0.029 | no selectivity |
| `r.1.2` | 0.947 | 0.032 | no selectivity |
| `r.1.2.0` | 0.799 | 0.091 | no selectivity |
| `r.1.2.1` | 0.995 | 0.037 | no stable selectivity |
| `r.6.2.2` | 1.198 | 0.319 | mean crosses the bar, but extremely seed-sensitive |

The earlier `r.1.1.1` result cleared its bar by only 0.001, while its observed seed
standard deviation is 0.046. That single-seed margin was not robust. The completed
test also corrects “zero of the other five” to “one other circuit is seed-dependent.”
This branch is now low priority: it has not produced a stable family of editable M16
children.

## 6. Did the mathematics help?

Yes, in two concrete ways.

First, the signed response tensor converts “shared versus private hierarchy” from a
visual similarity claim into a held-out prediction problem. A block-term or DAG model
must predict new documents' causal effects, not only compress vectors.

Second, a toy quotient-Jacobian instrument now distinguishes true parameter redundancy
from gauge freedom. For a map with parameters $\theta$ and represented tensor
$T(\theta)$, its Jacobian

$$
J=\frac{\partial\operatorname{vec}T}{\partial\theta}
$$

tells us how many independent infinitesimal changes the parameters can make. A
45-parameter rank-three matrix product had rank 36 and exactly nine known gauge-null
directions. A 45-parameter rank-three CP tensor had rank 39 and exactly six scaling
gauge directions. Duplicating a CP component reduced rank to 26, so the test correctly
detected extra non-identifiability rather than calling all nullity “symmetry.”

After fitting a response factorization, we can apply this test to reject a supposedly
simple program whose parameters contain unexplained, basis-dependent null directions.
This is local first-order identifiability, not yet semantic interpretation or a global
minimality proof.

## 7. Current blockers

There is no data or GPU blocker. The GPU is currently free. The blocker is launch
integrity: FIT must be one source-closed transaction that freezes authority before
parent/model loads, protects the exact checkpoint and model state, owns the collector,
publishes bundle and manifest create-only, and publishes a hash-bound receipt last.

Independent outcome-blind replay of exact commit `4b0cd7fa` found no remaining P0 in
this input/authority/bundle boundary: 54 safe tests passed and the three tests that
deserialize real row parents were deliberately omitted. This is a conditional GO to
retain the boundary and implement the next layer, **not** scientific execution
authority. The missing model/collector/manifest/receipt-last transaction must be
implemented and independently audited before the 12,400-forward FIT collection runs.

## 8. Ranked next work

1. **Finish the sealed FIT executor.** Highest information gain because it creates the
   missing signed causal interface for all 49 sources and targets.
2. **Fit shared-plus-private block terms on FIT and predict EVAL.** This directly tests
   whether a sparse DAG is simpler in a useful sense.
3. **Apply quotient-Jacobian gauge accounting to candidate factors.** Reject apparent
   compression caused by non-identifiability rather than a smaller causal program.
4. **Finish one already-developed terminal circuit lifecycle, bracket or successor.**
   This is the shortest independent route to a circuit passing extraction, removal,
   collateral, and OOD tests.
5. **Run suffix-Fisher/observability triage.** Identify early-write directions that the
   downstream model can actually distinguish, then compress modulo the downstream
   nullspace.

Further M16 rank-one fitting, geometry-only HOSVD/SAE selection, and factorizing
unsigned concentration ratios are pruned for now. They either repeat completed work
or do not predict the causal properties needed for extraction and selective editing.
