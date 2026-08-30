# Plain-language project update — 09:28 UTC, 2026-08-30

## The update, in one paragraph

The amount of the whole model we can honestly call reverse engineered has **not
increased** this hour. The important change is that the experiment needed to test a
new kind of simplification is now independently audited and queued to run. It will
measure a signed causal map between 49 known circuits: delete one direction, then
record which other circuits' prediction loss changes, separately on each document.
While another preregistered a8 job uses the GPU, I froze and tested the analysis that
will consume this map. The proposed simplification is a tensor-network hierarchy with
a small shared parent library and six component-private child libraries. A toy planted
example now verifies the contraction, price calculation, symmetry normalization, and
prediction of unmeasured response cells. This is real progress on the missing
measurement and analysis interface, but it is not yet a new explanation of model
behavior.

## What fraction is actually explained?

The strict balance sheet remains:

- **5.348245316%** of stored model values can be removed under the existing certified
  standard;
- named interventions account for **10.923302467%** of the model's deletion-induced
  cross-entropy change;
- **4.72714 nats, or 89.076697533%**, remain causally unnamed;
- **0 of 68** registered circuits yet pass extraction, selective removal, low
  collateral damage, and OOD transport together.

These percentages use different denominators and should not be added. The first is a
storage fraction. The second and third refer to prediction-loss changes caused by
deletion. A low-rank reconstruction or a successful optimizer does not move either
number.

## What is running, and how long has it taken?

The GPU is currently fitting learned directions for all 16 known a8 circuits at three
random seeds. At the 09:28 check it had completed 11 of the 48 fits in about 776
seconds; each completed fit was healthy. Its purpose is narrow: test whether a
geometry-derived grouping also organizes independently learned directions. It is not
being counted as a circuit explanation while incomplete.

The signed causal-response collection is queued directly behind it. The collection
will make exactly **12,400 outer model forwards** over 496 rows from 343 FIT source
documents. It has not begun, so there is no response number or runtime receipt yet.
The queue prevents two GPU experiments from silently colliding.

## What computation is the signed causal map?

Take a source circuit $s$ and its registered direction $d_s$. At the source component,
replace its native output vector $y$ by

$$
y' = y - \langle y,d_s\rangle d_s.
$$

This deletes only the component of $y$ parallel to $d_s$. The remainder of the model
runs normally. For every target circuit $t$ and document, compute

$$
R_{pstd}=
\operatorname{mean}_{i\in M_t}\Delta\mathrm{CE}_{psdi}
-\operatorname{mean}_{i\in O_t}\Delta\mathrm{CE}_{psdi}.
$$

Here:

- $p$ selects the full direction or the direction after removing the component's
  shared direction;
- $s$ is the deleted source circuit;
- $t$ is the target circuit whose token positions are being examined;
- $d$ is a source document;
- $M_t$ is the target circuit's registered member positions;
- $O_t$ is the target's off-slice comparison set;
- $\Delta\mathrm{CE}$ is intervened cross-entropy minus native cross-entropy.

Positive $R$ means the deleted source direction helped the target positions more than
the off-target comparison positions. Sign and document identity are retained, unlike
the earlier absolute concentration ratios.

The dense object has shape

$$
2\times49\times49\times343.
$$

It is not the model itself. It is a measured causal interface between already known
circuits.

## What simplification will be tested?

The candidate tensor program is

$$
R_{pstd}\approx
\sum_{k=1}^{K_0}A_{pk}B_{sk}C_{tk}H_{dk}
+\sum_g\mathbf 1[g(s)=g]
\sum_{k=1}^{K_g}A^{(g)}_{pk}B^{(g)}_{sk}C^{(g)}_{tk}H^{(g)}_{dk}.
$$

Each product is one reusable response atom:

- $A$ says whether it acts in the full or residual phase;
- $B$ says which source deletions use it;
- $C$ says which targets it changes;
- $H$ says how strongly it appears in each document.

The first sum is a library shared by all six owner components. The second gives each
owner component a private child library. The fixed indicator $\mathbf 1[g(s)=g]$ is
just a mask based on the source component. There is no data-dependent top-k decision,
so the computation remains sums and multilinear products: a genuine tensor-network
program.

This is a concrete version of the proposed DAG. The shared factors are parent nodes;
component-private factors are children. We are not assuming the recent post-hoc a8
clusters are nodes in this version.

## An important limitation: a free document code is not zero-shot prediction

$H_d$ is a fitted coordinate for a document. If a new document has never been seen,
we cannot know $H_d$ from the response tensor alone. Pretending otherwise would turn
reconstruction into fake OOD prediction.

The preregistration therefore separates two tests:

1. **Unconditional transport:** predict the average response on 114 held-out FIT
   documents using no response from those documents.
2. **Calibrated missing-cell prediction:** use a fixed panel of 384 response cells to
   infer a small document code, then predict all other source-target cells in that
   document.

The second is analogous to measuring a few sensors and predicting a much larger
system. It is useful if the number of sensors and code values is far smaller than the
full response array, but it is not zero-shot OOD behavior.

## How simplicity is priced and validated

For a shared/private program, persistent stored values are

$$
P=K_0(2+49+49)+\sum_gK_g(2+|S_g|+49).
$$

The temporary code for one new document costs

$$
C=K_0+\sum_gK_g.
$$

We report both rather than hiding them in one favorable score. We also report
calibration cells and multiply-adds. Candidates are compared with:

- global-only CP factors;
- six independent component factors;
- shared-parent plus private-child factors;
- an unstructured low-rank control matched by price;
- a per-cell mean control.

There are three optimizer seeds. The full seed range is retained. A pooled average is
not enough: error is also reported for every source-owner/target-owner pair, and the
worst pair is a selection coordinate. We retain the whole nondominated price/error
curve rather than choosing a visually attractive knee.

The hierarchy only wins this first test if one shared/private point is strictly no
more expensive and strictly more predictive than both global-only and independent-only
alternatives on held-out cells. Later, a quotient-Jacobian check must show that the
only parameter non-identifiabilities are known scale, basis, or permutation gauges.

Most importantly, even a winning response compression is not enough. Its factors must
later predict a fresh intervention, allow extraction or selective removal with less
unrelated damage, and transport to another domain. Those consequences are how the
simplicity definition earns validity beyond reconstruction.

## What the toy test proves—and does not prove

Six CPU tests pass in about six seconds. They verify:

1. signed response equals member mean minus off-target mean;
2. unsupported document cells are masked rather than imputed;
3. document and anchor splits depend only on frozen IDs, not responses;
4. shared and private tensor contractions route to the correct source groups;
5. literal parameter prices are counted correctly;
6. a planted shared/private program can be fit, canonicalized, and used to recover
   unmeasured cells from sufficient anchors; insufficient anchors fail closed.

The toy verifies algebra and code. It does **not** show that bilin18 has this
hierarchy. The queued response measurement is the falsifying experiment.

## Recent a8 result: useful, but not yet a discovered hierarchy

The earlier closed-form directions produced a geometry-only grouping whose aggregate
within-group causal similarity beat a size-matched permutation null
($p=0.0185$). But no individual cluster survived the appropriate multiple-test
qualification. The tightest geometric pair was actually causally anti-associated.
This is the confusing result worth retaining: coarse geometry contains some causal
signal, yet individual geometric neighbors need not be one editable mechanism.

That is exactly why the new factorization is optimized and judged in signed causal
response space rather than chosen from cosine similarity or HOSVD energy alone.

## Current priority order

1. **Finish the audited FIT response collection.** It supplies the missing signed,
   composable causal interface and is already queued.
2. **Fit the frozen response-factor curve.** Compare shared/private, global,
   independent, and dense controls using only FIT.
3. **Run quotient-Jacobian gauge accounting.** Reject factors that are not separately
   identifiable beyond known symmetries.
4. **Build and independently audit the separate EVAL loader only after the candidates
   are frozen.** EVAL must not influence topology or rank.
5. **Advance a mature bracket or successor circuit to the full extraction/removal/OOD
   standard.** This is the closest route to changing the 0/68 terminal count and gives
   a qualitatively different downstream observable for interpreting early layers.

Deprioritized for now are more M16 rank-one fits, a powerset over the arbitrary five
a8 circuits, geometry-only HOSVD/SAE selection, and treating the post-hoc a8 grouping
as an established DAG.

## Exact blockers

There is no missing data, dependency, or authorization blocker for FIT. The only
immediate wait is that a previously queued a8 job owns the GPU. The FIT process is
already waiting and will start automatically when the GPU is free.

Scientific blockers remain: no measured signed response tensor yet; no evidence yet
that the shared/private family beats controls; no factor-level fresh intervention;
and no terminal circuit. None requires user input at this point.

