# A consistent shared-reader interface, and its specificity control

11 September 2026. **Changing consumer scope repairs much of the apparent
cross-start disagreement, but a control shows the resulting agreement is common
among other weight-derived readers.** The exact rewrite is useful infrastructure;
it does not identify these readers as special circuits.

The completed graph fits had two close input-reader pairs: spectral parent1 versus
native parent6, and spectral parent9 versus native parent4. Input cosines exceeded
0.98, while their named-node removal functions agreed at only0.4405 and0.4888.
The prior scope audit showed why: named-node deletion only changes explicitly
declared consumers, while other groups can read the same input direction.

## An exact change of consumers

Fix one spectral reader $u$ and use it in both graphs. For every quadratic group,
choose an orthonormal private basis $V_g$ perpendicular to $u$, retaining all
directions needed to span the projected old input readers. Then

$$
z=u^\top x,\qquad y_g=V_g^\top x,
$$

$$
x^\top Q_gx=\alpha_g z^2+z\,\beta_g^\top y_g+y_g^\top H_gy_g,
$$

$$
\alpha_g=u^\top Q_gu,\qquad
\beta_g=2V_g^\top Q_gu,\qquad H_g=V_g^\top Q_gV_g.
$$

Diagonalize the private $H_g$ as in the existing graph executor. Compute the
shared read and its square once. Unlike the previous approximate merge, permit
all16 private directions plus the shared reader, rather than forcing the total
group input dimension to remain16. This preserves the whole fitted function.
Setting $z=0$ now has the same meaning as replacing the quadratic layer's input
by $(I-uu^\top)x$, because every private read is perpendicular to $u$.

That equality is local to this layer's quadratic input. It is not automatically
an upstream intervention through RMS normalization or a claim about native behavior.

The first implementation exposed a rank-deficient QR problem: groups already
containing $u$ have only15 projected input directions, and unrestricted QR
completion can reintroduce $u$ as the sixteenth private direction. This caused
double-counting and up to10% whole-function execution error. The
[V1 failure](GLOBAL_READER_REBASE_V1_FAILURE.json) and source are preserved.
[V2](global_reader_rebase_v2.py) detects rank by SVD and completes the basis
explicitly inside $u^\perp$. No readers, thresholds, or experimental selections
changed in the repair.

The [repaired result](GLOBAL_READER_REBASE_V2_RESULT.json) passes all registered
bars. Thin coefficient, executor, and removal checks agree within5.17e-14;
private/shared overlap is at most4.95e-15. With the same global reader interface:

| Fixed spectral reader | Cross-start removal-function cosine | Relative function difference | Groups above1% own-energy threshold |
|---|---:|---:|---|
| 1 | 0.9538 | 30.31% | 15 / 17 |
| 9 | 0.9647 | 26.51% | 15 / 15 |

High cosine does not mean identical effects, as the relative differences show.
These group counts measure algebraic overlap, not semantic consumers.

The exact rewrite costs1,256,640 floating coefficients,1025 linear reads and1089
variable products. The original64-group rank16 representation costs1,254,400,
1024 and1024 respectively. Thus the rewrite adds0.179% coefficients and65 products
relative to that representation; it is also larger than the compressed shared
graphs. No compression or adoption claim follows.

## Is the apparent stability distinctive?

A positive result could arise because many projections of the two fitted layers
already agree. The [executed control](GLOBAL_READER_AGREEMENT_CONTROLS_V1.json)
uses the pre-existing256 converged-square readers. Exclude directions with absolute
cosine at least0.95 to the anchor, then choose16 readers closest in log mean
cross-start projected energy. This matches contribution scale approximately;
selected energy ratios are0.843–1.188 and0.869–1.144 for the two anchors.

| Reader | Anchor agreement | Control mean | Control95th percentile |
|---|---:|---:|---:|
| 1 | 0.9538 | 0.9670 | 0.9879 |
| 9 | 0.9647 | 0.9683 | 0.9879 |

Both specificity bars miss: neither anchor exceeds its control mean by0.02,
and neither exceeds the control95th percentile. Batched projected-inner scores
agree with explicit partner matrices within3.18e-15. The whole fitted functions
have cosine0.9076; their high agreement on many strong reader projections is
therefore a real but broad property.

The existing alias census also records reader1 close to old square126
(input cosine0.9780) and reader9 close to old square204 (0.9881). Their complete
node functions are not aliases of those single squares, but these are not newly
discovered input directions. [Prior census](SHARED_READER_POSTFIT_ALIASES_V1_SPECTRAL.json).

The conclusion is narrower than either initial extreme: inconsistent consumers
did hide comparable computation, but defining consistent consumers does not by
itself establish a distinguished circuit. The frozen native branch screen remains
queued with its original named-node intervention. Its behavioral outcome and the
retained-history optimization comparison should guide the next experiment.

## Two parents: preserving the other read changes the intervention

The [two-parent extension](global_multi_parent_rebase_v1.py) now exposes both
fixed readers together, globally, in each fitted graph. It preserves the entire
quadratic function and reuses the rank-deficiency repair.

Let $R$ have the two readers as rows. Its dual columns are

$$
V=R^\top(RR^\top)^{-1},\qquad RV=I.
$$

Write $z=Rx$ and keep private reads perpendicular to the row space of $R$.
Setting only $z_i$ to zero while holding the other read fixed corresponds to

$$
x'=x-v_i(r_i^\top x).
$$

This is an **oblique projection**: the removal direction $v_i$ preserves the
other reader, rather than necessarily point along $r_i$. The identity
$r_j^\top v_i=0$ for $j\ne i$ makes these deletions commute. Removing both gives
the orthogonal projection off their joint span:

$$
x'=\left[I-R^\top(RR^\top)^{-1}R\right]x.
$$

Individually subtracting $r_i(r_i^\top x)$ instead changes the other reader
whenever their inner product is nonzero. These actual readers have cosine0.06898
and Gram condition1.148. Nevertheless, their single-removal outcomes differ by
7.4–10.2% relative output norm between the orthogonal and node-preserving
definitions. Small reader overlap is not an exact interchange contract.

The [receipt](GLOBAL_MULTI_PARENT_REBASE_V1_RESULT.json) passes all registered
bars: coefficient/executor/dual/input-intervention/commutation checks agree
within7.05e-14; scope differences exceed1% in both fits; joint-removal cross-start
cosine is0.9587. The preceding specificity control still prevents interpreting
this agreement alone as a distinguished circuit.

The graph computes $z_0^2,z_0z_1,z_1^2$ once for their output consumers, alongside
each group's private quadratics and two shared/private mixed terms. Summing the
two individual removals counts $z_0z_1$ twice. Subtracting its vector contribution
once gives the exact joint effect. The naive sum has8.77% / 8.89% relative
execution error in the two fits. This local polynomial identity does not make
native RMS, tanh, or CE effects additive; they require downstream evaluation.

The two-parent version costs1,258,944 floating coefficients,1026 linear reads,
and1155 variable products, versus1,254,400/1024/1024 in the original64-group
representation. It provides a precise executable interface and adds cost;
it is not yet a simpler adopted decomposition or a behavioral result.
