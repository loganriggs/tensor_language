# Better feature sharing passes the coefficient test; the cheap graph still loses fidelity

21 September 2026, 12:37 UTC. **Stage one improved: dictionaries reused by pairs of components beat a single common dictionary, with the same additional optimization budget. Stage two remains the obstacle: converting those quadratic forms into a cheap executable graph loses too much accuracy.**

This is the smaller six-read, three-component problem from the [overall review](research_update_2026-09-21_0104_full_coverage_and_shared_baselines.md). Native earlier and later states are still supplied. Neither the broad folded computation nor a standalone circuit is solved.

**Stage one: changing who shares a feature helped**

The comparison retains 352 input dimensions per component pair. The old layout has 128 directions common to all three pairs and 224 private directions per pair. The new layout has three 64-direction dictionaries, shared by consumers 1–2, 1–3 and 2–3, plus the same private widths.

Both layouts received another 1,800 Muon steps from initially identical functions. The new layout also had a random start in each fitting geometry. All six fits completed in 397.12 seconds.

| Matched warm-start comparison | Old common dictionary | Pairwise dictionaries |
|---|---:|---:|
| Covariance-shaped fitting error | 7.918% | **7.577%** |
| Native-isotropic fitting error | 51.117% | **49.082%** |

Both improvements pass the registered 1% relative-improvement test. The primary covariance fit also passes the two original coefficient limits: **7.577% covariance-shaped error versus a 7.734% limit**, and **58.24% native-isotropic error versus a 60.79% limit**.

But coefficient fidelity is not the whole goal. Its three component errors are **2.13%, 2.10% and 10.55%**; the original baseline gives 1.87%, 2.05% and 8.81%. Individual value requirements still fail. Fixed-later-state input-derivative requirements pass. These measurements use the previously examined 448 states, not fresh OOD text.

The unrestricted dense cores would use 1,548,240 source multiplications, exceeding the original baseline's 1,330,560. Passing the coefficient tests therefore does not establish a simpler executable circuit.

**Stage two: an exact rewrite works, but learning new product wiring remains difficult**

We implemented a graph with literal pairwise reuse, including physical storage and arithmetic counts. Converting the four earlier shared/private programs into this representation preserves their outputs to numerical precision. Five planted product programs also pass independent loss and gradient checks.

With known input subspaces, random fitting recovers all five planted graphs using the old whole-block product assignment. A variant that puts shared and private inputs into opposite slots of each product block recovers only **two of five**. That miss is retained; we did not treat the new topology as optimizer-validated or launch a native mixed-topology fit on that basis.

**A direct algebraic conversion produces a cheap graph—and exposes its error**

Instead of another random retry, we partitioned each quadratic form into shared coordinates $s$ and private coordinates $v$:

$$
Q_o=\begin{bmatrix}A_o&B_o\\B_o^\top&C_o\end{bmatrix}.
$$

Here $o$ selects one of the component pair's two scalar reads. The cross block $B_o$ describes interactions between shared and private inputs.

We chose one matrix $E$ for both reads by minimizing

$$
\sum_o\|B_o-E C_o\|_F^2.
$$

This gives the program

$$
s^\top(A_o-E C_o E^\top)s
+(v+E^\top s)^\top C_o(v+E^\top s).
$$

It preserves both diagonal blocks exactly. Its error within the chosen input span is the mismatch between $B_o$ and $E C_o$. Each pair of remaining quadratic forms is compiled into shared signed products by the existing matrix-pencil compiler.

The primary compiled graph has **1,056 products, 1,058,124 stored coefficients and 1,047,648 source multiplications**—a **21.26% arithmetic reduction** from the original baseline. Fixed-product output coefficients were then refitted.

| Primary covariance fit | Dense stage-one forms | Cheap compiled graph |
|---|---:|---:|
| Covariance coefficient error | 7.577% | **11.150%** |
| Native coefficient error | 58.24% | **63.40%** |
| Component 3 error | 10.55% | **14.21%** |

The arithmetic requirement passes; fidelity fails. Execution replay agrees to about $2.4\times10^{-15}$, so the primary's miss is an approximation failure.

A separate covariance restart failed the compiler's stricter matrix-reconstruction check: errors around $3.5$–$3.8\times10^{-10}$ exceeded its $10^{-10}$ limit. That run is recorded as an instrument failure, not a successful reconstruction. The threshold was not relaxed.

**Can a few additional cross products repair the graph?**

We tested two different cost models, because charging for newly computed directions and reusing existing activations are different operations.

For the fixed, unrefitted completion with both diagonal blocks retained, an optimistic singular-value bound requires at least **199** additional shared-times-private atoms to reach the covariance limit. Independently computing dense left and right projections would leave budget for only **34**. This rejects that particular dense correction strategy. It does not rule out sparse projections, reuse, refitting the old terms, or arbitrary arithmetic graphs.

Products of already-computed activations are cheaper. We can afford **5,600** additional shared/private coordinate products at three multiplications each: one product and two output weights. This reaches, but does not exceed, the original 20%-saving budget.

| Fixed selection of 5,600 positions | Covariance error after coefficient refit |
|---|---:|
| Largest normalized coefficient groups | 10.96% |
| Largest correlations with the residual | **10.70%** |
| Required limit | **7.734%** |

Both linear solves converged to the registered normal-residual tolerance. Independent small dense solves check the conjugate-gradient implementation. Thus these masks fail despite successful coefficient fitting. This is not a proof that every mask of the same size fails. The shared-core update and executable recompilation were not pursued for these unsuccessful candidates.

**The next distinction to test**

The algebraic conversion fixed the private forms $C_o$ while choosing the coupling $E$. That restriction was convenient, but it is not required by the final graph's cost. We can instead optimize

$$
\sum_o\left[
\|C_o-C_o^{\rm target}\|_F^2
+2\|E C_o-B_o^{\rm target}\|_F^2
\right].
$$

With the shared diagonal form adjusted exactly, this is the remaining within-span coefficient objective. For fixed $C_o$, the best $E$ is a least-squares solve. For fixed $E$, each best symmetric $C_o$ solves

$$
\left(\tfrac12 I+E^\top E\right)C_o
+C_o\left(\tfrac12 I+E^\top E\right)
=C_o^{\rm target}+E^\top B_o^{\rm target}
+(B_o^{\rm target})^\top E.
$$

We implemented these alternating exact updates. Five independent dense-system checks pass and the objective decreases at every tested update. Joint global optimality is not guaranteed. No native result from this refinement is available at this report revision.

This next test keeps the graph family and cost fixed while removing an unnecessary fitting restriction. It will distinguish a poor algebraic initialization from a stronger limitation of this shared/private graph family.

**Evidence**

- [Six native subspace fits and matched comparisons](../../direct_tensor_match/PAIRWISE_SUBSPACE_NATIVE_V1.json), [independent component audit](../../direct_tensor_match/PAIRWISE_SUBSPACE_NATIVE_AUDIT_V1.json).
- [Actual graph rewrite and physical cost checks](../../direct_tensor_match/PAIRWISE_READER_GRAPH_PREFLIGHT_V1.json).
- [Whole-block toy recovery](../../direct_tensor_match/FIXED_SUPPORT_PRODUCT_TOY_V1.json), [mixed-block recovery failure](../../direct_tensor_match/FIXED_SUPPORT_PRODUCT_TOY_V2.json).
- [Algebraic conversion results, including compiler failure](../../direct_tensor_match/SHARED_PRIVATE_COMPLETION_V1.json).
- [Dense-correction rank bound](../../direct_tensor_match/COMPLETION_CROSS_RANK_V1.json), [magnitude-mask refit](../../direct_tensor_match/REUSED_CROSS_REFIT_V1.json), [residual-mask refit](../../direct_tensor_match/REUSED_CROSS_REFIT_V2.json).
- [Alternating exact-solve controls](../../direct_tensor_match/COMMON_PRIVATE_ALS_PREFLIGHT_V1.json).

No candidate here has been adopted as a circuit. Fresh OOD prediction, stable feature identity, selective manipulation, reuse across tasks and extraction without native intermediate states remain outstanding.
