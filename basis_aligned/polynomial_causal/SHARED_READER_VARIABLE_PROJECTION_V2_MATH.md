# Joint reader fitting and a false convergence signal

We now have a checked objective that jointly moves shared input readers, private input spaces, and output directions while solving all interaction coefficients at each step. It fits every near-initialized planted example. Independent starts are still unreliable. A red-team test found a specific optimization failure: raw parameters can grow into the millions without changing the represented function, making the raw gradient falsely satisfy a stopping tolerance. Re-encoding the same function exposes the missing gradient and rescues one failed start.

This is progress toward fitting shared arithmetic during decomposition. It is not yet a robust general graph-discovery algorithm or a native behavioral circuit result.

## Objective and gradient

For each group, QR factorization of its shared readers and projected private readers gives an orthonormal basis $B_g(\theta)$. Normalize its output vector to $c_g(\theta)$. The numerical parameters $\theta$ include the global shared reader bank, private reader matrices and output vectors. The graph incidence is fixed during one continuous fit.

$$
\mathcal L(\theta)=\min_{H_1,\ldots,H_m}
\frac{\left\|T-\sum_g c_g(\theta)\otimes
B_g(\theta)H_gB_g(\theta)^T\right\|_F^2
+\eta\sum_g\|H_g\|_F^2}{\|T\|_F^2}.
$$

The symmetric cores $H_g$ are eliminated by the previously checked matrix-free positive-definite solve. Its true relative normal residual must be at most $10^{-11}$ during these checks. The penalty is the same whole-group tensor-energy penalty because the input bases are orthonormal and output vectors unit norm.

At the conditional optimum, the core derivative vanishes. Therefore the envelope gradient is the partial derivative with respect to $\theta$, holding the solved cores fixed. We evaluate the complete folded-tensor coefficient gradient using implicit product contractions, then propagate it only through the QR and output-normalization maps. We do not differentiate through a finite sequence of conjugate-gradient iterations.

The small dense check gives relative gradient error $3.50\times10^{-15}$ and packed directional finite-difference error $1.96\times10^{-9}$. Emitted-graph and signed row-rescaling checks are below $7.2\times10^{-12}$. A V1 planted-control setup typo failed before evaluating the kernel and is preserved separately. V2 also makes `physical()` safely evaluate an uncached point before entering its no-gradient emission path.

## Planted recovery at a supplied topology

The test uses three rank-3 quadratic groups in nine input dimensions and five output dimensions. One shared reader feeds all three groups; another feeds two groups. Their private directions and output writes differ. The topology is supplied, so this tests numerical fitting, not discovery of the number of nodes or their consumers.

With penalty $10^{-6}$, each start received at most1,000 L-BFGS iterations/60seconds. All four small perturbations of the true representation recover the function with relative coefficient error about $4.2\times10^{-6}$. All four independent starts miss the $10^{-3}$ recovery bar:

| Seed | Independent-start error | Raw maximum packed gradient | Termination |
|---|---:|---:|---|
| 2914 | 0.10764 | $5.65\times10^{-5}$ | Iteration limit |
| 2915 | 0.03704 | $9.63\times10^{-9}$ | Relative objective reduction |
| 2916 | 0.01735 | $8.55\times10^{-10}$ | Gradient tolerance |
| 2917 | 0.12532 | $7.21\times10^{-10}$ | Gradient tolerance |

The registered near-start recovery bar holds; the independent-start bar fails0/4 versus at least2/4. A tiny gradient at these raw coordinates is not sufficient evidence of a bad local minimum, as the following test shows.

## Discrete consumer moves and the same-topology control

We replayed seed2916 exactly, then changed the second parent's consumer pair from groups[0,2] to[0,1] or[1,2]. Each move rebuilt the approximate graph at the same rank and cost, followed by joint reader/writer fitting and core elimination. Both alternative assignments failed to improve the error; their final errors were0.06335 and0.02957. The registered topology-move improvement and recovery bars therefore fail.

The same-topology control unexpectedly recovered to $4.17\times10^{-6}$. Its starting function and objective were unchanged to numerical precision. That control reset the numerical coordinates and L-BFGS history, so we tested those explanations separately rather than crediting topology search.

## Coordinate scaling creates a false stopping signal

Normalized readers have an exact scaling freedom:

$$
\frac{\alpha a}{\|\alpha a\|}
=\operatorname{sign}(\alpha)\frac{a}{\|a\|}.
$$

The corresponding sign can be absorbed into the core. Thus growing a raw reader's norm need not change the represented function. Its derivative with respect to raw coordinates, however, contains an inverse norm. Output normalization and the QR parameterization have related coordinate freedoms.

In seed2916, raw shared-reader norms had grown to2.3–5.3million; private-reader norms reached0.8–3.2million, and output-vector norms0.9–3.0million. Seed2917 reached output norms above21million. These are raw coordinate scales, not growing physical group functions or a finding about native weights.

Restarting L-BFGS at seed2916's same raw point takes **zero steps**: it immediately accepts the $8.55\times10^{-10}$ gradient. Re-encoding the **same function and topology** with normalized readers, orthonormal private bases, and unit output directions gives a maximum gradient of **0.0007878**. The function changes by only $1.28\times10^{-15}$. The subsequent fit recovers the function. This distinguishes coordinate conditioning from merely clearing optimizer history.

Across the four independent starts, at most four solve/re-encode cycles recover only one. The2/4 recovery bar still fails. The other three retain errors0.08498,0.03704 and0.09841. Their final normalized-coordinate gradients and objective-stopping behavior remain recorded; no global recovery guarantee follows.

The controller should bound or regularly reset these redundant raw scales and check stopping after a consistent re-encoding. A normalized-coordinate check is a practical repair, not a proof of an intrinsic, globally sufficient stationarity test. We must continue separating function recovery, local stopping and graph identification.

## Native status and next managed measurement

The earlier output-eliminated LL1 comparison completed at16:01:20UTC. It captured11.8891%/11.8510% of coefficient energy. Both20-minute arms failed their registered local convergence criteria. Cross-start whole-function cosine0.90849 and7/64 matched groups above0.8 also fail the stability bar of0.95 plus16groups. These are modest improvements over the short pilots, not stable identified circuits. They use the older parameterization, not the new all-core shared-reader objective.

The new managed [native preflight](SHARED_READER_PROJECTED_PREFLIGHT_V1_PREREGISTRATION.md) measures its GPU gradients, emitted graph, memory, latency and agreement with the completed CPU core optimum. It performs no optimizer steps or model-body forwards. Source SHA256 is `531cd43147a8fc667959a6814fcdaa4d6811467bb2a78f9724cd931393e31e20`;18 dependencies are bound. Its outcome determines the next fitting budget and the controller must account for the demonstrated scaling issue.

Primary artifacts: [gradient controls](SHARED_READER_VARIABLE_PROJECTION_V2_CONTROL.json), [eight planted starts](SHARED_READER_VARIABLE_PROJECTION_V2_PLANTED.json), [consumer moves](SHARED_READER_CONSUMER_MOVES_V1_AUDIT.json), [coordinate restart audit](SHARED_READER_GAUGE_RESTART_V1_AUDIT.json), [completed native LL1 comparison](PROJECTED_LL1_CONVERGENCE_V3_RESULT.json). Code: [objective](shared_reader_variable_projection_v1.py), [emission API repair](shared_reader_variable_projection_v2.py), [coordinate audit](shared_reader_gauge_restart_v1.py).


## Native preflight completed; bounded local fits started

The native preflight completed16:08:42 with all bars held: relative finite-difference error at most1.85e-7, executor error at most1.3e-15, true inner residual at most7.7e-12, CPU objective agreement at most2.3e-16, median evaluation0.280/0.281seconds and peak allocation0.378/0.396GiB. [Receipt](SHARED_READER_PROJECTED_PREFLIGHT_V1_RESULT.json).

The bounded controller constrains raw entries to[-1,1] and re-encodes every200acceptediterations or solver termination. Every normalized reader/private basis/unitwriter representation remains feasible. It preserves the same starting functions and prevents raw scale blow-up, but independent planted recovery remains1/4; the2/4bar fails. The JSON C field checks reporting coverage only and provides no scientific recovery evidence. Two failed fresh-coordinate fits have Hessian minimum eigenvalues about−4.2e-8/−6.7e-8, below the credibility of the registered−1e-4negative-curvature test; that proposed saddle escape did not trigger. This is not a proof of global or strict local optimality. [Bounded control](BOUNDED_SHARED_READER_FIT_V1_CONTROL.json), [curvature audit](SHARED_READER_CURVATURE_V1_AUDIT.json).

Four matched native20-minute local-refinement arms are now managed: original versus compatible graph for both pilot starts. Spectral-original started16:22:02; the other three are queued. They use the same full weight metric and penalty, with fresh-coordinate stopping and explicit matched capture/cost comparisons. [Protocol](SHARED_READER_JOINT_FIT_V1_PREREGISTRATION.md). No native result from these fits is claimed yet.


## First completed joint fit: convergence still misses; group changes cancel

Spectral-original completed at16:42:07UTC. Its numeric/executor checks and registered objective-improvement bar hold, but fresh-coordinate convergence fails: gradient0.000183719 versus1e-7bar,20-step relative progress3.50e-5 versus1e-6bar. Capture is11.8838721%, slightly below the older output-projected20-minute spectral fit's11.8890983%. Penalized objectives are0.884187035 and0.884135548 respectively. Neither converged. The stronger convergence instrument is not evidence of a faster optimizer. Preserve the older fit as a secondary baseline when evaluating the shared graph. [Completed arm](SHARED_READER_JOINT_FIT_V1_SPECTRAL_ORIGINAL.json), [earlier matched-time observations](JOINT_READER_PROGRESS_COMPARISON_2026-09-11_1634.json).

An existing-tools CPU comparison matched the64signed group tensors using their exact full-unembedding coefficient inner products. Whole-function cosine0.97049 passes the registered0.95bar;54/64groups exceed0.8, passing48/64. Independent CP-inner replay agrees to3.51e-16. These are the same starting family under two optimizers, not independent-start recovery. Their full fitted functions nevertheless differ by24.29% relative to the older fit's norm. Cosine is not a relative-error guarantee. [Comparison and source hashes](JOINT_READER_OPTIMIZER_GROUP_COMPARISON_V1.json).

Let the matched group difference be

$$
\Delta_g=T_g^{\mathrm{new}}-T_g^{\mathrm{old}},\qquad
K_{gh}=\langle\Delta_g,\Delta_h\rangle.
$$

The calculation reuses the group Gram matrices:

$$
K=G_{\mathrm{old,old}}+G_{\mathrm{new,new}}
-G_{\mathrm{old,new}}-G_{\mathrm{old,new}}^\top,
\qquad
\left\|\sum_g\Delta_g\right\|^2=\mathbf1^\top K\mathbf1.
$$

The sum of individual difference energies is11.995times the total difference energy: changes in different groups strongly cancel. The10groups below0.8cosine alone contribute a difference with2.132times total energy; the other54contribute2.564times; the cross term is−3.696times. These are interacting energies, not additive explained fractions. Group9 even changes to cosine−0.128 despite both whole fits having almost the same reconstruction score.

The circuit consequence is specific: reconstruction-score similarity cannot justify substituting individual LL1 groups or attaching stable circuit identities to them. The next shared-graph results must be examined as executable node interventions, using the already implemented pair-corrected removal scorer. Whether the shared topology stabilizes these interfaces remains unresolved. Spectral-graph is now running; the two native-start arms remain managed and queued. No behavioral claim or general DAG-search result follows from this fit.

## Native metric measurement and retained-history comparison

The16:51review's fixed-core horizontal-reader metric was measured on the completed spectral-original graph. Median block condition32.92, maximum389.75;0/64exceed1e4, and the global reader-metric ratio1157.90misses1e6. Writer-metric ratio8.02. Dense directional checks agree with the derived metric to2.3e-15. These are moderate-to-large local scale differences, not the extreme self-conditioning predicted; cross-group coupling and the eliminated-core Hessian remain unmeasured. [Receipt](LL1_NATIVE_CORE_CONDITIONING_V1.json).

A simpler numerical hypothesis is that mandatory200-step re-encodings discard useful L-BFGS history. One matched20-minute spectral-original run is now queued with the existing controller's iteration limit set to1000000, retaining history until solver termination or budget. Bounds, core solves and fresh-coordinate checks remain. The C bar now asks for1e-4objective advantage over the completed200-step baseline; earlier gain-over-initial results keep their original meaning. No new preconditioner, corpus fitting or circuit claim. [Protocol](SHARED_READER_RETAINED_HISTORY_V1_PREREGISTRATION.md), [29frozen dependencies](SHARED_READER_RETAINED_HISTORY_V1_BINDING.json).
