# Mathematical review: downstream-minimal tensor programs

Date: 2026-08-28 06:30 UTC
Status: CPU mathematics and prospective experiment design only. This document grants
no row, model, validation, final, or promotion authority.

## Project state which changes the mathematics

The model is not “36/36 explained.” All 36 sites have structural inventory entries,
but named behavior is only $32.1\%\pm6.4\%$, named causal recovery is $10.923\%$,
and no executable program has earned credit against the paired $+0.8976$ current-ship
CE gap. A different constant-replacement experiment has a $55.038\%$ discovery and
$53.694\%$ held-out ceiling; those denominators cannot be combined.

The most relevant local facts are:

- MLP0/1 already have an exact 64-dimensional projected interface and a source-closed
  suffix/transport experiment. The unresolved question is what part of the 64-D code
  the *downstream model can use*, not whether another Euclidean fit can improve.
- A generic prefix/continuation Hankel experiment failed in the regime tested. Its
  synthetic splices were $+3.54$ to $+3.61$ CE off the natural distribution, rank-95
  was 23--24 of 48, and low-rank completion improved only 4.5--10.1% against a 30%
  bar. Unnatural token splicing is therefore not the next Hankel experiment.
- The existing code covariance, physical code-edit operator, gauge transformations,
  suffix KL transaction, and 32 frozen covariance-shaped directions supply nearly all
  ingredients for a *same-forward observability* experiment.
- One-at-a-time component importance is non-compositional: it ranks near random for
  budget allocation, while program-context greedy selection is much better and the
  objective is demonstrably non-submodular. The completed allocation-basin control
  now has all three random starts converging to the exact same six-site set and
  $1.2037$-nat value as greedy. This makes the joint target reproducible, but does not
  make local importance compositional. A useful mathematical state must predict joint
  downstream consequences, not merely reconstruct a local tensor.
- Fixed-mask fit-size evidence already separates statistical complexity from raw
  parameter count: moving 96 to 480 fit rows changed recovery by +5.294 points for the
  additive family, +2.643 for linear, and only +0.215 for the table family.
- A result receipt completed during this audit at the stable six-site set. A
  table-plus-input-linear correction recovered only $8.29\%$ of the held-out native-six
  gap at rank 8, $7.01\%$ at rank 32, and $5.69\%$ at rank 128; the five attention sites
  averaged negative rank-128 recovery. The correction was fitted under the live model
  and deployed with other sites tabled, so these are lower bounds under a known context
  mismatch. Even so, failure of rank monotonicity shows that Euclidean residual SVD is
  not itself the downstream simplicity ordering we need.
- The declared context-mismatch control then separated MLP and attention behavior.
  Fitting in the actual all-tabled deployment context raised held-out six-site recovery
  to $38.14\%$ and MLP17 to $92.06\%$ of its own gap at rank 128, while the five
  attention sites remained negative on average. The all-site map strengthened this:
  median rank-128 own-gap recovery was $91.23\%$ for MLPs and $-1.45\%$ for attention.
  Yet installing all 36 corrections jointly *lost* $0.5462$ nat versus tables at rank
  8. Thus simple local MLP maps exist, but simultaneous fits are not a composable
  compiler; attention still lacks a nonlocal routing state. The GPU lane now owns the
  prospectively interleaved bottom-up composition test, so this CPU lane does not
  duplicate it.

## Ranked genuinely new moves

### 1. Finite-horizon balanced predictive quotient

**Exact bilin18 object.** Let $z\in\mathbb R^{64}$ be the selected MLP0 code on its
natural fit trajectory, with covariance $C$. Let the native downstream suffix map,
under a same-forward physical edit $\delta B_0^\top$, produce logits $\ell_x(z)$ in
context $x$. Define the local downstream Fisher/response metric

$$
O = \mathbb E_x[J_x^\top F_xJ_x],
\qquad J_x=\frac{\partial \ell_x}{\partial z},
\qquad F_x=\operatorname{diag}(p_x)-p_xp_x^\top.
$$

This is estimable without materializing a $50{,}304\times64$ Jacobian: draw
$y\sim p_x$, backpropagate $e_y-p_x$, and average the outer products of the resulting
64-D VJPs. Multiple deterministic registered probes reduce variance.

**Operational theorem.** Draw a covariance-shaped code edit $\delta$ independently of
the evaluation context, with $\mathbb E[\delta\delta^\top]=C$, and use the mean local
response metric $O$. On the support of $C$, whiten $\delta=C^{1/2}u$ and set
$H=C^{1/2}OC^{1/2}$. If $\lambda_1\geq\cdots\geq\lambda_r\geq0$ are the eigenvalues
of $H$, then the best rank-$d$ linear reconstruction of this edit ensemble for expected
downstream quadratic distortion retains the top $d$ eigenvectors, and its minimum
discarded distortion is

$$
\min_{\operatorname{rank}(A)\leq d}
\mathbb E[(z-\hat z)^\top O(z-\hat z)]
=\sum_{i>d}\lambda_i.
$$

Proof: with $M=O^{1/2}C^{1/2}$, the objective is
$\lVert M(I-P)\rVert_F^2$ for a rank-$d$ whitened projector $P$; Eckart--Young/Ky
Fan chooses the top right singular vectors of $M$, whose squared singular values are
the eigenvalues of $H$. These $\sqrt{\lambda_i}$ are the finite-interface analogue of
Hankel singular values. Classical balanced realization similarly combines
controllability and observability rather than state variance alone
([Moore 1981](https://algos.inesc-id.pt/projects/mor4less/Moore_81.pdf)). For nonlinear
systems, trajectory-wise differential Gramians can be computed from variational or
empirical responses
([Kawano and Scherpen 2019](https://arxiv.org/abs/1902.09836)).

This also gives the requested principled clustering definition. Two codes are
downstream-equivalent if every registered continuation/test has the same response.
Predictive-state representations define state by such future tests, rather than by a
latent generative label
([Littman, Sutton, and Singh 2001](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)).
The exact nonlinear equivalence is a causal/bisimulation quotient; $\ker O$ is only
its local linearization.

This theorem does **not** say that independently averaged $C$ and $O$ solve paired
natural-state reconstruction when a context-specific $O_x$ is correlated with $z_x$.
That objective contains the joint moment $\mathbb E[O_x z_xz_x^\top]$. The current
quotient deliberately prices an ideal independently assigned edit ensemble with raw
covariance $C$; the finite clipped-and-RMS-normalized 32-direction bank is an external
consequence test, not an exact draw from that prior. Per-row nonlinear prediction
separately tests whether the mean quotient hides important context dependence.

**Gauge statement.** Under the current orthogonal code gauge $z'=Q^\top z$,
$C'=Q^\top CQ$ and $O'=Q^\top OQ$. Therefore $H'$ is orthogonally similar to $H$,
the spectrum and tail certificate are invariant, and the natural projector transforms
as $A'=Q^\top AQ$. This is a physically meaningful dimension, unlike coordinate
sparsity or PCA axes.

**Assumptions that may fail.** The suffix is nonlinear; Fisher KL is only second-order;
the empirical context/probe set may miss rare downstream tests; $C$ can be ill
conditioned; a low-rank local Gramian need not transport to a second corpus; and no
stable-LTI $H_\infty$ error bound applies. Exact bisimulation would require all relevant
contexts and finite interventions. Reachability-based approximate bisimulation can
provide assured output-error bounds for feed-forward networks, but scales poorly and
usually over-approximates
([Xiang et al. 2022](https://arxiv.org/abs/2202.01214)).

**Prediction beyond reconstruction.** A retained rank $d$ must predict, before final
observation, (i) the average KL response to held-out covariance-shaped edits; (ii) the
null response of discarded directions; (iii) the response of mixtures of directions;
(iv) stability under exact gauge replay; and (v) which code edits can be selectively
removed without collateral downstream change. The tail sum predicts quadratic edit
distortion, not local code MSE.

**Cheapest falsifier.** On the existing validation transaction, estimate $O$ using
registered Fisher VJP sketches, freeze the spectrum/rank without final rows, and test
the already frozen 32 directions at the selected amplitude. Reject immediately if
there is no stable spectral gap, gauge replay changes the spectrum/projector, validation
quadratic prediction has low response $R^2$, or discarded directions exceed a
preregistered KL remainder. Only a passing object reaches final/OOD.

**CPU action completed.** `predictive_quotient.py` implements covariance and VJP
Gramian estimators, the balanced spectrum, optimal projector, exact tail quantity,
retained-fraction rank, and quadratic response. Five tests prove a known diagonal
problem, orthogonal-gauge covariance, VJP/covariance identities, null-observability,
singular covariance-support separation, and fail-closed PSD/symmetry checks.

### 2. Quotient-canonical description length for tensor programs

**Exact bilin18 object.** The factorizations $W=LR$, bases $B_0,B_1$, transport map
$A$, later shared attention/MLP dictionaries, and any polynomial tensor cores have
internal gauge freedom. Raw factor counts and sparse coordinates overcount the same
physical program and can change under harmless refactorization.

**Theorem/definition.** Price the canonical representative of the gauge orbit, plus
support indices and coefficient precision, rather than a chosen factorization. Modern
geometric invariant theory gives a minimal canonical tensor-network form: equality of
minimal forms characterizes gauge equivalence up to orbit closure, and equality of all
contractions characterizes the same tensor-network state
([Acuaviva et al. 2022](https://arxiv.org/abs/2209.14358)). For the current simple
orthonormal interfaces, physical contractions such as $B_0AB_1^\top$ and canonical
SVD blocks are cheaper exact invariants than a general GIT solver.

**Assumptions that may fail.** The theorem is for contracted tensor networks; bilin18
also has residual addition, RMSNorm, and nonlinear output maps. Degenerate singular
spaces make a byte-level canonical basis discontinuous. General $\mathrm{GL}$ gauges
can be ill-conditioned. Thus apply the theorem only to a declared polynomial tensor
subgraph, and price physical contractions when canonical axes are not identifiable.

**Prediction beyond reconstruction.** Gauge-equivalent implementations must receive
the same complexity, selection, extraction graph, and edit predictions. Quotient
dimension should predict fit-size requirements better than raw factor parameters.

**Cheapest falsifier.** Generate orthogonal and bounded-condition-number $\mathrm{GL}$
refactorizations of the same selected program. Canonicalize and require identical
physical tensors, complexity bits, chosen frontier point, and intervention responses.
If cost or circuit identity changes, the proposed simplicity score is invalid.

### 3. Block-prequential MDL over executable program families

**Exact bilin18 object.** Compare L/R/T, ranks, shared versus independent dictionaries,
and later attention grammars with one frozen training algorithm and ordered fit blocks.
The code transmits each next block using a program fit only to earlier blocks:

$$
L_{\rm preq}=t_1\log_2 K+
\sum_s -\log_2 p_{\hat\theta(D_{1:t_s})}
(D_{t_s+1:t_{s+1}}).
$$

This is an operational complexity, not “number of float32 parameters.” Prequential
coding avoids explicitly transmitting weights and measures how quickly a family learns;
its exact blockwise definition and catch-up failure mode are described by
[Blier and Ollivier 2018](https://proceedings.neurips.cc/paper/2018/file/3b712de48137572f3849aabd5666a4e3-Paper.pdf).

**Assumptions that may fail.** The training algorithm, seeds, block order, stopping
rule, and hyperparameter-code cost must be frozen. Early blocks create catch-up bias;
validation selection cannot be reused as code data; and low in-domain codelength need
not imply editability or OOD transport.

**Prediction beyond reconstruction.** At matched validation suffix distortion, lower
prequential bits should predict better data-doubling stability and second-corpus
performance. It need not predict edit locality; that is scored separately.

**Cheapest falsifier.** Reuse the frozen fit schedule and candidate bank with
prospectively ordered 24/48/96/192/384-row blocks, uniform-code the first block, and
score only the next block at each step. Reject prequential MDL as a useful simplicity
measure if its ordering fails to predict untouched-document and second-corpus ordering
better than factor-complete parameter count. Existing 96/480 fixed-mask results are a
motivation, not a valid prequential code because intermediate next-block losses were
not retained.

## Other mathematical routes reconsidered and pruned

| route | exact object and valid mathematics | measurable consequence | cheapest falsifier / decision |
|---|---|---|---|
| Simultaneous factorization and shared dictionaries | Stack *physical, observability-whitened* site maps and use the SVD/Eckart--Young shared subspace; never align arbitrary code axes. | A shared dictionary should match independent dictionaries at lower total bits and improve cross-site interchange. | After one interface is admitted, compare shared versus independent bases at equal total rank on held-out response. Defer now: raw shared-axis work duplicates failed local fitting. |
| Generic Hankel / weighted automata | For a rational series, finite Hankel rank equals minimal weighted-automaton state count; spectral factorization recovers a realization ([Arrivault et al. 2016](https://proceedings.mlr.press/v57/arrivault16.pdf)). | Predict unseen prefix/continuation values and yield an executable automaton. | Already falsified for the tested unnatural splice regime. Do not enlarge it. The same-forward observability quotient above is a different object. |
| Tensor and arithmetic-circuit rank | Each bilinear MLP and squared-attention contraction is a low-degree polynomial piece; TT ranks equal unfolding ranks for a fixed tensorization, while arithmetic-circuit size prices shared intermediate products. | Exact smaller contraction count and predictable polynomial interventions. | Compute degree-2/4 jet unfolding ranks and norm-controlled Taylor remainder at one admitted interface. Defer: RMSNorm introduces inverse square roots, degree compounds rapidly, and prior product families lost to linear maps. |
| Polynomial invariant theory | Contracted cycles/physical maps are gauge invariants; canonical orbit representatives can remove fictitious factor complexity. | Gauge-stable circuit identity and quotient MDL. | Included in priority 2. Full-network invariant rings are pruned until the polynomial subgraph is explicitly delimited. |
| Causal abstraction and bisimulation | Quotient two internal states only if every allowed downstream intervention produces the same output distribution; approximate versions attach an output-error bound. | Selective replacement/removal and collateral guarantees. | Included as the nonlinear target of priority 1. Finite empirical contexts cannot be called exact bisimulation. |
| Information bottleneck | A code sufficient for downstream outputs while minimizing information about input is a plausible objective. | OOD prediction with fewer nuisance bits. | Pruned now: deterministic continuous mutual information is ill-defined/infinite without an arbitrary noise model, and low MI can delete rare causal variables. Predictive equivalence is more directly falsifiable. |
| Sparse program synthesis | Search over typed tensor primitives with MDL penalties can expose a small executable graph. | Extraction, edit locality, and lower execution cost. | Defer until producer/consumer primitives and a clean consequence role exist. Searching the present discovery ledger would overfit hypotheses and non-submodular interactions. |
| Global approximation certificates | Interval/reachability or Lipschitz/Hessian bounds could certify replacement error through the suffix. | Guaranteed CE/logit or safety envelope. | Use a local Taylor-remainder certificate inside priority 1 first. A naive 18-layer global Lipschitz product is expected to be vacuous, especially near RMSNorm small norms. |
| Pure low-rank/PCA or local MSE | Eckart--Young is exact for the chosen Euclidean reconstruction tensor. | Storage/reconstruction only. | Pruned as a definition of understanding: the project already has rank-64 local structure and failed composition. It survives only as the controllability half $C$, not the state definition. |

## Falsifiable preregistration for move 1

Before observing its validation spectrum or edit responses, freeze:

1. interface: selected L0 physical code, all 384 fit rows for $C$;
2. output metric: suffix softmax Fisher on positions 64:256 under the exact O/O/N
   downstream teacher, with deterministic probe seeds and no label access;
3. estimator: mean VJP outer product, exact context/probe counts, float64 CPU reduction;
4. rank rule: smallest $d$ retaining at least 95% of $\sum_i\lambda_i$, provided the
   cut gap $\lambda_d/\lambda_{d+1}\ge2$; otherwise record “no certified knee”;
5. local consequence: on held-out registered edit directions, quadratic KL prediction
   must have pooled $R^2\ge0.5$ and median relative error $\le25\%$ at the selected
   calibration amplitude;
6. discarded-direction consequence: observed median KL in the discarded subspace is
   at most 10% of the retained-subspace median under matched natural RMS;
7. gauge consequence: all eight exact gauge replays preserve the spectrum to
   $10^{-8}$ relative error and the physical rank-$d$ projector response to $10^{-6}$;
8. transport: repeat the frozen rank/projector without reselection on the final role
   and later a genuinely shifted corpus/code role. Failure receives no causal-interface
   or simplicity credit.

Thresholds 4--7 are prospective design choices and may be tightened by an independent
audit before any role opens, but never after spectrum or response observation.
