# Mathematical review: from local compression to a minimal causal state

Date: 2026-08-28 12:45 UTC

Status: literature-backed CPU mathematics, proof kernel, and prospective design. This
review opens no new row or model role and grants no GPU authority.

## Project state that changes the mathematical question

The complete-program frontier is no longer empty. Shared-QK rank640 owns the entire
standalone program with 516,707,766 stored values, 5.3481% below dense. It has only
+0.00553/+0.00445 nat cross-task all-position CE harm and passes a prospective
16-intervention bank with mean recovery 0.94442 (one-sided 95% lower bound 0.92726),
mean cosine 0.97238 (lower bound 0.96367), and 14/16 individual joint passes.

This establishes a working contextual shell and changes the next mathematical object.
Dense MLPs still hold 286,675,200 values, 52.51% of the model, and each MLP physically
evaluates 4,608 scalar products

$$
h_n(x)=(\ell_n^\top x)(r_n^\top x),\qquad
M(x)=b+\sum_{n=1}^{4608} d_n h_n(x).
$$

The strongest earlier executable early-MLP compiler retained only 26.10% of the exact
projected pair's final CE gain. Final-CE training beat local residual fitting by 43%,
and locally good MLP fits failed when installed jointly. The missing object is therefore
not another Euclidean rank: it is the state which carries *joint upstream intervention
effects through RMSNorm, residual addition, attention, and later MLPs*.

The GPU is idle. Hash-authorized FineWeb caches and the admitted rank640 receipt are
present. There is no `rspd`, data, checkpoint, or compute blocker. Existing untracked
bilinear-quotient artifacts and another agent's MLP-composition drafts were left
untouched.

## Work pruned as repetition before ranking

- The single-interface balanced predictive quotient, gauge-canonical MDL, block-
  prequential coding, deterministic information-bottleneck objection, and generic
  prefix/continuation Hankel experiment were already reviewed at 06:30. They are not
  renamed as new moves here.
- Generic token splices remain rejected: their prior CE was 3.54--3.61 OOD and their
  Hankel completion missed its gate.
- Raw local MSE/PCA and all-position-free per-token tables cannot define a composable
  state. The exact position-wise no-context theorem still applies to that family.
- A larger rank sweep is not mathematics. Rank640 already resolves the immediate
  attention-capacity issue; MLPs dominate the remaining executable price.

## Ranked move 1: cutwise tangent Hankel/Schmidt realization

### Exact bilin18 object

Linearize the admitted rank640 program on natural trajectories. At MLP write interface
$i$, inject a registered residual perturbation $u_i$. Let $y_j$ be registered downstream
softmax-Fisher score tests. The same-forward response blocks are

$$
H_{ji}=\frac{\partial y_j}{\partial u_i}.
$$

For a depth cut $k$, stack all tests after the cut against all interventions before it:

$$
\mathcal H_k=
\left[H_{ji}\right]_{j\ge k,\ i<k}.
$$

Attention's cross-position reads, the first-value bus, every RMSNorm derivative, and
residual mixing are inside $H_{ji}$. They are not approximated as independent modules.

### Theorem and operational definition

If a linearized causal program passes every upstream effect through an $r$-dimensional
state $z_k$, then $\mathcal H_k=D_kE_k$ and

$$
r\ge\operatorname{rank}(\mathcal H_k).
$$

Thus the cut rank is a certified lower bound on the required tangent-state/bond
dimension. Eckart--Young gives the best rank-$r$ approximation at that cut, with exact
squared response tail $\sum_{q>r}\sigma_q^2$. This is simultaneously:

- the operator-Schmidt rank across the tensor-network depth cut;
- the finite-horizon reachability/observability transmission rank;
- the block-Hankel rank used by minimal-realization methods.

Balanced truncation for linear time-varying systems uses time-indexed reachability and
observability Gramians and supplies error bounds under its stated linear assumptions
([Sandberg and Rantzer 2004](https://lup.lub.lu.se/search/files/4812893/625601.pdf)).
Empirical balancing extends the input-output construction to simulated nonlinear
systems, but as an approximation rather than an exact nonlinear theorem
([Lall, Marsden, and Glavaški 2002](https://www.cds.caltech.edu/~marsden/bib/2002/06-LaMaGl2002/LaMaGl2002.pdf)).

This also explains the tensor-network analogy precisely. DMRG keeps the subsystem
subspace selected by its environment/reduced density matrix
([White 1992](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.69.2863));
here the environment is the downstream behavioral test operator, not Euclidean local
activation energy or a quantum-state norm.

### Assumptions that may fail

- The theorem is exact only for the tangent program and the declared finite tests.
  Rank640's finite synthetic interventions need not remain in the linear tube.
- Averaging contexts before factorization can hide context-dependent state directions;
  source-document split stability and per-context mixture tests are mandatory.
- A low-rank factor at each cut separately does not automatically give one compatible
  nested causal realization across every cut. Transport/intertwining constraints must
  be checked before compilation.
- Fisher probes measure predictive distribution response, not every semantic or safety
  observation. OOD tests can add rank.

### Consequence predicted beyond reconstruction

A true cut state must predict arbitrary heldout mixtures of upstream interventions,
which discarded directions are safely removable, how joint MLP0/1/2 replacements
compose, and whether the same state transports to a second corpus. It also gives a
lower bound on any claimed smaller tensor program: a candidate with bond dimension
below a stable measured cut rank cannot reproduce the registered causal tests.

### Cheapest falsifier

Use MLP0--2 only, 32 fixed covariance-shaped write directions per site, and 16
categorical-Fisher probes on skip80. Split by source document. Reject immediately if
the 95%-energy cut rank has no factor-two gap, differs by more than two across halves,
has normalized spectral distance above 0.10 or projector distance above 0.15, or fails
heldout two-site mixture prediction. Only a stable tangent state reaches a complete-
program finite-edit test.

### CPU action executed

`finite_horizon_tangent_realization.py` now implements fail-closed block assembly,
exact cut rank and spectrum, optimal truncated factors and tail errors, and typed
orthogonal-gauge replay. A create-only time-varying proof fixture contains a third state
which is unreachable and unobservable. All three cut ranks equal two; rank-two factors
reconstruct every cut to $3.4\times10^{-16}$ maximum error, and independent typed gauge
replays preserve singular values to $4.5\times10^{-17}$. Six tests pass. The production
pilot is frozen separately in
`FINITE_HORIZON_TANGENT_REALIZATION_PREREGISTRATION.md`.

## Ranked move 2: downstream-environment column selection of multiplication gates

### Exact bilin18 object

Each MLP exposes 4,608 *physical* bilinear gates $h_n(x)$. Construct a response-design
matrix $E_l$ whose $n$th column is gate $n$'s trajectory across registered contexts and
downstream tangent tests from move 1. Select actual columns, then refit only the Down
decoder. The executable program still evaluates recognizable original products; it is
not a dense rotated latent basis.

### Theorem and operational definition

Statistical leverage-score column selection returns actual columns whose span gives a
relative-error approximation to the best rank-$q$ matrix approximation
([Drineas, Mahoney, and Muthukrishnan 2006](https://doi.org/10.1007/11830924_30)).
For a whitened response Gramian written as a sum of rank-one terms, deterministic
spectral sparsification can preserve the whole quadratic form with only linearly many
weighted terms in the response dimension
([Batson, Spielman, and Srivastava 2012](https://arxiv.org/abs/0808.0163)).

The complexity is the number of retained physical multiplications plus the refitted
decoder price. This is an arithmetic-circuit measure with a useful approximation
theorem, unlike raw CP factor count or coordinate sparsity.

### Assumptions that may fail

- Matrix column-selection guarantees apply to the measured linear response matrix,
  not the nonlinear MLP function on all residual states.
- A gate with low average leverage can be essential on a rare context or after another
  gate is removed. The current program objective is non-submodular.
- Refitting Down may destroy simple semantic meaning even though the product gates are
  physical. OOD leverage can differ sharply from fit leverage.

### Consequence predicted beyond reconstruction

At matched tangent-response error, selected gates should use fewer bilinear multiplies,
retain rank640 CE/context transport, and make removal/extraction more localized than a
dense rotated factorization. Leverage should predict which gates survive data doubling
and a second corpus.

### Cheapest falsifier

At MLP0 only, compare $k\in\{64,128,256,512\}$ environment-leverage gates with
top-norm and random gates at identical price. Reject if leverage does not win heldout
response, finite-edit KL, and complete-program CE simultaneously, or if its selected
set has Jaccard stability below 0.5 under data doubling. Do not launch an all-layer
gate sweep first.

## Ranked move 3: approximate bisimulation congruence for semantic components

### Exact bilin18 object

The candidates are physical product gates, shared-routing directions, or small gate
packages. Two candidates belong to one semantic class only if no allowed downstream
test distinguishes their interventions *and* the relation is preserved when the next
rank640 transition is applied. This turns “clustered tokens/components” into an
operational equivalence rather than geometry in an arbitrary latent basis.

Define a finite behavioral pseudometric from natural, synthetic, removal, interchange,
and mixture responses. A cluster is admitted only when within-cluster interchange has
small output distance and future transitions map its members into the same quotient
classes. Approximate bisimulation formalizes bounded rather than exact observational
equivalence; for constrained linear systems, bisimulation functions give explicit
precision bounds
([Girard and Pappas 2007](https://doi.org/10.1016/j.automatica.2007.01.019)).
Interchange interventions are likewise the operational test for causal abstractions
of neural computations
([Geiger et al. 2021](https://papers.nips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html)).

### Assumptions that may fail

- A finite test algebra can merge states separated by an unmeasured continuation.
- Approximate equivalence need not be transitive without an explicit metric/threshold
  construction. Clustering response vectors alone is not a congruence.
- bilin18 may use a continuous superposition with no stable discrete quotient. The
  current circuit audit already fails its profile-discrimination prediction, so a
  positive cluster count cannot be assumed.

### Consequence predicted beyond reconstruction

An admitted quotient must predict within-class interchange, selective class removal,
collateral damage to other classes, and transfer of the same edit to OOD contexts.
This directly validates whether “simpler” buys extraction and manipulation.

### Cheapest falsifier

Learn the quotient threshold and clusters on half of the existing intervention bank;
freeze them; then require within-cluster interchange to be at least five times closer
than matched between-cluster interchange on the other half, with removal collateral
inside the cluster-specific bound. Failure means the interface is continuous at the
tested resolution and must not be narrated as discrete classes.

## Remaining routes: use, defer, or prune

| mathematical route | decision after current evidence |
|---|---|
| Tensor/arithmetic-circuit rank | Multiplicative tensor rank is the exact number of bilinear products, but practical certified lower bounds for a $1152^3$ MLP tensor are too weak/expensive. Use physical-gate count and move-2 response approximation first; do not claim exact CP minimality. |
| Simultaneous factorization/shared dictionaries | Potentially valuable only after move 1 supplies physical cross-layer state gauges. Stacking raw per-layer axes would align unrelated coordinates and repeat failed local composition. |
| Polynomial invariant theory/gauge quotients | Retained as a validity condition: cut spectra, physical projectors, prices, and circuit identities must be gauge invariant. A full invariant ring across RMSNorm and residual addition is not the next compiler. |
| Algebraic complexity | Move 2 gives an executable sparse arithmetic circuit. Border-rank-only decompositions are pruned because numerical limiting algorithms can be ill-conditioned and hard to edit. |
| System identification/minimal realization | Promoted as move 1 on same-forward internal interventions. Generic token-prefix Hankel completion remains pruned. |
| Hankel/weighted automata | The formal-series theorem is valid, but the tested token-splice object was OOD. The new cut operator is a derivative of the actual contextual program, not a relabeling of that failed experiment. |
| MDL/prequential coding | Still useful to choose among candidates *after* moves 1/2 create multiple executable families. It cannot create the missing causal state and is not repeated now. |
| Causal abstraction/bisimulation | Promoted as move 3; exact claims require closure over interventions, while finite tests license only approximate empirical congruence. |
| Information bottleneck | Pruned: deterministic continuous mutual information requires an arbitrary noise/quantization model and can discard rare causal variables. |
| Sparse program synthesis | Restricted to move 2's theorem-guided physical gate selection. Untyped search over discovery artifacts is pruned as overfit and non-compositional. |
| Global approximation certificates | The SVD tail is exact for the tangent response operator. A global 18-layer Lipschitz product remains expected to be vacuous near RMSNorm; finite nonlinear consequences remain mandatory. |
| Raw tensor-network/DMRG sweep | The environment principle is used, but bilin18 is not a normalized MPS and CE is not a Hilbert-state norm. Importing DMRG truncation error literally would be mathematically invalid. |

## Decision

The top three are, in order: (1) cutwise tangent minimal realization; (2) environment-
weighted physical gate selection; (3) approximate bisimulation congruence. Move 1 has
the highest information gain because one pilot distinguishes a genuinely shared
cross-layer state from incompatible local quotients and supplies a lower bound on every
future compiler. Its safe CPU kernel, proof check, tests, and prospective pilot are now
complete. The next GPU action is not authorized by this document; it requires a
source-closed matrix-free response collector and an independent lifecycle audit.
