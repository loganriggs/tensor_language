# Three-hour mathematical review — 2026-08-29 10:00 UTC

## What changed the mathematical portfolio

The strongest new fact is not another low-rank fit. S1913 proves that the deployed
compiled program's logits are functions of the current token: within-token top-two
margin spread is at most `2.86e-05`, and the contextual margin differs from its
length-one value by at most `1.43e-05`. This turns program confidence into a literal
token table that can drive a priced cascade. S1914 then compared that cached token
margin and the native model's live margin against the **same** global permutation
null. The token signal had the larger low-to-high quartile enrichment gradient on all
three roles (`+9.50/+9.79/+9.28` versus `+6.03/+6.36/+6.02`). This is evidence that a
zero-forward-pass token table can rank where the compiled program is reliable; it is
not yet a calibrated error guarantee. S1914 also found roughly `+/-0.2` Monte Carlo
variation at only eight permutations, so tight reproductions of enrichment statistics
must either reuse frozen draws or increase the permutation count.

At the same time, the failures now sharply constrain generic state compression:

- closed rank-512 linear recursion costs `1.08978--1.27276` nat, and closed-input
  refitting costs about `5.5` nat;
- the E3 rank-64 destination oracle already has normalized error `0.2709`, with direct
  and chained errors `0.4861/0.4520`;
- Family F's refitted MLP3 program has NRMSE `0.70275--0.78860`, while native Down's
  better suffix KL remains a fit-only anomaly; and
- E2's shared bases help only at tight rank 64/128, not at the large deployable budget.

E4 is the exception: it now has audited fit/selection/final/code roles, a fixed
eight-candidate attention grammar, exact copy/control labels, an exact per-head physical
adapter, and simultaneous document-bootstrap statistics. Therefore new mathematics
should either deepen a passing E4 circuit, explain the native-Down causal reversal, or
certify whole-program composition. It should not delay the first E4 screen.

## Ranked top three genuinely new moves

### 1. Behavior-anchored finite Nerode/tensor-train realization of an E4 passer

**Exact bilin18 object.** After at least one registered E4 candidate passes selection,
expand only the six physical copy heads into the ordered action alphabet

$$
\{0,1\}_{5.5}\times\{0,1\}_{7.3}\times
\{\varnothing,8.3,8.4,8.3+8.4\}\times
\{0,1\}_{13.0}\times\{0,1\}_{14.7}.
$$

This is a 64-cell intervention cube. For each mask, retain a vector of native-relative
consequences: target log-probability/CE on positive, matched-negative, and off-target
cells plus the synthetic reciprocal-association difference-in-differences. Do not
collapse these to one scalar CE.

**The theorem/operational definition.** Arrange the response as a tensor
$T(a_1,\ldots,a_5,q)$. At physical cut $k$, matricize prefix actions against suffix
actions and consequence coordinate $q$. For an exact tensor train, the minimal bond
dimension at that cut is the rank of this unfolding. Equivalently, for this finite
ordered alphabet it is the state dimension of a layer-indexed linear weighted
automaton. Two prefix interventions may be merged only when every registered suffix
test agrees and their next-action transitions remain equivalent. This is a finite
behavioral/Nerode quotient, not a claim about bilin18's intrinsic hidden dimension.
Tensor-train ranks and stable SVD construction come from
[Oseledets 2011](https://doi.org/10.1137/090752286); the weighted-automaton/tensor-network
connection is made explicit by
[Li, Precup and Rabusseau](https://arxiv.org/abs/2010.10029).

**Assumptions that may fail.** The six-head alphabet may omit an essential reader;
same-layer heads 8.3/8.4 must be treated as one four-valued action; approximate ranks
may be ill-conditioned; document transfer may fail; and low average rank may hide one
catastrophic consequence cell. Finite equivalence licenses only this copy-test family.

**Prediction beyond reconstruction.** TT cores selected on untouched masks must
predict held-out head-subset effects, final-natural and code-OOD responses, and the
effect of removal/composition. A valid prefix-state merge gives an executable smaller
circuit and a direct notion of which interventions are behaviorally equivalent.

**Cheapest falsifier.** Conditional on an E4 passer, freeze a 32/16/16 train/validation/
final split of the 64 masks before new forwards. Fit TT ranks 1/2/4 on selection
documents. Require simultaneous worst-cell improvement over additive/Mobius and
matched low-rank controls, stable adjacent-cut ranks, and no refit on final/code. Stop
the entire minimal-realization branch if unseen-mask maximum error does not improve.

### 2. Native-Down finite-secant consequence Gram and certified column subset

**Exact bilin18 object.** Use the sealed K512 MLP3 products with their native Down
columns. For document $d$, product-coefficient direction $v$, amplitude $\epsilon$,
and a fixed vector response $R_d$ of target/donor log-odds and suffix-state probes,
measure

$$
J_d(v;\epsilon)=\frac{R_d(+\epsilon v)-R_d(-\epsilon v)}{2\epsilon},
$$

$$
Q_d(v;\epsilon)=\frac{R_d(+\epsilon v)+R_d(-\epsilon v)-2R_d(0)}{\epsilon^2}.
$$

Let $J$ denote the Jacobian of the stacked registered responses with respect to the
physical product coefficients. The random secants give the range sketch $Y=J\Omega$;
they do **not** identify the full product Gram by themselves. Orthonormalize
$Y=QR$, then use vector-Jacobian products to obtain

$$
B=Q^\top J.
$$

Now $B^\top B$ is the randomized low-rank approximation to $G=J^\top J$ on the
observed consequence range, and pivoted QR on $B$ can select physical product
columns. This changes the factorization target from coefficient energy to suffix
effect without requiring 512 separate finite-difference forwards.

**The theorem/operational definition.** Truncated SVD is the best rank-$r$ Frobenius
approximation to the measured linear consequence operator. A randomized range finder
approximates that subspace using $J\Omega$; the projected matrix $Q^\top J$ still has
one column per physical product, so a pivoted column subset can select executable
products. The residual norm of $J-QB$ bounds missed infinitesimal response over the
probed consequence coordinates. Because the object uses the physical write $d_jh_j$,
it is invariant to legitimate per-gate scale/sign/permutation gauges. It is not
invariant to arbitrary refactorization and is not a global CP-rank theorem.

**Assumptions that may fail.** Central secants must agree at $\epsilon$ and
$\epsilon/2$; the even response must be small enough at removal-sized amplitudes;
random directions must span consequential edits; and the consequence spectrum must
replicate by document. The suffix is nonlinear, so the linear spectral bound applies
only to the registered finite family unless amplitude transfer passes.

**Prediction beyond reconstruction.** A low-rank/subset consequence Gram predicts
unseen signed edits and selective-removal damage. It directly tests whether native
Down's `0.05772` suffix KL versus refitted Down's `0.08476` reflects real downstream
alignment or observational compensation.

**Cheapest falsifier.** Use 16 frozen Rademacher directions to construct $J\Omega$ by
JVP, recover $Q^\top J$ by one VJP per retained consequence direction, and check the
same directions by central secants at $\epsilon$ and $\epsilon/2$ on fresh selection
rows. Compare native Down, refitted Down, a same-support permuted cross, and matched
random columns. Reject if JVP/secant or amplitude transfer fails, effective rank is not
below matched price, or held-out direction/worst-coordinate error does not beat
controls. A run that has only random secants may test rank, but must not claim a
physical column subset.

### 3. Exact nonlinear hybrid telescoping certificate for whole-program drift

**Exact bilin18 object.** Define 37 physical hybrid executions

$$
H_k=\text{native suffix}_{k+1:36}\circ
\text{compiled prefix}_{1:k},\qquad k=0,\ldots,36,
$$

on the same row and token positions. Let $z_k$ be their capped logits.

**The theorem/operational definition.** For arbitrary nonlinear blocks,

$$
z_{36}-z_0=\sum_{k=1}^{36}(z_k-z_{k-1})
$$

exactly. Thus

$$
\|z_{36}-z_0\|_\infty\leq
B=\sum_k\|z_k-z_{k-1}\|_\infty.
$$

Cross-entropy is 2-Lipschitz in logit infinity norm, so its per-token change is at
most $2B$. Native top-1 is certified unchanged wherever the native top-two margin is
greater than $2B$. Unlike existing single-site restorations, adjacent hybrids evaluate
each local change on the state actually produced by the already-compiled prefix. The
ratio $B/\|z_{36}-z_0\|_\infty$ measures cancellation and tells us whether any additive
local certificate can be useful.

**Assumptions that may fail.** The equality cannot fail except through an execution or
support bug, but the triangle bound may be extremely loose. Empirical document
quantiles are not deterministic OOD bounds, and BF16 accumulation requires a frozen
tolerance for the telescope replay.

**Prediction beyond reconstruction.** A nonvacuous bound certifies CE/top-1 stability
and allocates drift at the correct recursively compiled state. A huge cancellation
ratio is also useful: it formally explains why local reconstruction and independent
rank allocation fail, and forces joint replacement grammars.

**Cheapest falsifier.** Run the 37 hybrids first on a small prospectively frozen role.
Require exact telescope replay, and stop if the p95 cancellation ratio exceeds 10 or
the implied mean CE bound already exceeds the measured E1 gap. This is nonredundant
with singleton restoration and broad factorials because neither forms a nested hybrid
chain.

## Useful fourth move executed now: a risk-controlled selective compiler

S1913 permits a near-free cascade: use the compiled token program when its cached
margin exceeds a threshold and defer otherwise. The risk object is not local MSE; it is
the conditional disagreement or task-error rate among accepted positions.

For independent documents, let $a_d(\tau)$ be accepted token mass and $e_d(\tau)$
accepted error mass. Hoeffding bounds with a $2K$ union correction simultaneously
bound $\mathbb E[e]$ and $\mathbb E[a]$ over every one of $K$ predeclared thresholds,
so the adaptively chosen threshold satisfies

$$
\frac{\mathbb E[e]}{\mathbb E[a]}
\leq
\frac{\min(1,\bar e+\epsilon)}{\max(0,\bar a-\epsilon)},
\qquad
\epsilon=\sqrt{\frac{\log(2K/\delta)}{2n}}.
$$

Positions may be arbitrarily dependent within a document. This is a conservative
cluster-level specialization of the broader risk-control framework in
[Angelopoulos et al.](https://arxiv.org/abs/2208.02814).

**Work executed:** `selective_compilation_risk.py` implements the simultaneous ratio
certificate and fail-closed threshold choice;
`SELECTIVE_COMPILATION_RISK_V1_PREREGISTRATION.md` freezes its interpretation.
`hybrid_telescoping_certificate.py` implements the exact telescope, CE bound, margin
certificate, and cancellation diagnostic.
`NATIVE_DOWN_CONSEQUENCE_GRAM_V1_PREREGISTRATION.md` freezes the corrected
JVP/range/VJP design and prevents a random directional sketch from being mislabeled as
a physical-product Gram. Fifteen new CPU tests pass; the combined
new plus E4 pure-statistics/contract suite passes 38/38. No model outcome was opened.

This selective compiler ranks below the three causal moves because deferring to native
does not explain rejected tokens or earn whole-model removal credit. It becomes a true
compression result only when the fallback is native-free and fully priced. It is still
the cheapest rigorous way to demonstrate that one simplicity measure enables a useful
action—safe selective use of the simpler program.

## Explicit pruning of the requested mathematical families

| Family | Decision after current evidence |
|---|---|
| Raw CP/HOSVD and arithmetic-circuit rank | Pruned for saved Family-F K256/K512: exact product-rank lower bounds meet the displayed product count. Approximate/deeper cross-layer reuse remains possible but has no causal coordinate yet. |
| Simultaneous/shared dictionaries | Large q512 global, typed, and hierarchical versions are negative. Permit one q64/q128 shared-private fit-only stability check, not another large sweep. |
| Gauge norm minimization before HOSVD | The legitimate reciprocal gate gauge only balances factor norms and leaves the folded tensor/HOSVD invariant. More general GL transformations are not symmetries of the elementwise product or identity residual stream. |
| Geometric invariant theory/operator scaling | Mathematically rigorous canonical forms do exist for tensor-network gauge orbits: [Acuaviva et al.](https://arxiv.org/abs/2209.14358) use minimal-norm orbit representatives, building on noncommutative scaling such as [Garg et al.](https://arxiv.org/abs/1511.03730). But bilin18's allowed physical gauges have not been shown to contain the required group action. Revisit only for two proven functionally equivalent tensor programs; do not use it as a blind compressor. |
| Generic system identification/Hankel | E3's pointwise rank-64 transport is negative and the old token Hankel is OOD. Keep only the behavior-anchored finite E4 TT/Nerode object above. |
| Generic lexical weighted automaton | Missing Hankel entries require matrix-completion assumptions; [Balle and Mohri](https://papers.nips.cc/paper/2012/hash/700fdb2ba62d4554dc268c65add4b16e-Abstract.html) provide a relevant framework, but current token IDs/byte strings have no demonstrated low-rank closure. Defer until the E4 behavior tensor or a serialized token-output table provides an actual consequence target. |
| MDL/prequential coding | Use only to compare two causally admitted autonomous programs, charging basis, router, precision, and search. It cannot make a causally wrong abstraction correct. The selective cascade may report literal price now. |
| Information bottleneck | Deterministic continuous mutual information is ill-posed/trivial without a noise/quantization convention. The finite action/test quotient gives a measurable sufficiency target with fewer estimator assumptions. |
| Causal abstraction/bisimulation | Keep only when the action alphabet and suffix-test family are explicit and unseen compositions are scored, as in the E4 finite quotient. Activation clustering alone is not bisimulation. |
| Sparse program synthesis | E4's fixed head grammar is the current bounded synthesis problem. Expand to a typed DSL only after one circuit passes; otherwise search cost and description length are post-hoc. |
| Approximation certificates | Global products of Lipschitz constants will be vacuous across 36 RMSNorm/residual interfaces. Run the exact hybrid telescope and finite secants first; formalize a tighter certificate only if their empirical bounds are nonvacuous. |

## Priority and execution consequence

1. Do not interrupt the E4 screen. If it yields a passer, its 64-mask vector-valued
   TT/Nerode assay is the best route to a minimal editable copy circuit.
2. Run the native-Down two-amplitude finite-secant Gram rather than another local
   decoder fit.
3. Pilot the 37-arm hybrid telescope; stop immediately if cancellation makes it
   vacuous.
4. S1914 has resolved the common-null comparison in favor of the cached token signal.
   Calibrate the selective-risk gate using document-level labeled errors and many more
   than eight frozen permutations where enrichment is reported; preserve final/code
   as untouched replication roles.

The mathematics this review added is operational: each retained notion of simplicity
predicts a capability beyond reconstruction—unseen composition, selective removal,
whole-program error certification, or risk-controlled deferral.
