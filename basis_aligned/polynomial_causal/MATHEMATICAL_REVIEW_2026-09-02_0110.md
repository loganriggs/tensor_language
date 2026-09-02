# Three-hourly mathematical review (Claude lane) — 2026-09-02 01:10 UTC

Grounding: the 00:47 user direction correction (task-conditioned cross-module decomposition; rank/reconstruction/
CE-only progress rejected), TASK_CONDITIONED_CROSS_MODULE_DECOMPOSITION_PLAN.md, the 01:03 artifact audit routing
to the four-head equality-edge subset factorial (L5H5/L7H3/L8H3/L8H4), and §2572–§2574.

## Ranked mathematical moves for the NEW direction

### 1. (EXECUTED) Exact Moebius/Harsanyi dividends as the factorial's interaction calculus
Object: the 2^4 retain-subset outcomes of the equality-edge factorial (and any future term-subset factorial).
Theorem: Moebius inversion on the subset lattice gives the UNIQUE additive decomposition y(S)=Σ_{T⊆S} d(T)
(Harsanyi 1963 dividends; Rota 1964); Shapley values are dividend shares. Operational meanings with y = task
effect recovered: pairwise d<0 = redundancy (substitutable matchers), d>0 = complementarity, d≈0 = additive
independence — exactly the redundancy-vs-complementarity question the audit poses, answered with signs rather
than ad-hoc contrasts. Assumption that may fail: measurement noise; propagation is exact — σ(d_S)=2^{|S|/2}σ,
so with CUDA wobble σ≈.003 the floors are ~.006 (order 2), ~.0085 (order 3), ~.012 (order 4); PRE-HOC
RECOMMENDATION for the factorial's bars: any registered claim on an order-k dividend should clear these floors.
Shipped ops/mobius.py: dividends/shapley/reconstruct/noise-floor, verified on synthetic redundant (−1),
complementary (+1), additive (0), and exact 2^4 reconstruction. Zero GPU; consumes the factorial's receipt.

### 2. Interchange-commutation as the formal grouping criterion (causal abstraction)
Object: the plan's "task fingerprint" grouping of producer/composition/consumer terms. Operational definition
(Geiger et al., Causal Abstractions of Neural Networks, 2021; causal-abstraction framework 2023; Chan et al.
causal scrubbing 2022): a grouping is a valid abstraction iff interchange interventions on grouped variables
commute with the abstraction map — measurably, within-group interchange must be behavior-preserving up to ε
while between-group interchange is not; the single statistic is the commutation error of the abstraction square.
The plan's step 4 is this test informally; freezing it as a commutation error with matched-position controls
makes group validity a bar, not a narrative. Assumption that may fail: off-manifold interchanges (activations
outside the data support) — use natural-position interchange pools only. Consequence beyond reconstruction:
licensed swap/edit operations with predicted effects. Cheapest falsifier: one proposed group with within-group
interchange error above the between-group floor kills that group before any graph fitting.

### 3. Common-kernel quotient at TERM grain (the user's u~v iff R(u−v)≈0, finally at the right granularity)
Object: candidate term writes; consumer response family R = the later attention/MLP read maps the factorial
already captures. Math: "same downstream variable" = equivalence modulo the common kernel ∩_r ker(R_r); the
grouping lattice is the invariant-subspace lattice of the consumer family. The head-grain version failed all
year (§2535/§2536: no quotient collapse at head level); the direction correction predicts it succeeds at term
grain — a falsifiable reframe, and the response tensors the factorial captures are exactly the needed data.
Falsifier: term-grain response Gram rank ≥ term count (no collapse ⇒ no grouping at this grain either).

## Pruned (duplicates/closed): Hankel realization (§2560 closed), normalizer spectral certificates (closed,
22:10 memo), MDL/prequential (done, 2233 review), archetypal/convex dictionaries (0-for-6 across geometries),
INDSCAL/common congruence (§2555), any rank-sweep framing (excluded by the 00:47 direction).
