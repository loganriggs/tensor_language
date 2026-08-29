# Three-hour mathematical review — 2026-08-29 21:50 UTC

## Bottom line

This review produced one genuinely new result rather than another proposal:

> The native quadratic maps of MLP0, MLP1, and MLP2 cannot be globally equal to
> any 512-product bilinear program.  A deterministic slice of each native tensor
> has numerical rank 1,152.  Consequently an exactly equal product program needs
> at least 1,152 multiplicative channels, even if constant and linear corrections
> are free.

The best possible rank-512 approximation to that one matrix slice still has relative
Frobenius error at least **28.53% / 30.17% / 29.49%** for MLP0/1/2.  This does **not**
contradict the good natural-text CE of MLP0-C512 or MLP2-FULL512.  Instead it proves
that their success is distributional: natural text and the suffix care about a much
smaller part of the global polynomial than arbitrary residual directions do.

That distinction changes the priority.  More unweighted tensor factorization cannot
make rank 512 globally faithful.  We should identify the *reachable and downstream-
observable* part of the tensor and certify approximation there.

## Evidence that sets the problem

- The strict whole-model ledgers have not moved: **36/36** structural sites have an
  intervention, **5.348245316%** of storage is certified removable,
  **10.923302467%** of causal CE is named, **4.72714 nat / 89.077%** remains unnamed,
  and **0/68** terminal actions is complete.
- MLP2-FULL512 recovers **68.27%** of the CE cost of deleting MLP2 at a price of 512
  products, but its local-write NRMSE is **0.6866**.  Downstream usefulness and local
  Euclidean reconstruction are therefore different objectives.
- MLP0-C512 and MLP2-FULL512 compose imperfectly: their physical joint CE cost is
  **0.064996 nat**, with interaction **+0.008739 nat** and bootstrap 95% interval
  **[0.007511, 0.010014]**.  The interface error is small but real.
- The matched native/C512 MLP2 fit found that C512 moves the MLP2 input by **0.15078
  NRMSE**, yet equal exposure improves dev normalized MSE by only **0.061%** on native
  and **0.282%** on C512 trajectories.  A simple two-background average is unlikely
  to be the missing language.
- The quotient lane has now found a sharp but semantically unnamed interaction around
  attention 5 and 6.  Both sites are extremely compressible *inside that intervention
  family*.  At attention 5, rank 16 versus rank 384 changes CE by only
  **0.0094--0.0099 nat** across roles; rank 16 arms cost **1.961--2.100 nat** while
  omitting attention 5 costs **4.876--5.355 nat**.  The run took **205 s** and all
  registered controls passed.  This is evidence for a narrow causal interface, not
  yet a semantic circuit.
- No Codex GPU job was active at review time.  The CPU certificate below took
  **15.71 s**; its two tests took **1.68 s**.

## New result: a global polynomial lower bound

### The computation

For one bilinear MLP, ignore its residual addition and write its quadratic part as

$$
q(x)=D\big((Lx)\odot(Rx)\big),
$$

where $x\in\mathbb R^{1152}$, $L,R\in\mathbb R^{r\times1152}$,
$D\in\mathbb R^{1152\times r}$, and $r$ is the number of multiplicative channels.
The native MLP has $r=4608$.

The symmetric polarization of $q$ is the bilinear map

$$
B(u,v)=\frac{q(u+v)-q(u)-q(v)}{2}.
$$

Fixing its second input to a vector $y$ gives an ordinary $1152\times1152$ matrix

$$
A_y=\frac12D\left[\operatorname{diag}(Ry)L+
                         \operatorname{diag}(Ly)R\right].
$$

Each multiplicative channel contributes one outer product: its output vector is the
column $d_i$ of $D$, and its input row is

$$
\frac12\left((r_i^\top y)l_i^\top+(l_i^\top y)r_i^\top\right).
$$

Therefore every $r$-product quadratic program obeys

$$
\operatorname{rank}(A_y)\le r.
$$

Constant biases and linear maps disappear under polarization, so adding either for
negligible executable cost cannot evade this lower bound.  Rescaling or changing the
internal gauge also leaves the represented $A_y$ unchanged.

The checker selected the deterministic coordinate $y=e_0$, constructed $A_{e_0}$
from the exact BF16 checkpoint coefficients in float64, and computed all singular
values.  The smallest singular value exceeded a deliberately conservative floating-
point error allowance at all three sites, so each tested slice has numerical rank
1,152.  The margins for the decisive 513th singular value were about
$1.59\times10^9$, $1.67\times10^9$, and $1.63\times10^9$ times the allowance.

| Site | $\sigma_{513}(A_{e_0})$ | Numerical slice rank | Minimum exact product count | Best rank-512 relative slice error |
|---:|---:|---:|---:|---:|
| MLP0 | 7.09357 | 1,152 | at least 1,152 | 0.28528 |
| MLP1 | 7.66801 | 1,152 | at least 1,152 | 0.30171 |
| MLP2 | 6.86038 | 1,152 | at least 1,152 | 0.29492 |

The last column follows from the Eckart--Young theorem: the optimal rank-512 matrix
approximation has squared Frobenius error equal to the sum of squared singular values
after the first 512.  Since any 512-product quadratic program induces a slice of rank
at most 512, it cannot do better on this slice.

### Exact claim boundary

This is a strong backward-error numerical certificate, not a symbolic determinant or
interval-arithmetic proof.  The enormous margins make floating-point reversal
implausible, but an exact finite-field minor is the proof-grade upgrade if needed.

It certifies a **global algebraic obstruction**.  It does not say how frequently
natural text visits the missed directions, how large the CE penalty is, which
directions are semantic, or whether a rank-512 student transports OOD.  Its useful
prediction is that adversarial or sufficiently broad OOD residual directions exist on
which every 512-product replacement differs materially from the native map.

Artifact: `early_mlp_quadratic_slice_rank_certificate.json`, SHA-256
`cac26ab42eeb0a977aacfcc98ff22b42a6d39e5e008091c37e2814850bca41fb`.

## Ranked move 1 — observable-weighted tensor certificates

### Exact object

Use the same MLP0/1/2 polarization tensor, but measure its slices only after projection
onto (a) input directions actually reachable at the site and (b) output directions
that verified downstream consumers can observe.  If $P$ is an empirical reachable
metric and $Q$ is a downstream-observability metric, analyze singular values of

$$
Q^{1/2}A_yP^{1/2}
$$

across native and compressed-prefix trajectories.  This converts “simple” from low
raw coefficient rank into low *causally relevant* rank while retaining a matrix-rank
certificate.

### Mathematics and assumptions

Empirical controllability/observability Gramians and balanced truncation rank states
by joint reachability and observability.  For stable linear systems, discarded Hankel
singular values control input-output error.  Empirical nonlinear balancing estimates
the analogous directions using finite perturbations; see
[Condon & Ivanov](https://arxiv.org/abs/math/0606430) and
[Kawano & Scherpen](https://arxiv.org/abs/1902.09836).

The transformer suffix is neither linear nor a stable time-invariant dynamical system,
so the classical global error theorem does not automatically apply.  $P$ depends on
the text/intervention distribution, $Q$ depends on the consumer bank, and RMSNorm,
softmax, and residual interactions can make a local metric inaccurate at deletion
scale.

### Prediction beyond reconstruction

If this is the right simplicity notion, the leading weighted subspace should be shared
between native and C512 trajectories and should predict held-out suffix CE, the
MLP0×MLP2 interaction, low-rank attention-5/6 behavior, and selective edits better
than equal-price local-MSE PCA/RRR.  Weighted tail singular values would also provide
a local response-error certificate in the measured consumer metric.

### Cheapest falsifier

On 64 unique documents, build randomized JVP/VJP sketches from the MLP2 input to a
small verified output bank: final centered logits plus the attention-5/6 response and
one late behavior readout.  Estimate $P,Q$ on native and C512 backgrounds.  Reject
the cheap common-state hypothesis if (1) no spectral knee appears, (2) the leading
subspaces have poor split/background agreement, or (3) a held-out finite response is
not predicted better than equal-price local MSE.  This differs from the failed E3.1
pilot: E3.1 used an undifferentiated final-output tangent panel; this test uses
verified low-capacity consumers, finite checks, and two upstream environments.

## Ranked move 2 — intervention-complete causal quotient

### Exact object

Define two early residual states $x,x'$ as equivalent only when replacing one with the
other preserves a specified vector of downstream consumer responses under a specified
family of allowed suffix interventions:

$$
x\sim_\varepsilon x'
\quad\Longleftrightarrow\quad
\sup_{a\in\mathcal A}
d\!\left(G_a(x),G_a(x')\right)\le\varepsilon.
$$

$G_a$ includes final-token prediction and named copy, capitalization, numeric, syntax,
entity, and attention-5/6 response tests; $a$ includes component replacements and
compositions.  This makes a “cluster” downstream-operational: tokens may remain
separated in raw MLP0 space yet be the same abstract state if every registered
consumer treats them alike.

### Mathematics and assumptions

Approximate causal abstraction formalizes when a high-level model commutes, up to an
error metric, with interventions in a low-level model; see
[Beckers, Eberhardt & Halpern](https://proceedings.mlr.press/v115/beckers20a.html).
Recent work studies semantic embedding and identifiability of such abstractions
([D'Acunto et al.](https://proceedings.mlr.press/v267/d-acunto25a.html),
[Li et al.](https://proceedings.mlr.press/v258/li25g.html)).

The dangerous assumption is completeness: a small consumer/intervention bank can
declare false equivalences.  The quotient also need not be linear, and approximate
equivalence may fail transitivity without careful construction.

### Prediction beyond reconstruction

A useful quotient must predict a held-out consumer, commute with unseen compositions,
and enable selective removal with a measured off-target bound.  Those are benefits an
ordinary reconstruction metric does not promise.

### Cheapest falsifier

Learn the quotient using all but one verified consumer and half the intervention
compositions.  Test the held-out consumer and held-out compositions.  Compare at equal
code/storage price with PCA, an SAE/dictionary, and token-class baselines.  If quotient
distance does not predict withheld causal behavior, the chosen consumer bank is not a
valid notion of simplicity.

The immediate blocker is experimental, not theoretical: the project still lacks the
multi-behavior late consumer bank.  The attention-5/6 low-rank interface is now the
best first additional consumer; capitalization/numeric/syntax/entity tests should
follow.

## Ranked move 3 — finite Hankel/minimal realization after the consumer bank

### Exact object

Build a matrix whose rows are early prefixes or controlled early interventions and
whose columns are future suffix tests/continuations.  Each cell records a scalar or
small vector consumer response.  The matrix is a finite block of a Hankel operator.

For rational weighted automata, Hankel rank equals the dimension of a minimal linear
realization; spectral algorithms recover that realization from finite blocks.  Useful
primary references are the weighted-automata minimal-rank construction of
[Balle and Mohri](https://papers.nips.cc/paper/4697-spectral-learning-of-general-weighted-automata-via-constrained-matrix-completion.pdf)
and the finite-state spectral framework summarized by
[Arrivault et al.](https://proceedings.mlr.press/v57/arrivault16.pdf).

The transformer is a nonlinear continuous-state finite-context process, not known to
be a rational series.  Therefore a finite low-rank block is only an operational lower-
complexity model, not proof of the transformer's exact minimal state.

### Prediction and falsifier

A low-rank realization must predict sealed prefix×suffix cells, new continuation
templates, and intervention compositions at a smaller executable price.  The cheapest
falsifier is a held-out block-completion test on the attention-5/6 response plus at
least two independent late behaviors.  Running it on copy alone would merely cap rank
by the number of probes, so this move is third, not first.

## Other mathematics reconsidered and pruned

- **Plain tensor rank, CP/Tucker/HOSVD, and shared dictionaries:** the new slice result
  proves why rank 512 cannot be globally exact, while Family F, global RRR, and the
  rank-512 hierarchy already show that unweighted local reconstruction does not buy
  composition.  Repeating these objectives is pruned.  Simultaneous factorization
  remains useful only inside a common observable metric.
- **Gauge quotients and norm-minimized HOSVD:** tensor-network canonical forms can
  remove representation gauge and identify equivalent parameterizations; see the
  minimal canonical form of [Acuaviva et al.](https://arxiv.org/abs/2209.14358) and
  loop gauge fixing by [Evenbly](https://arxiv.org/abs/1801.05390).  But gauge changes
  preserve $A_y$ and therefore cannot lower the certified functional slice rank.
  They should canonicalize a successful program, not generate the missing causal
  abstraction.
- **Algebraic complexity:** multiplicative product count is now a real lower-bound
  currency, but global arithmetic-circuit complexity is too strict for natural-text
  prediction.  Retain the slice certificate as one axis; do not make it the only
  simplicity score.
- **MDL/prequential coding:** prequential MDL tests whether a representation makes
  downstream tasks sample-efficient rather than merely linearly decodable
  ([Bornschein et al.](https://arxiv.org/abs/2210.07931)).  This is a valuable
  validation currency once candidate states exist, but it does not produce the
  observable basis by itself.
- **Information bottleneck:** consumer sufficiency is the useful part, but mutual
  information alone neither prices an executable tensor program nor ensures
  intervention equivalence.  Subsumed by the causal quotient.
- **Sparse program synthesis and SAE/dictionary learning:** retain as a decoder for a
  validated observable state.  Weight-only sparsity is not itself evidence of OOD
  transport or selective removal, and the earlier shared/global attempts were
  negative.
- **Worst-case approximation certificates:** the global slice certificate is useful
  and cheap.  Whole-suffix Lipschitz bounds remain too loose across RMSNorm and
  softmax; certify projected interfaces and empirical cells before attempting a
  network-wide norm product.

## Simplicity is now explicitly two-dimensional

No single scalar is currently justified.  Every candidate should carry at least:

1. **algebraic/executable price:** product count, stored coefficients, operations, and
   any tables or native calls; and
2. **behavioral sufficiency:** held-out CE/KL, OOD transport, unseen composition,
   extraction, and selective-removal/off-target performance.

The slice-rank certificate validates the first axis.  Observable balancing, causal
quotients, and Hankel completion test whether reductions on that axis buy capabilities
on the second.  A representation is called simpler only when its lower price enables
one of those operations at matched predictive quality.

## Next executable action

The next project-critical GPU action remains the already-frozen fresh eight-arm
physical evaluation of old FULL512, CONTINUE512, and ROBUST512, alone and with C512.
It decides whether exposure fitting repairs the measured interface penalty; the dev
fit suggests it probably will not, but the sealed test must decide.

The next *new mathematical* action is the 64-document observable-sketch falsifier in
Ranked move 1.  Before opening it, the attention-5/6 scalar response must be exposed
through the reusable intervention harness and one unrelated late behavior added, so
the result cannot be a one-probe rank artifact.

## Reproducibility

- Checker: `certify_early_mlp_quadratic_slice_rank.py`
- Tests: `test_certify_early_mlp_quadratic_slice_rank.py` — **2 passed**
- Result: `early_mlp_quadratic_slice_rank_certificate.json`
- Result SHA-256: `cac26ab42eeb0a977aacfcc98ff22b42a6d39e5e008091c37e2814850bca41fb`
- Checkpoint SHA-256: `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`
- Result runtime: **15.710 s** on CPU

