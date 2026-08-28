# Mathematical review: laws, hierarchies, and task-universal sufficiency

Date: 2026-08-28 16:30 UTC

Status: repository-specific review, source-only CPU implementation, proof tests, and
prospective preregistration. No new model outcome or data role was opened.

## What changed since the last mathematical review

The earlier mathematical moves now have discriminating evidence:

- Generic prefix/continuation Hankel state failed badly out of distribution: token
  splices added 3.54--3.61 CE, empirical rank stayed roughly 19--24 of 48, and low-rank
  structure improved only 4.5--10.1% over the additive baseline.
- Independent tangent frames failed stability: every measured half retained full
  numerical support, there was no registered twofold gap, and the same-context
  split-half rank-16 projector distance was 0.5621 against a 0.15 ceiling.
- Native MLP1 gate supports of 32, 128, and 512 did not earn causal promotion.
- Coefficient Tucker/HOSVD is dense at both MLP1 and MLP2: all three numerical mode
  ranks are 1152, and the registered dense approximations lose on storage/products.
- MLP4 joint diagonalization raised diagonal mass from 0.041 to 0.203, but added only
  0.005 causal gain over the linear control. Weight concentration alone is therefore
  not the target.
- The current low-table compiler result is a useful but different discovery: at table
  rank 1, map ranks 8, 16, 64, and 256 have identical top-1 to five printed decimals
  on all three roles. The explanation is algebraic, not empirical coincidence: if the
  table has rank $r$ plus a mean, its learned row map has rank at most $r+1$. The
  correctly priced rank-1 program is therefore 0.485M values with a rank-2 map, not
  the previously reported 5.628M values with a fictitious rank-64 capacity. At table
  rank 64, map rank 8 loses only 0.31--0.42 percentage points, so the old >=1 point
  “map floor” does not reproduce on accuracy. This discovery prices a current program;
  it does not explain the early computation, and its CE audit is still running.

The honest global ledgers remain: 36/36 modules have structural formulas; certified
whole-program storage removal is 5.3481%; strict named causal recovery is 10.923%;
and recovery of the current paired +0.8976 CE ship gap is zero. Held-out MLP0--2 owns
0.7277/0.8727 nats of the global ship loss and 43--64% of measured token-cell effects
are interactions. The missing object is therefore a compact **interface law** that
MLP1, MLP2, and the suffix can actually consume.

No checkpoint, FineWeb, `rspd`, cache, GPU, or network blocker exists. The suffix
lifecycle, exactly-once loading, reduced-statistic envelope, and adversarial replay now
pass a 224-test CPU suite. The immediate obstruction is narrower: the existing adapter
is validation-phase locked and does not expose a typed final intervention/gauge action
and final observation reduction. Scientific roles must stay sealed until that
capability and its adversarial tests exist.

## Ranked move 1: typed finite observable closure

### Object

The object is the consecutive early-interface trajectory: the selected MLP0 code,
the actual RMSNorm scalar and residual state entering MLP1, the MLP1 code/write, and
the registered MLP2/suffix response under physical edits. It is not the token-prefix
Hankel matrix and not an independently fitted tangent frame at each document.

### Mathematics

If an observable span is invariant under a dynamical map, its observables evolve by a
finite linear operator. This is the finite-dimensional Koopman construction. Dynamic
mode decomposition with control gives the empirical least-squares version. Crucially,
finite invariant spans containing the original state are restrictive rather than
automatic; failure is expected and informative. See
[Brunton et al. 2016](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0150171).

For bilin18 we use layer-specific operators and an explicit grammar containing the
constant, code, edit, inverse-RMS scalar, typed scalar products, and selected bilinear
monomials. Reduced-rank regression minimizes

$$
\lVert(Y-XK)G^{1/2}\rVert_F^2,
$$

where $G$ weights directions by registered downstream consequence. For two layers the
exact residual identity

$$
Y_2-X_0K_0K_1=e_1+e_0K_1
$$

turns local errors into a finite composition bound. This is the new piece: earlier
work measured response rank or fit components independently; it did not demand a
closed observable algebra whose error transports through the next fitted law.

### Assumptions that may fail

There may be no useful finite invariant dictionary. Attention makes the state depend
on other positions; the transition is layer-dependent; RMSNorm is rational rather
than polynomial unless its scalar is carried explicitly; chosen monomials can overfit;
and a Fisher metric can miss rare output decisions. Every one of these is handled as a
rejection route, not hidden in a flexible decoder.

### Consequence beyond reconstruction

A passing state predicts unseen edit mixtures and the two-step MLP0-to-MLP2/suffix
response. It supplies an executable sequence of small tensor/linear operators, a
certificate for accumulated error, and a basis in which selective edits have predicted
downstream effects. Local activation MSE cannot supply those consequences.

### Cheapest falsifier and executed action

Fit affine, RMS-typed, and controlled-polynomial dictionaries at ranks
4/8/16/32/64. On held-out actions, require the composed law to be within 10% of a
direct rank-and-cost matched map and to beat rotated/affine controls by at least 5%.
Reject if data doubling is unstable or a live consumer's norm explodes.

This review executed the safe source-only part: `typed_koopman_closure.py` implements
the exact metric-weighted reduced-rank solution and two-step identity; five synthetic
proof/validation tests pass. The full prospective contract is
`TYPED_EARLY_OBSERVABLE_CLOSURE_PREREGISTRATION.md`.

## Ranked move 2: causal successive-refinement rate--distortion

### Object

This applies to the proposed MLP0 hierarchy “mean + lexical class + continuous token +
context refinement.” The question is whether those levels form a genuinely useful
nested representation, rather than an appealing naming scheme.

### Mathematics

At budgets $R_1<R_2<\cdots$, independently optimized representations attain
distortions $D_1,D_2,\ldots$. A source is successively refinable only when those
rate--distortion optima can be coupled as a Markov refinement chain; otherwise forcing
one nested code incurs refinement regret. This is the operational theorem of
[Equitz and Cover 1991](https://isl.stanford.edu/~cover/papers/paper94.pdf).

For bilin18, rate includes the entropy/description of token memberships, continuous
coefficients, producer, and downstream decoder. Distortion is registered suffix KL/CE
plus finite-edit response error, not MLP0 write MSE.

### Assumptions that may fail

The transformer source is not memoryless; the chosen distortion is task-dependent;
the optimum is computationally inaccessible; and learned nested quantizers can be
stuck in worse local minima. We therefore compare nested and independently optimized
codes with matched optimizer budget and multiple source-frozen restarts, rather than
claiming an information-theoretic optimum.

### Consequence beyond reconstruction

A small refinement regret would validate progressive extraction: a cheap lexical code
can be deployed first and token/context detail added without changing the earlier
meaning. It also makes selective removal testable by cutting a refinement branch. A
large regret would tell us that “mean/class/token/context” is not the right simplicity
order even if every individual fit looks good.

### Cheapest falsifier

On frozen response rows, fit nested and independent codebooks at identical bit/value
budgets and measure held-out causal distortion. Reject the hierarchy if its excess
distortion is larger than sampling uncertainty at two adjacent budgets or reverses on
replication. This can begin CPU-side after the suffix observation type is available.

## Ranked move 3: finite Blackwell/Le Cam deficiency

### Object

Treat the native and compressed early interfaces as two finite experiments: the state
of nature is a registered physical intervention, and the observation is a discretized
downstream response or probability vector. Ask whether a decoder from the compressed
observation can reproduce the native experiment.

### Mathematics

Blackwell's comparison says one finite statistical experiment is at least as
informative as another exactly when it can achieve no worse risk in every decision
problem, equivalently when the latter is a stochastic garbling of the former
([Blackwell 1953](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-24/issue-2/Equivalent-Comparisons-of-Experiments/10.1214/aoms/1177729032.full)).
An operational approximate version is the deficiency

$$
\delta(C,N)=\inf_K\sup_a
\operatorname{TV}\!\left(P_N(\cdot\mid a),KP_C(\cdot\mid a)\right).
$$

Unlike one CE task, small deficiency controls the risk gap for every bounded decision
problem represented by the registered action/observation bank.

### Assumptions that may fail

A finite intervention bank is not universal; discretization can hide information; an
unpriced stochastic decoder can memorize the bank; and exact total variation over the
vocabulary can be expensive. Decoder storage and held-out actions must therefore be
priced, and coarse output bins are only a screen.

### Consequence beyond reconstruction

This tests task-universal downstream sufficiency. A small, cheap deficiency decoder
would justify transferring extraction, classification, and removal decisions from the
native interface to the compressed one; a CE-equivalent representation with large
deficiency would be exposed as task-specific.

### Cheapest falsifier

Solve the finite stochastic-decoder linear program on a small frozen bank of covariance
edits and output bins, then test its risk/TV on held-out interventions. Compare against
a representation selected to match CE. Reject if the decoder cost erases compression
or deficiency/risk bounds fail on the held-out bank.

## Pruning ledger for the requested mathematical families

| Family | Decision now | Reason |
|---|---|---|
| Tensor rank / HOSVD / CP | Dense coefficient HOSVD is pruned; empirical metric-weighted structure remains subordinate to move 1. | MLP1/2 coefficient modes are numerical rank 1152 and dense Tucker points lose storage/products. CP has no current stability or causal certificate. |
| Arithmetic-circuit rank | Defer as a lower-bound language, not the next experiment. | Exact polynomial gate count is useful, but approximate circuit rank across RMSNorm/attention has no cheap identifiable optimum. Closure gives a falsifiable upper construction first. |
| Simultaneous factorization / shared dictionaries | Prune generic joint diagonalization; allow only a shared dictionary that improves composed closure. | MLP4 diagonal mass rose fivefold for only 0.005 causal gain; exact commutants were already noise-fragile in A2 work. |
| Gauge quotients / invariant theory | Keep as a constraint on every move, not a standalone ranking metric. | Orthogonal code gauges and native gate scale/permutation gauges can make coordinates look sparse without changing the program. The implemented loss is output-gauge invariant when the metric transforms with it. |
| Algebraic complexity | Use monomial count and multiplication count as prices after causal passing. | A smaller symbolic circuit that the suffix cannot consume is not progress. |
| System identification / minimal realization | Generic tangent realization is pruned; typed finite observable closure is promoted. | Independent tangent frames failed. Closure asks a different, stronger compositional question with finite actions and explicit RMS state. |
| Hankel / automata | Pruned in the tested token-splice form. | Severe OOD CE and no compact stable rank. Rephrasing the same splice as an automaton adds no information. |
| MDL / prequential coding | Keep only as a tie-breaker among causally valid programs. | Description length cannot validate its own semantics; decoder/action metadata must be charged. |
| Causal abstraction / bisimulation | Operationalized by moves 1 and 3; do not fit another unconstrained abstraction. | Closure checks transition consistency; deficiency checks downstream decision sufficiency. |
| Information bottleneck | Deterministic IB is pruned. | It can reward arbitrary invertible reparameterizations or task-specific deletion and does not ensure executable composition. |
| Sparse program synthesis | Defer generic search; restrict synthesis to the frozen typed grammar. | Unconstrained synthesis has a vast multiple-testing surface and can optimize validation CE without stable interfaces. |
| Approximation certificates | Promote the two-step residual identity now; retain global Lipschitz bounds only if locally calibrated. | The identity is exact and cheap. Worst-case network norm products are too loose to rank current candidates. |

## Why these three, in this order

1. **Typed observable closure** has the highest information gain: it directly tests
   whether the MLP0 code participates in a small composable law, uses polynomial and
   RMS tensor structure, and can fail cheaply before a compiler run.
2. **Successive refinement** is the cleanest validation of the proposed lexical
   hierarchy and turns “simplicity” into a measurable deployment advantage, but it
   depends on having trusted downstream distortion rows.
3. **Blackwell/Le Cam deficiency** is the strongest task-universal notion of
   sufficiency and editability, but its finite discretization and decoder price make it
   a later validation layer rather than the first representation finder.

The current suffix executor remains priority infrastructure because all three moves
need a sealed downstream observation. The closure CPU kernel does not bypass that
lifecycle; it makes the next collected rows answer a more compositional mathematical
question.
