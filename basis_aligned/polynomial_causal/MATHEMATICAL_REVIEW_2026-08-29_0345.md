# Three-hour mathematical review — 2026-08-29 03:45 UTC

## Outcome-changing update

The covered-fit rank-512 stream map completed during this review.  Against the
uncovered-token ceiling it leaves `0.174/0.214/0.214` nat, versus
`0.596/0.672/0.672` for the same-rank embedding map and `0.114/0.141/0.142` for
the uncovered-fit stream oracle.  Covered CE is bit-identical.  Rank 512 from the
embedding is already useful; changing the input at the same rank is substantially
more useful.

Source inspection changes the operational interpretation.  The fitted parameters use
covered tokens only, but the evaluation inputs for uncovered tokens are the **native
length-one streams**, produced by native forwards.  The experiment proves cross-token
transfer conditional on those features.  It does not yet prove that a standalone
compressed program can produce its own inputs.  No whole-program or global causal
ledger moves.

Stable ledgers remain: structural inventory 36/36, certified whole-program storage
removal 5.3481%, strict named causal CE recovery 10.923% with 4.72714 nat remaining,
and terminal extraction/removal/OOD actions 0/68.  Family F has no numerical outcome
and remains source-audit blocked rather than data/GPU blocked.

A rank-1024 control then recovered only `0.028/0.037/0.024` additional stream-map nat
per uncovered position for 42,467,328 additional factor floats; the embedding map
moved only about `0.008`.  The factor representation is already more expensive than a
dense map at that rank.  This closes “increase rank again” as a high-return move and
leaves rank 512 as the operating point, conditional on input closure.

## Ranked top three genuinely new mathematical moves

### 1. Closed predictive realization for the stream map

**Exact bilin18 object.**  The sequence of length-one states and site outputs

$$
x_0(t)\longrightarrow y_0(t)\longrightarrow x_1(t)\longrightarrow\cdots
\longrightarrow y_{35}(t)
$$

used to generate context-free rows for token (t).

**Operational definition.**  A compressed state is *closed* if every input consumed
by a compressed site is produced by already retained/compressed operations, with no
native-model call or unpriced token table.  Two states are behaviorally equivalent
for an allowed test family if every allowed suffix intervention produces the same
output distribution.  Quotienting by that equivalence is the causal/bisimulation form
of a minimal interface.

For linear controlled systems and weighted automata, finite Hankel rank characterizes
the dimension of a minimal realization.  Predictive-state representations express
state in terms of future-test predictions rather than inaccessible latent variables.
The exact theorem does not transfer automatically to a nonlinear transformer, but it
supplies the right success criterion: predict all registered downstream tests from a
state that the compressed prefix itself computes.  See the original
[Carlyle–Paz realization paper](https://doi.org/10.1016/S0022-0000(71)80005-3) and
[Littman–Sutton–Singh predictive-state paper](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html).

**Assumptions that may fail.**  The finite intervention family may not be closed under
composition; the transformer suffix is nonlinear; approximate rank can be unstable;
and the recursively compiled stream may drift far from the native length-one stream.

**Prediction beyond reconstruction.**  If the state is closed, the whole context-free
program runs without original-model calls and preserves both ordinary CE and finite
downstream tests.  If native-stream performance disappears under the recursively
compiled stream, the current result is feature leakage rather than executable
compression.

**Cheapest falsifier.**  At rank 512, replace `ORC['stream']` by the site-entry stream
produced by the already-built fully compiled length-one prefix.  Fit on the same 5,419
covered tokens and score the same disjoint uncovered positions.  Require covered CE
bit identity and preregister a hard comparison to the embedding-rank-512 deficits.  No
rank sweep is justified before this check.

### 2. Simultaneous reduced-rank regression with a shared output dictionary

**Exact bilin18 object.**  The 36 stream-to-row maps

$$
W_j:\mathbb R^{1152}\to\mathbb R^{1152}.
$$

Rather than 36 unrelated factorizations (W_j=A_jB_j), use

$$
W_j=A_jV^T,
$$

where the output basis (V\in\mathbb R^{1152\times q}) is shared and only
(A_j\in\mathbb R^{1152\times q}) is site-specific.

**Exact theorem/solution.**  Let (G_j=X_j^TX_j) and (C_j=X_j^TY_j).  For the
joint ridge objective

$$
\sum_j\left(\|Y_j-X_jA_jV^T\|_F^2+\lambda\|A_j\|_F^2\right),
\qquad V^TV=I,
$$

the global optimum has

$$
A_j=(G_j+\lambda I)^{-1}C_jV,
$$

and (V) is the top-(q) eigenspace of

$$
M=\sum_j C_j^T(G_j+\lambda I)^{-1}C_j.
$$

This is simultaneous reduced-rank regression, closely related to classical
[reduced-rank regression](https://doi.org/10.1016/0047-259X(75)90042-1).  It is a
predictively weighted alternative to applying
[HOSVD](https://www.math.ucdavis.edu/~saito/data/tensor/lathauwer-etal_mulilinear-SVD.pdf)
to the raw coefficient tensor: site input covariance determines which shared output
directions matter.

**Assumptions that may fail.**  Squared row error may not track CE; attention and MLP
sites may not share one output space; the shared eigenspace may have no stable gap; and
RMSNorm/contextual composition may amplify discarded directions.  Separate attention
and MLP dictionaries are a prospectively specifiable fallback, not a post-outcome
choice.

**Prediction beyond reconstruction.**  At (q=512), separate factors store
42,467,328 floats.  One shared output basis stores 21,823,488, a 48.61% reduction,
with the same 1,179,648 dense multiplies per site.  If held-out CE remains within a
frozen tolerance, the result strictly improves executable storage and creates one
common coordinate system for later sparse/semantic tests.

**Cheapest falsifier.**  Retain only (G_j,C_j), compute the exact shared basis on CPU,
and evaluate (q=256,512) on the existing three roles.  The first prospective gate
should require shared-(q512) uncovered CE to stay within `+0.01` nat of independent
rank 512 while covered CE stays bit-identical.  Compare also to an iso-storage
independent rank near 263.

**Action executed.**  Implemented
`simultaneous_shared_output_rrr.py` and its adversarial/synthetic tests.  The tests
verify exact recovery of a known shared subspace, global-objective dominance over a
random subspace, full-rank replay of independent ridge maps, rotational gauge
invariance, validation failures, and the literal price.  Result: 8/8 CPU tests pass.

### 3. Vector-valued Hankel/predictive-state interface for composed interventions

**Exact bilin18 object.**  A response matrix whose rows are early-prefix interventions
(the 34 MLP0/MLP1 configurations or exact typed gate edits) and whose columns are
downstream tests: selected logit groups, finite MLP2 backgrounds, residual directions,
and suffix interventions.  Each entry must be a vector response or a fixed scalar
projection, not only aggregate CE.

**Operational definition.**  The rank of a closed action/test Hankel matrix is the
number of linear predictive coordinates needed to reproduce all measured
prefix–suffix compositions.  A candidate interface is useful only if a basis chosen on
discovery rows predicts unseen action/test compositions and transports to disjoint
documents.

**Assumptions that may fail.**  The current 68 actions are not a mathematically closed
semigroup; nonlinear response may require amplitude-indexed tests; rank can change
with the chosen tests; and the old 8×8 scalar-CE cross already showed unstable low
rank.  Hence the construction must use fresh vector outcomes and a prospective
action/test split.

**Prediction beyond reconstruction.**  Success predicts unmeasured compositions,
defines a behavioral equivalence class for selective removal, and yields an explicit
minimal causal port.  These are direct extraction/edit/OOD currencies, unlike local
weight MSE.

**Cheapest falsifier.**  Before all 68 final actions, use a small fresh discovery role
and 8–16 fixed logit/residual tests.  Fit rank (r=2,4,8,16) on a checkerboard of
prefix/test pairs; score completely untouched pairs and a document-disjoint role.
Reject if it does not beat additive and independent low-rank baselines or if singular
subspaces are unstable under document bootstrap.

## Other promising mathematics, with pruning decision

| Idea | Exact object | Beyond-reconstruction consequence | Cheapest falsifier | Decision now |
|---|---|---|---|---|
| Odd/even Volterra or information-geometric secants | MLP3 candidate error (\delta) through Blocks 4–17 | Separates even Fisher/Gauss–Newton response from cubic/odd sign response; predicts edit asymmetry | Evaluate (K(\pm\delta/2)) and (K(\pm\delta)) for fixed typed directions | Keep as the first post-candidate port diagnostic; do not complicate Family-F fitting. |
| Gauge norm minimization then HOSVD | Bilinear Left/Right/Down tensor or map factors | Could choose balanced representatives before comparing factors | Toy recovery under known reciprocal gauges, then held-out CE | For linear maps, balancing factors does not change the invariant (W_j), and predictive RRR is exact for the relevant loss. For bilinear weights, Family A shows local HOSVD error is insufficient. Deprioritize. |
| CP/Tucker/arithmetic-circuit rank of MLP3 | Exact 4,608-gate polynomial | Fewer multiplications and literal bytes | Held-out suffix KL at equal product count | Family F is already the consequence-aware sparse arithmetic-circuit test. Another local factorization duplicates it. |
| Weight SAE/dictionary learning | MLP0/1/2 factors or shared map basis | Sparse named atoms, selective edits | Does atom support predict held-out response classes and survive ablation? | Wait for a shared predictive coordinate system; weight reconstruction alone has no validated utility. |
| MDL/prequential coding | Shared-map ranks, dictionaries, routers | Charges fitting/search cost rather than only deployed bytes | Code discovery labels/documents sequentially and compare predictive codelength | Useful tie-breaker after two candidates pass consequence tests; not a mechanism finder now. |
| Information bottleneck | Candidate interface state versus downstream tests | Minimal sufficient predictive state | Estimate held-out test prediction under decreasing state dimension | Operationally subsumed by the prospective Hankel/state test; mutual-information estimation would add fragility. |
| Sparse program synthesis | Native gate subsets and context routers | Executable, editable branch program | Beat Family-F controls under literal price | Family F already covers the first sparse gate grammar. Add a router only if global support fails and gradient covariance is low rank. |
| Approximation certificates | Closed map and admitted MLP3 port | Certified response bounds under finite edits | Interval/Jacobian bounds on frozen directions and amplitudes | High value after a candidate exists; premature before input closure/Family-F admission. |

## Why this ranking differs from the previous review

The rank-512 result removes the immediate case for nonlinear embedding maps: a linear
stream map already transfers most of the oracle gain.  But the source audit reveals
that the feature producer is outside the claimed compressed program.  Therefore
closed realization outranks more rank or nonlinear capacity.

If closure passes, shared predictive factorization is the cheapest route to make the
42.47M-float solution genuinely simpler and more canonical.  The response-Hankel move
then supplies the missing whole-model composability/editability criterion.  Family F
continues independently because it is already frozen and tests a different exact
bilinear interface.
