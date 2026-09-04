# Three-hour mathematical review — 2026-09-04 12:30 UTC

## 1. The model, target, and price are different mathematical objects

The model remains an 18-block tensor contraction with residual width $D=1152$, nine attention heads of width
$d_h=128$, and bilinear-MLP product width $M=4608$. For a prompt of length $T$, a block-boundary residual is

$$
R_l\in\mathbb R^{T\times1152}.
$$

For head $h$ and positions $i,j$, double-bilinear attention computes

$$
a^h_{ij}=
\frac{\langle q^h_i,k^h_j\rangle}{128}
\frac{\langle \widetilde q^h_i,\widetilde k^h_j\rangle}{128},
\qquad
o^h_i=\sum_{j\le i}a^h_{ij}v^h_j.
$$

One normalized MLP input $z\in\mathbb R^{1152}$ is mapped by

$$
F_l(z)=W_{D,l}\big[(W_{L,l}z)\odot(W_{R,l}z)\big]+b_l,
$$

where $W_{L,l},W_{R,l}\in\mathbb R^{4608\times1152}$ and
$W_{D,l}\in\mathbb R^{1152\times4608}$. Conditional on normalized inputs, the MLP is degree two and the attention
value term is degree five in its local projected inputs. The complete network is not polynomial in raw tokens because
RMS normalization appears repeatedly and the logits are soft-capped by $30\tanh(\ell/30)$.

The live task-14 observable is not a reconstruction norm. It is the signed behavioral contrast

$$
m(x)=\ell_{\texttt{ are}}(x)-\ell_{\texttt{ is}}(x)
$$

under registered natural-donor residual interchanges on the frozen FIT prompts. A rank-$k$ interchange uses a
projector $P=UU^{\mathsf T}$,

$$
r'(x\leftarrow d)=r(x)+P(r(d)-r(x)).
$$

The scientific scores are held-out signed transfer, same-state leakage, necessity, two-site interaction, and ordered
reader reset/rescue. No replacement-model approximation norm is currently licensed. SELECT, TEST, and OOD remain
closed.

The physical experiment is a second object: a finite conditional program with 3,821 possible chunks and 743,881
possible call identities. Its current branch-complete upper bound is 60,000 optimizer updates, 119,207 forwards,
60,004 backwards, 9,207,984 evaluated sequences, 63,782,508 tokens, and 63,394,944 retained numeric bytes. The hard
wall-clock allowance is 28,800 seconds. These prices constrain the experiment; none identifies a circuit.

The gauges are unchanged. $U$ and $UO$ define the same projector for every $O\in O(k)$; rank one is invariant to
$u\mapsto-u$. Bilinear product units can be permuted, the two input branches exchanged, and factors reciprocally
rescaled with compensating output weights. Attention QK and OV bases have reciprocal changes of basis. Therefore raw
head coordinates or product indices are not semantic units. The candidate invariant is equivalence under registered
downstream response.

## 2. The compiler is an adapted causal program, not a static call list

Let the experiment have ordered stages $s=0,\ldots,S$. Let $E_s$ be the exact evidence emitted after stage $s$, and
let

$$
\mathcal F_s=\sigma(E_0,\ldots,E_s)
$$

denote all information legitimately available then. A stage activation decision $A_{s+1}$ is prospective exactly
when it is **adapted**:

$$
A_{s+1}=\pi_{s+1}(E_0,\ldots,E_s),
\qquad A_{s+1}\text{ is }\mathcal F_s\text{-measurable}.
$$

In plain terms, a decision may read frozen inputs and earlier receipts, never a result that will be produced later.
Compiler v2 violated this condition. Its linear chunk index placed selected-family fits for an early boundary before
later joint rank-one fits had finished, although those joint fits determine the selected boundary. Its replay API
also required the final selection/necessity/reader state before the first call. The file was deterministic, but the
policy was anticipatory.

This maps exactly to deterministic dataflow. In [Kahn's original process-network semantics](https://www.cs.columbia.edu/~sedwards/papers/kahn1974semantics.pdf),
deterministic processes communicate through ordered channels; a process blocks until its required input exists. Our
stage receives a predecessor receipt, emits evidence and a one-use successor capability, and cannot read a future
channel. Kahn's result supplies determinacy when processes themselves are deterministic and channel use respects the
network. It does **not** prove the neural experiment scientifically valid, and its unbounded-stream setting is more
general than our finite acyclic protocol.

[Session types](https://di.fc.ul.pt/~vv/papers/honda.vasconcelos.kubo_language-primitives.pdf) provide a neighboring
formalism: a typed communication protocol makes incompatible interaction sequences unrepresentable. The relevant
translation is a linear capability

$$
C_s\xrightarrow{\text{replay stage }s}R_s
\xrightarrow{\text{validate evidence }E_s}C_{s+1}.
$$

Neither $C_s$ nor $R_s$ may be consumed twice. This is why a hash string is insufficient: hashes bind content but do
not enforce single use. A process-local sealed capability with an internal consumed flag is the executable analogue.
The full session-type theorem assumes a typed calculus; Python does not provide that theorem automatically, so the
runtime validator and duplicate-use adversarial tests are still required.

## 3. A static non-anticipation test

For every stage $s$, freeze three finite sets:

$$
\operatorname{reads}(s),\qquad \operatorname{writes}(s),\qquad
\operatorname{calls}(s).
$$

Let $F$ be the frozen preflight fields. A proposed topological order is legal iff

$$
\operatorname{reads}(s)\subseteq
F\cup\bigcup_{t<s}\operatorname{writes}(t)
$$

for every stage, every written field has exactly one defining stage, and every branch-specific stage is dominated by
the stage writing its guard. A depth-first cycle test plus a topological pass checks these conditions in
$O(|V|+|E|+K)$ time, where $K$ is the total number of declared field references. The check is exact for this finite
compiler graph.

Applied to v2, `selected_h` and `selected_q` are read by selected-fit chunks before the complete joint-fit evidence
that writes them, so the test fails immediately. A v3 manifest should include these read/write sets as hash-bound
data, not infer them from English guard strings. Stage-specific state types are then generated or checked against the
same table, preventing a future field from being smuggled into an early call.

## 4. Exact runtime certification is a longest-path problem

Once an independently reviewed timing receipt supplies $p_{99}(q)$ for each physical call shape $q$, assign stage
weight

$$
w_s=t^{\mathrm{fixed}}_s+\sum_qN_{s,q}p_{99}(q),
$$

where $N_{s,q}$ is the exact active-call count for that stage and branch. For terminal node $v$, let $C(v)=w_v$; for
other nodes use the backward recurrence

$$
C(v)=w_v+\max_{u\in\operatorname{succ}(v)}C(u).
$$

Evaluating this recurrence in reverse topological order gives the exact worst compatible branch in $O(|V|+|E|)$
time, along with a witness path. Authorization requires

$$
t_{\mathrm{bootstrap}}+C(\text{start})+t_{\mathrm{publication}}\le28{,}800\ \mathrm{s}.
$$

This is stronger than summing independently maximal stages, which may combine mutually exclusive branches, and
stronger than the current crude rung-522 scaling. It also exposes which physical shape controls the margin. The
algorithm assumes that reviewed per-call p99 bounds compose additively and that fixed overheads are separately
bounded; asynchronous CUDA synchronization, allocator peaks, and an external watchdog remain empirical producer
requirements rather than consequences of graph theory.

## 5. Interaction-defined units have an exact Möbius decomposition

For a finite set of candidate mediators $M$, let $f(S)$ be the registered behavioral score after intervening on
exactly subset $S\subseteq M$. Define the pure interaction assigned to $A\subseteq M$ by

$$
\widehat f(A)=\sum_{B\subseteq A}(-1)^{|A|-|B|}f(B).
$$

[Rota's incidence-algebra Möbius inversion](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/rota1.pdf) gives the exact
and unique reconstruction

$$
f(S)=\sum_{A\subseteq S}\widehat f(A).
$$

For two sites, $\widehat f(\{i,j\})=f(\{i,j\})-f(\{i\})-f(\{j\})+f(\varnothing)$, exactly the hidden interaction term
that single-site activation patching mixes into marginal effects. This gives a basis over **interventions**, not over
heads or residual coordinates. It is gauge-invariant once $f$ is defined by downstream behavior. Its limitation is
exponential cost: full inversion over $n$ mediators needs $2^n$ subset evaluations (or $O(n2^n)$ arithmetic), so task
14 correctly uses a registered two-site family rather than pretending to decompose all model components at once.

Across the existing behavior battery, columns of pure-interaction response values can later be quotiented by equality
on held-out interventions. This is related to finite linear realization: [Carlyle and Paz](https://www.sciencedirect.com/science/article/pii/S0022000071800053)
show that, for an exact finite-rank string-response Hankel matrix, its rank is the minimum dimension of a linear
weighted state realization. The mapping would use intervention histories as prefixes and registered downstream tests
as suffixes. The theorem guarantees a minimal linear realization under complete exact Hankel access, up to change of
basis; we currently have only a finite, noisy, task-selected submatrix and a nonlinear RMS-normalized network.
Therefore empirical response rank is a screen/lower bound, not yet an exact Theseus solution.

## 6. Executable consequences and decision

Task 14 remains the highest-information route, but v3 must satisfy three new mathematical invariants before another
producer review:

1. Freeze a machine-readable DAG whose stages declare exact reads, writes, guards, and calls. Run the topological
   non-anticipation test; deliberately restoring v2's chunk order must fail.
2. Represent each transition by a one-use sealed capability. Duplicate replay, duplicate completion, branch forking,
   post-terminal continuation, and reuse of one global preflight for a second invocation must all fail.
3. After timing canaries exist, compute the weighted longest compatible path and its witness from the same frozen DAG.
   No task call may be authorized from a hand-added runtime estimate or a caller-supplied p99 value.

This route dominates another rank or reconstruction sweep because it protects the held-out interchange, necessity,
interaction, and reader claims. Kahn/session semantics solve ordering and single-use protocol structure, not circuit
identification. Möbius inversion solves finite subset interaction accounting exactly, not the choice of semantic
mediators. Hankel rank solves minimal **linear** realization under complete exact response access, which our finite
nonlinear experiment does not yet satisfy. Their usable consequences are respectively an anti-leakage compiler test,
an exact two-site interaction term, and a later gauge-invariant response-state screen.

The concrete continuation is active: compiler v2 is durably BLOCKED at review commit `60892e399`; the v3 builder has
accepted the non-anticipation, one-use capability, immutable captured-input, strict-type, prefix-abort, and stateful
deadline requirements. No task-14 model, checkpoint, GPU, outcome, result namespace, or managed queue has been opened.
