# Three-hour mathematical tensor-network review — 2026-09-08 02:26 UTC

## Current tensor program

Bilin18 has residual width (d=1152), (L=18) decoder blocks, (H=9) attention heads of width
(p=128), bilinear-MLP width (m=4608), and vocabulary size (V=50304).  With example (i),
token (t), residual coordinate (a), construction expert (e\in\{A1,A2\}), and cross-fit parity
(f\in\{0,1\}), the newly identified entry-12 object is

\[
X^{0}_{fita}\in\mathbb R^{T_i\times1152},\qquad
D^{e}_{fita}=X^{e}_{fita}-X^{0}_{fita},\qquad
U_f\in\mathbb R^{1152\times2},\quad U_f^\top U_f=I_2.
\]

(X^0) is the upstream-off residual state, (X^e) is produced by the fixed construction-specific
four-head oracle, and (U_f) is fit only on the opposite parity from the two construction mean
writes.  Let (M_{it}\) be the semantic-prefix mask and let the router output the one-hot vector
(z_i=(z_{i,A1},z_{i,A2},z_{i,off})\).  The executable state replacement is

\[
\widetilde X_{fita}=X^0_{fita}+M_{it}\sum_{e\in\{A1,A2\}}
z_{i,e}\,[D^e_{fit:}U_f]_{k}U_{f,ak}.
\]

This is a contraction of a row-dependent expert-write tensor, a rank-two state projector, and a
three-way selector.  The successful v15 router computes (z) from the aligned base/source token
pair string: unequal pairs are made unordered, two registered signatures map to A1/A2, and all
other strings map to off.  It is therefore a deterministic finite transducer over paired alphabet
(\Sigma\times\Sigma), followed by a degree-one state projection.  The router is discrete rather
than polynomial.  The rejected Gram router was degree two in proposed writes.  The transformer
suffix remains nonpolynomial because RMS normalization and attention softmax/squared-score
normalization are input dependent; each MLP core itself is degree two,
(D_\ell[(L_\ell x)\odot(R_\ell x)]\).

The tied token embedding/output matrix remains part of the original model.  Gauge freedoms are:
(U_f\sim U_fQ) for (Q\in O(2)); sign gauges on each upstream rank-one head axis; simultaneous
permutation of expert labels and router outputs; and reversal of each unordered token pair.  A
general residual (GL(d)) gauge is not free with fixed RMS normalization; orthogonal changes paired
with every adjacent tensor preserve the RMS norm, while arbitrary changes do not.

Allowed causal inputs presently require row-aligned base/source token lengths and a semantic-prefix
endpoint.  V15 A1/A2/P/C satisfy this.  The v16 A1/A2/P bank is capability-qualified, but its cue
pairs are unseen.  Outputs are measured separately by rowwise answer-minus-foil signed projection,
direction fraction, relative error, full-vocabulary KL, and top-one flips.  The v15 program reaches
`.795-.881` target projection with zero P/C effect and exact learned/gold composition.

For one deployed cross-fit fold, literal coordinate storage is 1,024 scalars for two experts times
four 128-dimensional head axes, 2,304 scalars for (U_f), and four cue token IDs plus two labels for
the router, before sites/order metadata: approximately 3,334 scalar/integer entries.  Storing both
evaluation folds doubles this.  Per patched token, the rank-two state projection costs about
(4d=4608) multiply-adds, in addition to four upstream rank-one head projections and the unchanged
full layer-12--17 suffix.  This is an identified subprogram price, not an adopted smaller model:
the suffix has not been extracted or removed.

## Exact neighboring theory and assumption audit

### Finite automata, Myhill--Nerode, and Hankel realization

The token router defines a total function on paired-token strings by giving every unregistered
string the off output.  Myhill--Nerode equivalence groups prefixes that have identical outputs for
all continuations; the number of equivalence classes is the exact number of states in a minimal
deterministic realization.  Nerode's original automata paper is the primary historical source
([Proceedings of the AMS, 1958](https://doi.org/10.1090/S0002-9939-1958-0135681-9)).  On the current
restricted one-difference language, the three distinct terminal outputs already require at least
three distinguishable terminal classes; a prefix trie plus an off sink gives an immediate finite
upper bound.  Exact minimization is possible once the full accepted paired-token language is fixed.

For real-valued output indicators, the same router can be represented as a weighted finite
automaton.  The full paired-string function has Hankel matrix
(H_f(u,v)=f(uv)); finite Hankel rank equals the minimum exact linear-automaton state dimension.
Spectral algorithms recover such realizations from suitable finite-rank Hankel blocks, and modern
work makes the missing-entry assumptions explicit
([Balle and Mohri, NeurIPS 2012](https://papers.neurips.cc/paper_files/paper/2012/file/700fdb2ba62d4554dc268c65add4b16e-Paper.pdf)).
Linear 2-RNN/tensor-network equivalences similarly map finite-rank Hankel tensors to multilinear
recurrent realizations
([Li, Precup, and Rabusseau, 2022](https://doi.org/10.1007/s10994-022-06164-1)).

These theorems do **not** supply semantic OOD generalization.  Our finite v15 observations do not
determine the unobserved rows/columns of the infinite Hankel matrix, and the explicit default-off
extension is only one of infinitely many functions agreeing on v15.  Exact automaton minimization
can compress that chosen lookup; it cannot infer that `Right now` and `Back then` instantiate the
same temporal relation as `Today` and `Yesterday`.

### Orthogonal invariant theory and the failed Gram router

For vectors under a common orthogonal action, the first fundamental theorem says invariant
multilinear forms are generated by pairwise bilinear contractions.  A modern primary proof states
the orthogonal-group generators explicitly as products of pairings
([Lehrer and Zhang](https://publications.ias.edu/sites/default/files/2015-06-ortho.pdf)).  For two
proposed writes (a,b\in\mathbb R^d), their Gram entries
(a^\top a,b^\top b,a^\top b) therefore generate polynomial invariants under arbitrary common
orthogonal gauges, including simultaneous sign reversal.

This maps exactly to the six-feature router only after two lossy reductions: it observes the Gram
matrix at the semantic token and an average Gram matrix over the prefix.  It discards the full
token-indexed Gram tensor and uses nearest centroids rather than every polynomial of the Gram
entries.  Therefore its three leaked off rows reject that six-scalar/centroid object, not all
gauge-invariant nonlinear routers.  The theorem explains why direction reversal was repaired; it
does not rescue control separation lost by prefix averaging.

### Exact weight translation must include normalization

For a state coordinate (U_f), a raw static reader score such as (W_Q U_f) omits the fact that
every downstream reader sees (n(x)=x/s(x)), where
(s(x)=\sqrt{d^{-1}x^\top x+\epsilon}).  Its exact Jacobian is

\[
J_n(x)=s(x)^{-1}I_d-\frac{xx^\top}{d\,s(x)^3}.
\]

Thus the local query response is (W_QJ_n(x)U_f), not (W_QU_f); key, value, and MLP left/right
responses have the same correction.  The exact finite response is
(W[n(x+U\alpha)-n(x)]).  These maps are weight-tensor translations of an already causal state
coordinate, but remain reader nominations until a reset/rescue at the corresponding live input
shows mediation.  This qualification is required by the observed L15H5 counterexample, where
strong static alignment carried only a small causal side channel.

## Executable consequences and decision

The exact lookup router has an immediate zero-forward OOD certificate.  Freeze its two v15
signatures and tokenize every v16 aligned pair.  If a v16 target signature is absent, the router
definition forces (z_{A1}=z_{A2}=0), hence
(\widetilde X=X^0) and the causal target response is exactly zero without running the suffix.
No optimizer or GPU causal replay can change that fact.  Opposing outcomes are:

- **Covered OOD signature:** a v16 target unexpectedly equals one registered unordered token
  signature; run the causal OOD arm.
- **Uncovered OOD signature:** preserve the v15 circuit as a task-specific screen, record exact
  zero predicted transfer, and require a semantic/token-level relation router before promotion.

In parallel, a zero-forward weight atlas is now mathematically well specified: recompute (U_f)
from frozen v15 training parity; for every downstream Q/K/Q2/K2/V and MLP Left/Right tensor, report
raw (WU_f), state-conditioned (WJ_n(x)U_f), and exact finite normalized responses; include the
final tied-unembedding answer-minus-foil covector.  Rank only interfaces with nontrivial response,
and causally test complete reader inputs before naming an edge.

The OOD signature audit dominates first because it is exact, nearly free, and decides whether the
current finite gate can possibly transfer.  The normalization-aware weight atlas can proceed next
as diagnostic infrastructure regardless of that null, but no static ranking is promoted without
reader mediation.  The live plan therefore becomes: perform the hash-bound v16 signature coverage
audit now; preserve the predicted default-off null if it occurs; then build the normalization-aware
weight translation while a semantic unseen-cue router is specified.  This advances held-out
prediction and computational specification without drifting into compression.
