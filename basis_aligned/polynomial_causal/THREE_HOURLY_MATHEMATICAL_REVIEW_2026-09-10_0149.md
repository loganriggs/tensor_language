# Mathematical review — query-source interaction tensors

Due 01:49 UTC; performed at the next research boundary on 10 September 2026.
Follow the original bilinear handoff and reconstruction pilot. The objective
is a previously unspecified reusable computation with OOD prediction,
extraction, selective removal/interchange and composition, with all opaque
weights and adapters charged. The command-mode is/was loop remains parked.

## Decision and exact object

The pilot already established faithful execution and rejected trivial local
sharing as circuit discovery. Two newer trained-model screens reject a fixed
shared query and a raw direct-embedding query at L9H1/H4. The next useful
object is therefore the contextual query producer, explicitly separated
from its attention consumers. Another global rank scan would not identify it.

Bilin18 has18 layers, residual dimension D=1152, nine heads of width h=128,
MLP width4608 and545902902 parameters. At query position t before attention9,
write u(z)=sum_a z_a f_a, a=0,...,18. Here f0 is the normalized embedding
times its full direct skip/reentry coefficient; the other f_a are native
attention and MLP writes at layers0..8 times subsequent residual coefficients.
All source values and all keys/values are fixed to their native execution.
The gains z are real query-input EDGE edits, not global producer removals.
For global removal the later sources and keys can change, invalidating this
fixed bank; they would have to be regenerated.

For query kind i in{1,2} and either head, let P_i[a,:]=W_Qi f_a,
G_ab=f_a^T f_b, and

    H_i=P_i P_i^T/h + epsilon_head G/D,
    d_i(z)=z^T H_i z + epsilon_head epsilon_residual.

Combining the residual RMS and head RMS gives exactly, in real arithmetic,

    q_i(z)=R_t P_i^T z / sqrt(d_i(z)).

This follows by multiplying the head-norm denominator by the residual-norm
denominator; no normalization is frozen. Native FP32 uses both epsilons
equal to1.1920928955078125e-7. R_t is the ACTUAL rounded rotary matrix; it
need not be orthogonal. Positive epsilons keep denominators nonzero even
at z=0. Floating-point Gram cancellation is a separate instrument concern;
the implementation rejects negative computed squared norms without clipping.

Let k_i,s be native normalized, rotated keys and v_s the mixed native value
at causal source position s<=t. Define a_i,s[a]=(R_t P_i[a,:])^T k_i,s.
The vector read is

    y(z) = sum_s (z^T a_1,s)(z^T a_2,s) v_s / h^2
           / sqrt(d_1(z)d_2(z)).

Thus each output coordinate has a quadratic numerator and the square root
of a product of two quadratic denominators. This is an algebraic normalized
function, not a polynomial whole transformer. The numerator tensor is

    N_v = sum_s v_s[v] sym(a_1,s a_2,s^T)/h^2.

There are190 degree-two monomials in19 variables, within the pilot's256
bank cap. Off-diagonal monomial coefficients are2*N_v[a,b]. A downstream
linear output reader or the attention output projection folds into v_s.
The subsequent nonlinear residual/MLP suffix must still be recomputed.

## Theorem/algorithm mapping and limits

**Associative attention contraction.** The feature-product construction of
[Katharopoulos et al.](https://proceedings.mlr.press/v119/katharopoulos20a.html)
maps here to query feature q1 tensor q2 and key feature k1 tensor k2. Our
model has no softmax or positive-kernel normalizer; its two RMS factors are
handled explicitly above. Once features are fixed, associativity is exact.
It supplies an evaluation identity, with no uniqueness or semantic recovery
guarantee. Our source-space contraction uses two S-by-T score-factor matrices
per head and two S-by-S norm matrices. Evaluation costs O(ST+Th+S^2) per
head per gain vector; initialization costs include O(SDh+S^2D+STh).
For short T this avoids a190-by-h coefficient tensor. It does not make the
native prefix or key/value generators cheaper or explained.

**Bilinear weight tensors and identifiability.**
[Pearce et al.](https://arxiv.org/abs/2410.08417) express bilinear MLP outputs
as weight-defined quadratic forms. Our restricted attention numerator is
another such form in a DIFFERENT input domain: source gains, not arbitrary
residual vectors. Its symmetric coefficients are uniquely determined as a
polynomial on an open real gain domain. Factorizations of those coefficients
need not be unique or match semantic units. A supplied CP factorization
that satisfies the three-mode Kruskal inequality k1+k2+k3>=2r+2 has the
usual scaling/permutation uniqueness guarantee; see the primary
[robust Kruskal analysis](https://arxiv.org/abs/1304.8087). No such condition
has been established here. In particular, the obvious2T-term symmetrized
factorization repeats each value vector twice, so its third-mode Kruskal
rank is at most1 and that sufficient inequality cannot hold. This failure
does not prove that every alternative minimal factorization is nonunique.
Therefore use physically specified source edges and joint edits to identify
sharing; do not label the output of a CP fit a circuit. Dense coefficient
construction costs O(S^2Th); no general efficient semantic-recovery
algorithm or native tensor-rank result is claimed.

**Constrained lumping.** [CLUE](https://arxiv.org/abs/2004.11961) finds a
minimal linear state reduction for polynomial ODE dynamics preserving
specified linear observables. Its coefficient-matrix invariant-subspace
closure suggests preserving both numerator readers AND normalization
observables. A naive dense closure with M known n-by-n generators takes
at most n enlargements and O(M n^4) field arithmetic as a loose bound;
that is our implementation bound, not a quoted paper complexity. This
transformer is a discrete normalized recurrence, and our source bank is
conditional on a context. It does not satisfy the polynomial ODE premise.
Consequently we do not import CLUE's minimality or biological interpretability
claim. The executable consequence is to include H1/H2 in every source edit.

**Hankel/automata and contraction width.**
[Rabusseau, Li and Precup](https://proceedings.mlr.press/v89/rabusseau19a.html)
map linear second-order RNNs to weighted automata and recover parameters
from suitable full-rank Hankel blocks. Our normalized recurrent state has
not been shown to be a finite linear automaton; sampled low-rank blocks
would not establish that premise. Finite block SVD costs O(n^3) for an
n-by-n block, with construction and alphabet-dependent blocks additional.
Neither minimal state nor parameter recovery applies to our model as-is.
[Markov and Shi](https://arxiv.org/abs/quant-ph/0511069) instead control
contraction cost through graph width: bounded local dimensions and small
width permit efficient contraction, with exponential width dependence.
This explains why a small LOCAL source-gain bank is practical; it does not
certify cheap global polynomial expansion, a minimum circuit, or identifiable
semantic factors. TT/HT decompositions similarly need a specified tensor
and cut-rank/approximation assumptions; none identify our missing producers.

## Executable consequence and source-sharing criterion

Implemented source_gain_attention.py. Synthetic FP64 singleton, joint,
signed and zero source edits agree with independent nested-RMS/query-dot
evaluation to1.2213e-15. Dropping norm cross terms changes the read by.8092
in the planted fixture. Rounded nonorthogonal rotation and causal endpoints
are exercised. A ten-layer tiny model preserves the capture forward exactly;
compiled native reads agree to6.94e-17 and full output to1.11e-16. Hooks
restore. SOURCE_GAIN_ATTENTION_V1_CONTROLS.json records three tiny CPU
forwards, zero trained-checkpoint forwards. This is tool evidence, not a
trained-model positive result.

For a fixed context, a sufficient local interchange criterion for two sources
is invariance of every N_v and both H_i under swapping those source labels.
It tests their consumers, rather than raw source cosine. It is sufficient,
not necessary: numerator/denominator rescalings and exchanging query kinds
can represent the same function. Across tasks, this condition must hold for
the appropriate contexts and interventions before merging producers.

The managed L9_QUERY_SOURCE_ATLAS_V1 now measures all19 single-source and
leave-one-out queries, full/zero controls, and an independent raw query-port
omission replay. The registered common-singleton hypothesis opposes the
distributed-query hypothesis. The overlap of robust omission dependencies
asks whether the two tasks use the same upstream inputs. Full output and
paired causal errors are measured by the existing scorer. Any nominated
source remains exploratory until new texts and joint task-specific edits.

All545902902 model parameters, native initialization, keys/values, adapters
and suffix remain charged. There is no structural reduction yet. The local
conditional representation improves computational specification and source
manipulation; it does not establish token-to-logit extraction or OOD reuse.
This circuit-local test has greater information value than another fixed
query, rank allocation, or global obstruction scan. Next math review is due
04:49 UTC; ordinary hourly review remains separately due02:06 UTC.

Native receipt after derivation: managed02:00:05–02:00:21;336forwards,
6048sequence evaluations,13.179s. All instrument checks pass; no singleton
passes even one whole panel. Both robust omission sets are empty under the
fixed BOTH-panel .10 criterion. Preserve B/C nulls. The follow-on CPU exact
source/rest factorial audit has identity error0; summing19 individual
omission margin effects misses the joint query effect by73–81%. This
supports investigating joint computation, but does not localize its
nonlinearity between routing numerator, normalization and the suffix.
