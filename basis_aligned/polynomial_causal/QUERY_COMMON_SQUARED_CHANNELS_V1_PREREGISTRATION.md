# Can a coordinate change separate the coupled query computation?

The root-additivity bound rejects the native L/H/D bipartitions. Test whether
this coupling can instead be removed by an invertible linear change of those
three input coordinates, while preserving their RMS denominator and all route
readers. This is a circuit-splitting question under explicit interventions, not
a singular-value cutoff, variance target or rank sweep.

For a fixed native context, query q=aL+bH+cD has squared RMS z^T G z,
G_ij=mean(root_i*root_j), z=(a,b,c). At native query gain, every centered
route/output coordinate is z^T A_k z for a symmetric3x3 matrix A_k. The
candidate uses one coordinate change z=W u so the norm and every A_k become
diagonal: three shared squared channels u_1²,u_2²,u_3² with separate readers.
Keeping a coupled norm while claiming fully independent channels is excluded
from this specific grammar. More general bilinear features remain outside it.

Theorem3.2/3.3 of [Jiang and Li](https://arxiv.org/pdf/1507.05703) gives the
positive-definite whitening test: real symmetric forms including G>0 are
simultaneously diagonalizable by congruence exactly when their G-whitened
forms commute. Equivalently A_i G^-1 A_j=A_j G^-1 A_i. A necessary and
sufficient exact test can therefore avoid numerical eigenspace selection by
checking A_i adj(G) A_j-A_j adj(G) A_i=0 for every pair. This implication
follows by whitening G to the identity and using orthogonal diagonalization
of commuting symmetric matrices.

Fixed first context: overlapping_join_normalizer_reference.populations()['iid']
world-list entry0, token row3 (forward query/hop3, first eligible order).
No context/head/output selection based on results. Compile all58 centered
output forms:29 for I, then29 for J, original logit order. Evaluate roots and
pairwise sums to recover the six quadratic coefficients; verify held-out
amplitude vectors (1,1,1),(.3,-.7,1.2),(0,0,0),(-1,2,.5) against explicit
Q-port reads and native projection-hook contrasts. Keep source features,
native query gain and residual as in previous exact route work.

A instrument: numerical compiler/native agreement abs<=1e-9/relative<=1e-10
using live sums; exact Sylvester minors certify positive definiteness of the
stored Gram. No jitter, rank truncation or replacement context if G fails.
B: scan all matrix pairs in fixed lexicographic order until an exact nonzero
commutator witness is found, or all pairs are verified. Any witness rejects
the three squared-channel grammar even locally, hence also any common basis
claimed across contexts. Commutativity of only a selected pair is not a pass.

Use Python Fraction arithmetic on the stored FP64 matrices for exact dyadic
minors and witness entries. This is an exact statement about the compiled
coefficient artifact; compiler/native agreement is numerical, not an interval
certificate for the ideal real-valued neural network. Report a whitened
normalized commutator as a numerical scale diagnostic, not an approximation
lower bound. A null does not exclude nonlinear coordinate maps, non-diagonal
normalization, products of different forms, or larger shared programs.

Planted positive: common diagonal forms under a fixed invertible nonorthogonal
coordinate map plus its positive Gram. Negative: identity Gram, a diagonal
form with distinct entries and a noncommuting symmetric form. Singular/indefinite
Grams must be rejected as assumption failures. CPU threads2,alarm180s,
tensors below256MiB,no GPU/fit/training. All387968 native coefficients and
query-root production remain priced. No structural saving from a certificate.
