# Hierarchical Tucker and shared bilinear DAGs

User direction, 20 September 2026. This refines the earlier instruction to
symmetrize higher-order folded tensors. **Symmetry identifies the polynomial;
the stored computation need not be a symmetric coefficient tensor.**

## The representation

Hierarchical Tucker (HT) represents a large tensor as a tree of smaller
bilinear computations. For the order-five tensor obtained by folding two
pure bilinear layers,

\[
f_v(x)=\sum_{ijkl}H_{vijkl}x_ix_jx_kx_l,
\]

one tree is

\[
u^{(s)}=P_s^\top x\quad(s=1,2,3,4),\qquad
q_a=\sum_{pq}A_{apq}u^{(1)}_pu^{(2)}_q,\qquad
r_b=\sum_{pq}B_{bpq}u^{(3)}_pu^{(4)}_q,\qquad
f_v=\sum_{ab}C_{vab}q_ar_b.
\]

The leaves discover linear features, the next level quadratic features, then
quartic outputs. Deeper trees repeat this construction. Ordinary Tucker uses
one core; HT recursively factors it into smaller cores. The tree groups
**tensor slots**, not necessarily disjoint coordinate subsets. Every leaf can
read every coordinate of the same full x.

For example,

\[
(x_1^2+x_2^2)(x_3^2+x_4^2)=q(x)r(x)
\]

needs two products in each quadratic and one root product. Expanded monomial
counts grow with the sizes of the sums, while this factorization stays compact.

HT compression depends on internal ranks staying small. It provides a useful
baseline and initializer, not a guarantee of simple or causal features.
[Grasedyck, Hierarchical Singular Value Decomposition of Tensors](https://epubs.siam.org/doi/10.1137/090764189).

## What ordinary HT does not optimize for us

1. **Tree choice.** For independent slots, H_ijkl=A_ij B_kl has rank one across
   (ij)|(kl), but rank(A) rank(B) across (ik)|(jl). Compare alternative trees.
2. **Sharing between branches.** A feature already feeds many channels of its
   parent in HT. However, two branches computing q separately in q(x)^2 need
   not be recognized as identical. Tying them forms a DAG and computes q once.
   A tree does not forbid all sharing; it restricts where sharing is built in.
3. **Simple individual features.** A small intermediate space can contain
   dense quadratic forms. Invertible basis changes can be absorbed by a parent
   core; low rank alone does not choose an interpretable basis.
4. **Repeated-input polynomial identities.** We evaluate on (x,x,x,x), whereas
   ordinary tensor approximation compares independent input slots. Two
   different multilinear tensors can represent exactly the same polynomial.

## Symmetry can inflate representation rank

For f(x)=(x^T x)^2, the coefficient representative

\[
H_{ijkl}=\delta_{ij}\delta_{kl}
\]

has rank one across (ij)|(kl). Its fully symmetric representative is

\[
H^{sym}_{ijkl}=\tfrac13(\delta_{ij}\delta_{kl}
+\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk}).
\]

It computes the same polynomial but has unfolding rank d(d+1)/2. On symmetric
matrices that unfolding acts as X -> (tr(X)I+2X)/3 and is invertible; it
annihilates antisymmetric matrices. Thus a rank bound for a symmetric
representative is not automatically a lower bound on the cheapest polynomial
program. Keep an unsymmetrized compact representation and compare functions.

## Discovery objective and accounting

Use HT as an initialization/compression baseline, then optimize a **sparse,
signed, shared bilinear DAG**, with adaptive intermediate widths:

\[
\min_\theta \|\operatorname{Sym}_{inputs}(H-\widehat H_\theta)\|_F^2
+\lambda\sum_t\|G_t\|_0+\mu N_{features}.
\]

The output index is never symmetrized. The candidate tensor can remain implicit
and unsymmetrized. Feature scales need constraints for continuous sparsity
penalties. For L0 support, invertible basis freedom still affects simplicity;
report the chosen basis and executable price.

| Constraint | Intended effect |
| --- | --- |
| Few intermediate channels | Small feature dictionaries |
| Few nonzero core coefficients | Few interactions |
| Shared features across branches | Compute repeated functions once |
| Sparse leaf projections where useful | Few original source coordinates per feature |

Narrow dense and wide sparse cores are different candidates. Count all leaf
projections, intermediate coefficients, supports, adapters, root coefficients,
and distinct multiplication nodes. A local price must identify omitted source
generation and normalization costs. [VeST](https://arxiv.org/abs/1904.02603)
is precedent for sparse Tucker factors and cores; it uses iterative entry
pruning and updates for partially observed tensors. Our hierarchical,
signed, repeated-input shared-feature objective is an extension, not a result
already supplied by that paper.

## Implications for the current work

The v615 lower bounds remain valid for the stated **order-three symmetric
shared-input Tucker family** W G(P,P), measured in its declared z coordinates.
They do not bound all arithmetic DAGs, arbitrary HT representatives of quartic
polynomials, or behavioral error on normalized model inputs. In particular,
they must not be used to abandon wide sparse programs.

For two-layer and deeper folds, match the symmetrized difference while allowing
cheap unsymmetrized representatives. Test basis changes, prune interactions,
merge repeated computations, and compare groupings. HT alone is not the final
success criterion. Freeze discovered candidates, then test OOD predictions,
independent extraction, selective removal/restoration, and reuse with controls.
Keep RMSNorm, Q/K normalization, attention routing, final normalization and
softcap explicit; a polynomial numerator does not replace those operations.

## Executable reference and existing tools

`quartic_bilinear_quotient.py` evaluates a bank of quadratic features once and
reuses them across root edges. Its exact polynomial inner product avoids the
four-input tensor. For symmetric forms Q,R,S,T,

\[
\langle Sym(Q\otimes R),Sym(S\otimes T)\rangle
=\frac{\langle Q,S\rangle\langle R,T\rangle
+\langle Q,T\rangle\langle R,S\rangle+4tr(QSRT)}6.
\]

It supports signed sums, gradients, and local feature removal. The tests check
this identity and gradients against dense symmetrization, rank inflation,
tree grouping, sharing, and the cancellation
(x0²+x1²)²-(x0²-x1²)²=(2x0x1)². This is a reduced-space reference: quadratic
banks cost O(features*d²), and the reference program inner product takes
O(edges²*d³) work. Native-width scalability and automatic discovery are not
claimed.

Existing `quartic_pair_dag_v1.py` minimizes shared pair nodes **for a fixed set
of quartic monomials and 2+2 schedules**. Its optimality does not include
factoring sums of monomials into reusable quadratic features. Retain it as a
restricted baseline; do not relabel its optimum as a globally simplest DAG.

See [the joint sparsity audit](JOINT_FOLDED_SPARSITY_2026-09-20.md) for the
native-weight receipt and distinction between weight and behavioral fidelity.

The executable control `check_quartic_quotient_controls_v1.py` compares the
existing fixed-monomial compiler with the shared feature q=sum_i x_i².
Its [receipt](QUARTIC_QUOTIENT_CONTROLS_V1_RESULT.json) records, for d=2/4/8,
3/5/9 compact scalar products versus 5/14/44 in the proven-optimal restricted
monomial schedule. Both execute the same polynomial within 3.1e-16 relative
error. Symmetric unfolding ranks are 3/10/36 versus 1 for the compact
representatives. This is a planted regression control, not a discovered native
circuit or a proof that the compact program is globally optimal.
