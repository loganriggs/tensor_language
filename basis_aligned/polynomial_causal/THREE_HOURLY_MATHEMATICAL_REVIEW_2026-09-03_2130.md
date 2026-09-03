# Three-hour mathematical review — 2026-09-03 21:30 UTC

## Circuit target and exact model object

The program goal is a smaller transparent tensor program that predicts fresh and out-of-distribution text, composes
when several replacements are installed together, supports selective removals/swaps/edits, and is simpler under literal
storage, compute, edges, states, and executable-program price. Rank or reconstruction alone is not a circuit result.

bilin18 has residual width $D=1152$, 18 blocks, 9 attention heads of width $d_h=128$, and MLP product width
$H=4608$. Ignoring the residual coefficients and writing $\bar x$ for the relevant RMS-normalized state, one attention
head computes

$$
q=W_q\bar x,\quad k=W_k\bar x,\quad q'=W_{q'}\bar x,\quad k'=W_{k'}\bar x,
$$

$$
p_h(q,k)=\frac{\langle q_q,k_k\rangle}{128}
           \frac{\langle q'_q,k'_k\rangle}{128},
\qquad
o_h(q)=W_h^O\sum_{k\le q}p_h(q,k)v_h(k).
$$

There is no softmax. With normalization held fixed, the score is degree four in the input states and the value
contraction is degree five. RMS normalization makes the full map homogeneous/rational rather than an ordinary global
polynomial. An MLP computes

$$
M(x)=D_M\big[(L_M\bar x)\odot(R_M\bar x)\big]+b_M,
$$

which is quadratic with normalization held fixed. Weights are tied across token positions inside a layer, but not
across layers.

The current R585 object is smaller and exact. For endpoint $x$, site $h$, and semantic source role $r\in\{A,C\}$,

$$
e_h^x(r)=p_h^x(q_x,k_x(r))\mathbf 1[t^x_{k_x(r)-1}=t^x_{q_x}],
\qquad
u_h^x(r)=W_h^Ov_h^x(k_x(r)),
$$

and the isolated equality contribution is

$$
T_h^x=\sum_r e_h^x(r)u_h^x(r)\in\mathbb R^{1152}.
$$

R585 crosses recipient and donor factors as $e_xu_x,e_yu_x,e_xu_y,e_yu_y$ at four fixed sites. Exactness is measured
elementwise at $10^{-5}$; scientific outputs are held-out logit margins, cross-entropy, and full-vocabulary logit
changes. The complete held path costs 459 FIT plus 231 SELECT model forwards, zero backwards, and zero weight updates.
FINAL and OOD remain closed.

Relevant gauges remain:

- a value/output change of basis $v\mapsto Gv$, $W^O\mapsto W^OG^{-1}$ inside a head;
- paired rotations of normalized Q/K coordinates that preserve their dot product, separately for the two QK branches;
- branch exchange between the two multiplied QK scores;
- permutations and reciprocal scalings of bilinear MLP product units; and
- additional mixing whenever two sites or roles produce indistinguishable downstream functions.

Therefore native heads and hidden dimensions are coordinate containers, not assumed semantic atoms.

## Exact interaction decomposition before tensor factorization

For a set $N$ of mediators, let $v(S)$ be the output when mediators in $S\subseteq N$ take donor values and all others
take recipient values. The Boolean-lattice Möbius coefficient is

$$
I(S)=\sum_{A\subseteq S}(-1)^{|S|-|A|}v(A),
$$

and it reconstructs every intervention exactly:

$$
v(S)=\sum_{A\subseteq S}I(A).
$$

For score and payload, the pair interaction is simply

$$
I(\{e,u\})=v(e_y,u_y)-v(e_y,u_x)-v(e_x,u_y)+v(e_x,u_x).
$$

This is the correct finite computation for the user's multiple-mediator concern. Vaidyanathan et al. show that ordinary
single-component activation-patching effects contain mediator interactions and that those interactions decompose into
pairwise and higher-order group terms ([primary preprint](https://arxiv.org/abs/2606.27510)). Möbius inversion does not
decide which interaction is semantically meaningful, but it prevents the interaction from being silently assigned to
one native component. Shapley–Taylor gives a different axiomatic redistribution when only interactions up to a chosen
order are reported ([primary paper](https://proceedings.mlr.press/v119/sundararajan20a.html)); for our exact, small
factorial grids the unredistributed Möbius terms are the clearer primitive.

The cost is $2^{|N|}$ interventions for a complete lattice. We can afford exact score/payload or small site-group
lattices; a large head census needs a preregistered low-order restriction rather than pretending single-head effects are
interaction-free.

## A tensor decomposition that matches the factor object

Choose donor endpoints $i$, recipient endpoints $j$, and fixed linear output functionals $w_k\in\mathbb R^{1152}$.
For site/role index $a=(h,r)$, define

$$
A_{ia}=e_h^i(r),
\qquad
B^{(a)}_{jk}=\langle w_k,u_h^j(r)\rangle.
$$

The complete crossed immediate-output tensor is

$$
\mathcal T_{ijk}=\sum_{a=(h,r)}A_{ia}B^{(a)}_{jk}.
$$

Each summand has multilinear rank $(1,L_a,L_a)$ where $L_a=\operatorname{rank}B^{(a)}$. This is a block-term
decomposition (BTD), not ordinary low-rank compression. Flattening to a matrix gives $T_{i,(jk)}=AB^\top$ and permits
the full gauge

$$
A\mapsto AG,\qquad B\mapsto BG^{-\top}
$$

for any invertible $G$. Keeping the recipient and output modes separate can remove much of that ambiguity.

Domanov and De Lathauwer give uniqueness conditions and an algebraic eigenvalue-based recovery method for sums of
multilinear-rank $(1,L_r,L_r)$ terms ([primary paper](https://arxiv.org/abs/1808.02423)). Two useful generic sufficient
conditions from that work are:

1. for equal block size $L$, with tensor shape $I\times J\times K$ and $R$ terms,

$$
R\le \min\{(J-L)(K-L),I\};
$$

2. for variable block sizes,

$$
\sum_{r=1}^R L_r\le\min\{(I-1)(J-1),K\},
\qquad
J\ge\max_{r\ne s}(L_r+L_s).
$$

Under the paper's assumptions, the block terms are unique up to permutation and within-block bases, and can be found
with linear algebra rather than a nonconvex neural fit. In the stronger rank-one special case, Kruskal's classical CP
condition $k_A+k_B+k_C\ge2R+2$ gives uniqueness up to permutation and scaling
([primary paper](https://doi.org/10.1016/0024-3795(77)90069-6)). R585's payload-by-output blocks need not be rank one,
so BTD is the better match.

### What maps exactly

- donor index $i$ maps to the equality-score function;
- recipient index $j$ maps to the projected-value function;
- output index $k$ maps to fixed residual directions or FIT-frozen circuit readouts;
- the eight site/role pairs from four sites and two roles map to candidate block terms; and
- the tensor can be generated from cached endpoint factors without additional model forwards.

### What currently violates the theorem

1. R585 evaluates a sparse set of semantically paired donor/recipient directions, not the complete Cartesian tensor.
   A small complete tensor must be reconstructed from endpoint factors after capture.
2. The equality patterns are highly structured and repeated, so generic uniqueness is not a deterministic certificate.
   The actual rank and deterministic conditions must be checked.
3. The decomposition is exact for the immediate residual contribution. Passing it through later bilinear layers makes
   the downstream logit response nonlinear in the inserted term; that response is not automatically the same BTD.
4. Algebraic uniqueness does not imply that a term predicts behavior, is selectively removable, or is reused. Those
   require the existing interchange/removal/OOD gates.
5. A unique algebraic block may still combine several meanings, while several blocks may implement one redundant
   meaning. Downstream operational equivalence remains the semantic quotient.

Thus the theorem does not solve the full Theseus compilation task. It does give a principled way to test whether
head/site boundaries can be replaced by identifiable cross-boundary blocks before fitting an SAE or choosing a rank.

## Executable consequence completed now

`ops/circuit_block_term_identifiability_toy.py` constructs an exact two-block $(1,2,2)$ tensor, applies a nontrivial
$GL(2)$ matrix gauge, checks both published generic BTD bounds, and performs exact two-mediator Möbius inversion. Its
seven planted tests pass. The observed matrix-gauge reconstruction error is $1.78\times10^{-15}$ and the Möbius
reconstruction error is exactly zero. The toy explicitly records three limitations: generic is not deterministic,
paired observations are not a complete tensor, and immediate-output identification is not downstream causal use.

The next model-linked consequence, conditional on R585 producing valid factor evidence, is CPU-only:

1. choose a group-disjoint complete subset of donor and recipient endpoints spanning all selector/payload states;
2. freeze common output directions on FIT, including task-defined downstream readouts rather than variance-only axes;
3. build $\mathcal T$ implicitly from saved $e$ and $u$ factors, estimate every block rank, and check the deterministic
   and generic BTD conditions;
4. recover candidate blocks on FIT and test tensor prediction on SELECT endpoints without refitting;
5. compare recovered blocks across resamples after quotienting permutation/within-block gauge; and
6. test the surviving blocks with the same score/payload interchange, active removal, composition, and OOD criteria.

The route dies if the complete tensor is not well described by a small stable block family, the uniqueness conditions
fail badly, or algebraically stable blocks do not predict selective downstream effects. It survives only if it groups
or splits native heads/sites in a way that improves held-out circuit prediction and manipulation.

## Decision relative to the empirical route

R585 remains the highest-information immediate action. It directly tests opposing selector and payload predictions on
held-out groups, while the tensor theorem currently applies only to an immediate-output object reconstructed from those
same factors. The mathematical route should therefore run as a CPU identification analysis after valid R585 evidence,
not delay or replace the causal experiment. If R585 is a scientific null, its failure class determines whether the BTD
object is still meaningful; we must not salvage it by searching ranks or loosening thresholds.

The new mathematical contribution is a concrete alternative basis: identifiable donor-score × recipient-output blocks
across heads and roles, with explicit theorem checks and gauge. It is materially different from the current native-site
factor test and from rank reduction, yet remains subordinate to downstream causal validation and the full adoption
goal.
