# Mathematical review: a composable local response state is not a closed circuit

Due16:49 UTC; started16:49:42. Original bilinear handoff/pilot and the user's
two unembedding views govern, not the stale better_math goal wording. Previous
goal turn PROGRESSa27671861. Full goal: OOD prediction, extraction, selective
removal, composition/reuse, and a simpler executable explanation.

The recent native evidence separates these requirements. A weight-derived
grammatical scalar transfers across verbs/cues but fails agreement selectivity.
Blocking all18 post-RMS MLP scalar reads loses .291/.291 target recovery and
removes69.6% of base-endpoint agreement damage. A scalar-only edit predictor is
then closer, but its .153/.084 errors still fail the joint .10 bound. The next
question is what contextual state makes the identified consumer executable.

## Actual mathematical object

For one native bilinear MLP, d=1152 and m=4608:

    f(u)_i = sum_j D_ij (sum_a L_ja u_a)(sum_b R_jb u_b) + bias_i.

The contraction tensor is T_iab=sum_j D_ij L_ja R_jb. Its two input vectors are
tied, so only sym_ab(T) is observable. Input u is the post-RMS vector. Fix the
already registered unit e and the two output readers V=(e,g), where g is the
normalized runs/run unembedding contrast. No new factor fit or selected module:
MLP17 is the existing final-layer dossier subject.

Let F_e = D diag(Re)L + D diag(Le)R and a=D[(Le)*(Re)]. Then

    f(u+delta*e)-f(u) = delta*F_e*u + delta^2*a.
    K=V F_e; c=V a; selected response=delta*K*u+delta^2*c.

In code, readers are matrix rows. K is2x1152 and c has2 entries. Computing K
directly costs O(p*m*d) for p readers without forming the d-by-d matrix F_e or
the three-index tensor. Initialize q=Ku in O(p*d). Each subsequent command uses
O(p) arithmetic and p state values:

    response(q,delta)=delta*q+delta^2*c;
    q_next=q+2*delta*c.

The identity K e=2c proves exact sequential composition for repeated shifts
along this same e. Bias cancels from the response. This is an extracted local
response transducer conditional on the initial context; it is not an absolute
MLP evaluator or a token-to-answer program. Initialization still requires u.

The surrounding18-layer model has native residual mixing,9 attention heads per
layer with128-dimensional score/value factors, RMS normalization and final
30*tanh readout over50304 outputs. Attention has two multiplied QK scores, no
softmax, and shared first-layer values. Its free unnormalized local expression
has degree5; RMS square roots and tanh prevent global polynomial closure.
The local MLP is degree2, the response is affine in u and degree2 in delta.

Local factor permutations, reciprocal L/R scalings and L/R exchange preserve
the quadratic tensor. Reader-state basis changes q'=Hq are legal with the
corresponding output coefficients; the state coordinate basis is not unique.
Changing e to -e requires delta to -delta. Arbitrary residual-basis changes do
not automatically commute with component products and native normalization.

## Exact sufficiency and its boundary: our derivation

For fixed V and a nonzero allowed delta, equal selected response for u and u'
is equivalent to K(u-u')=0. Therefore the minimal *linear* context information
for these selected responses is the row space of K, with dimension rank(K).
This is a kernel statement, not variance preservation or a fitted-rank claim.
It does not establish any autonomous update under other native layers.

For the entire output, a proposed linear state S u is sufficient on an open
input domain iff ker(S) is contained in ker(F_e). Equivalently row(F_e) is
contained in row(S). Proof: subtract two responses at the same nonzero delta;
the quadratic constant cancels and leaves delta*F_e*(u-u'). For a constrained
reachable set the statement only applies to differences attainable in that set.

An executable witness also preserves the normalized-input magnitude. Let P be
the orthogonal projection onto span(e,K_e,K_g), p in its range and z in its
kernel. Then u+=p+z and u-=p-z share e^T u, Ku and ||u||, but their full response
difference is2*delta*F_e*z. Any deterministic program that sees only these shared
features must predict the same response for the pair. Its optimal common
prediction is their mean, and its minimum relative pair error is

    ||response+ - response-|| /
       sqrt(2*(||response+||^2+||response-||^2)).

This is a falsifier of the specified state on the chosen domain, not a lower
bound for every nonlinear representation, natural-text distribution or circuit.

## Literature mappings and limits

* [CLUE](https://arxiv.org/abs/2004.11961) computes the smallest constrained
  linear reduction of polynomial ODEs preserving specified linear observables.
  Mapping: start from V and close its row space under polynomial Jacobian
  coefficient matrices. Our single F_e response has a directly computable
  kernel criterion; the full normalized discrete transformer is not that ODE.
  A naive invariant-space closure over N dense coefficient matrices requires
  at most d accepted independent rows and O(N*d^3) arithmetic (our bound,
  excluding expansion). Minimal observable row space is fixed; a basis need
  not be unique. We have no full native polynomial coefficient representation.

* [Demin, Demitraki and Pogudin](https://arxiv.org/abs/2301.11653) extend exact
  polynomial-ODE reduction to a longest chain of linear reductions using
  representations of finite-dimensional algebras. This is directly relevant to
  the user's hierarchy idea when the hierarchy preserves computation. It does
  not make geometric token clusters invariant under a transformer. Algebra
  generation and invariant-subspace work replace arbitrary clustering; there
  can be multiple valid chains. No end-to-end runtime or recoverability bound
  is claimed here without constructing that algebra and checking assumptions.
  Thus it is a possible later method, not today's extraction algorithm.

* [Active-subspace gradient methods](https://arxiv.org/abs/1506.04190) estimate
  eigenvectors of a gradient-derived matrix. For this response, the gradient
  with respect to context is the constant delta*K, so our exact row-space result
  needs no sampling or eigengap heuristic. A dense SVD of p-by-d K would cost
  O(p*d*min(p,d)); it changes coordinates, not the closure domain. For nonlinear
  native producers, small average gradient eigenvalues would provide only an
  approximation under its measure, not arbitrary-intervention sufficiency.

* [Weighted-automata/linear2-RNN recovery](https://proceedings.mlr.press/v89/rabusseau19a.html)
  has a precise finite-state counterpart: a sequence of commands can be viewed
  as transitions of our two-state transducer after affine homogenization.
  Native text transitions are not known to be linear or finite-Hankel-rank.
  Spectral recovery needs informative prefix/suffix blocks; an SVD of a p-by-q
  block costs O(p*q*min(p,q)), before shifted-block recovery. Minimal linear
  realizations are identifiable up to a state basis under the appropriate
  rank/coverage assumptions. Our command composition proves neither a language
  parser nor fresh-text prediction.

[Kruskal-type uniqueness](https://arxiv.org/abs/1304.8087) still requires suitable
factor independence for the specified tensor; tied inputs, overcomplete native
factors and downstream normalization preclude importing semantic uniqueness.
[Graph-width contraction](https://arxiv.org/abs/quant-ph/0511069) can bound
evaluation cost of a given network, not discover its task variables. Tensor
trains/hierarchical Tucker and arithmetic bilinear complexity reparameterize or
price local contractions; they do not replace the kernel/intervention closure
condition. No new rank/variance sweep is selected.

## Executed consequence and result

[Frozen protocol](SELECTED_READER_RESPONSE_V1_PREREGISTRATION.md) and CPU
[result](SELECTED_READER_RESPONSE_V1_RESULT.json): all A/B/C predictions held.
The trained MLP17 program predicts both selected responses at relative error
1.01e-13; sequential composition error1.59e-13; updated state error2.51e-16.
All16 equal-state reflection pairs nevertheless differ in their full output
response. Minimum possible common-prediction relative errors range .1087–.2416.
Equal-feature and equal-squared-norm relative errors are <=2.35e-16.

The synthetic contexts have squared norm d/2, strictly inside the native RMS
map's ball; a suitable raw input exists for each. They are not claimed to be
reachable from text. Intermediate tested shifts also stay inside that ball.
No native body forward or GPU was used; runtime1.79seconds. Program K/c stores
2306 float64 coefficients,18448 raw tensor bytes. Two initialized state scalars
plus the command suffice for the local response; all model-level producers and
545902902 native parameters remain dependencies elsewhere. No whole-model saving.

This review advances extraction and composition for a precisely bounded local
operation, and prevents extending that claim to unmodeled outputs. It does not
advance behavioral selective removal or natural-text OOD by itself.

## Decision

Prefer folding the newly exposed contextual readers through actual upstream
producers and tracking downstream readouts, with live normalization explicit.
The exact local transducer is reusable machinery for those tests. Do not grow a
rank until arbitrary full-output reconstruction passes, and do not interpret
reader-specific exactness as a complete circuit. A high-information next native
test distinguishes the scalar numerator's context gate from the normalization
gain and attention consumers left live by the16:42 block. Before that, reuse the
compiled program on saved/native contexts to verify the chosen readout semantics;
do not refit it. The paired closure witness is already the performed CPU
continuation receipt. Next mathematical review due19:49; hourly17:14 unchanged.
