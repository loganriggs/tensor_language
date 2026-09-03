# Three-hour mathematical review — 2026-09-03 07:20 UTC

## Circuit interpretation target

The target is not a low-rank approximation. We want a smaller executable tensor program whose units say what is
read, what operation is performed, what is written, and which later computations use that write. The units may join
parts of different native heads or MLPs and split one native module. They must predict fresh and shifted inputs,
remain valid when composed, support a selective removal/swap/edit without damaging unrelated circuits, recur across
tasks when the same computation is reused, and survive document splits, fitting restarts, and the model's gauge
freedoms. Literal storage, compute, edges, state, and program length are prices after those circuit claims—not
substitutes for them.

Claude's 07:08 mathematical review and rung2668 are an important correction: the MLP10 rank-3 structure captures
about 12% of held-out effect energy and saves approximately zero prequential bits, despite explaining 76% of the
estimated noise-unbiased signal. Thus rank and in-sample signal fraction remain weak evidence. The live R522 route is
still stronger because it asks for held-out circuit prediction and two physical interventions.

## The exact R522 object

For recipient row `b`, token position `t`, natural donor map `m`, and direction `d`, let

`y[b,t,o], z[m,d,b,t,o] in R^1152`, with output index `o=1..1152`,

be the native attention8 write and the donor-minus-recipient write difference. A rank-4 orthonormal frame
`Q in R^(1152 x 4)` represents the gauge-invariant projector `P=QQ^T`; `Q` and `QR` are identical for every
`R in O(4)`. The physical interchange is

`y' = y + zP = y + ((zQ)Q^T)`.

With the real layers 9--17 and readout denoted by `F_x`, the saved scalar response tensor is

`G_Q[c,s,e,d,n] = CE(F_x(y+zP)) - CE(F_x(y))`,

restricted to matched member/control token pairs `n` for circuit `c`, document split `s`, donor ensemble `e`, and
swap direction `d`. The model suffix is a known differentiable contraction graph but not a polynomial: RMS
normalization and the final tanh cap are non-polynomial. Its weights are tied across tokens; `Q` is tied across every
row, token, donor, direction, and circuit. R522 preserves signed per-token effects, not merely mean CE.

The retained program state is 4,608 frame values plus four FIT means. A swap or mean-removal applies about
`2*1152*4` multiply-adds per token and one projector edge at attention8. These are matched prices across real, Haar,
recovery-only, and label-null controls. Adoption still requires the frozen A--D prediction, reuse, fingerprint,
damage, and removal gates.

## Candidate 1: active-subspace / ridge-function identification

For a fixed downstream context define `g_x(u)=CE(F_x(y+u))`. If there really is an exact ridge representation
`g_x(u)=h_x(Q^T u)`, then

`grad g_x(u) = Q grad h_x(Q^T u)`

and the gradient second moment `C=E[grad g grad g^T]` has image inside `span(Q)`. If the four within-subspace
directions are excited, `rank(C)=4` and its image recovers `P` up to the correct orthogonal gauge. This is the
strongest exact implication I found for the current object. Active-subspace work defines this gradient covariance
and ridge approximation explicitly; it also warns that the active subspace need not solve the globally optimal
ridge approximation ([Constantine et al., primary paper](https://www.osti.gov/servlets/purl/1538114)).

The assumptions do not currently hold as a theorem for R522. We observe `g_x` only along natural donor differences,
the context-specific `h_x` may change with `x`, and an approximately predictive finite swap does not imply that
`g_x` is constant in every orthogonal direction. Even a rank-4 gradient covariance could be a local tangent object
that fails finite removal. Therefore this cannot replace A--D.

It does yield an executable identifiability falsifier after R522: on held-out documents, measure the gradients of the
same circuit effects with respect to the attention8 write, estimate each target's gradient second moment, and test
whether (a) its reliable image lies in the frozen `P`, (b) the common image transfers to the omitted and fourth
targets, and (c) orthogonal gradients predict the residual finite effect. Compare projector overlap and held-out
response prediction against the same label-null family. Also report the singular values of the natural excitation
matrix `[(donor-recipient)Q]`; a missing fourth singular value means natural swaps cannot identify all four
directions. This tests stable identification and computational specification, not compression.

## Candidate 2: identifiable tensor factorization of the response array

If the five-way response tensor exactly had a CP form

`G[c,s,e,d,n] = sum_r A[c,r] B[s,r] C[e,r] D[d,r] H[n,r]`,

then a generalized Kruskal condition on the factor `k`-ranks can make the factors unique up to component permutation
and reciprocal scalings. Kruskal's original three-way theorem gives the core sufficient condition
`k_A+k_B+k_C >= 2R+2` ([Kruskal 1977](https://www.sciencedirect.com/science/article/pii/0024379577900696)); CP and
Tucker algorithms and their different uniqueness properties are reviewed by
[Kolda and Bader](https://epubs.siam.org/doi/epdf/10.1137/07070111X).

This does not exactly solve R522. The circuit masks give different and overlapping token supports, the tensor is
noisy and partly missing rather than an exact dense CP tensor, three experimental modes have size only two, and a
unique response factor is not automatically an attention8 activation projector or executable causal mechanism.
Tucker factors also retain rotational gauge and therefore do not solve identification by themselves.

The usable consequence is conditional: if R522 finds a broader-than-quartet variable, form the complete saved
`circuit x split x donor-ensemble x direction x member/control-response` array, fit CP only on predeclared training
entries, check the Kruskal ranks of the fitted factors, and require entrywise prediction on held-out circuits and
documents. A factor becomes a circuit hypothesis only after a corresponding activation intervention reproduces it.
This could discover cross-head/cross-MLP response groupings, but it is downstream of—not a substitute for—the live
physical experiment.

## Candidate 3: causal abstraction

Interchange-intervention training gives an exact abstraction guarantee when a low-level model matches a specified
high-level causal model under the required interventions
([Geiger et al. 2022](https://proceedings.mlr.press/v162/geiger22a.html)). The broader causal-abstraction framework
also formalizes graded faithfulness and mechanism transformations
([Geiger et al. 2025](https://www.jmlr.org/beta/papers/v26/23-0058.html)).

R522 currently has no named high-level program, and it matches scalar CE responses rather than all counterfactual
outputs of a specified causal model. Consequently an A--D pass is a stable, reusable, selectively manipulable
subspace on this intervention family—not yet an exact causal abstraction or semantic algorithm. This framing makes
the missing next evidence precise: infer a small high-level state/transition rule from the fitted circuit, then test
whether the diagram commutes on new interventions and shifted text.

## Decision and next executable step

No theorem found here replaces the sealed R522 test. The current route remains highest-information because it directly
addresses cross-circuit grouping, held-out/OOD-within-census prediction, selective manipulation, reuse, and gauge-
stable identification at a matched price. The active-subspace theorem adds the best genuinely different falsifier,
but running it now would change the frozen call budget and delay the stronger finite intervention test.

Therefore continue completing and auditing R522. If it passes its finite gates, preregister the gradient-image and
natural-excitation-rank test above before interpreting rank 4 as identified. If it fails because the fitted response
is broad, use the saved multiway response tensor for a held-out CP screen; if it fails because effects are below the
noise floor, neither more ranks nor another factorization is justified—more independent documents or a different
causal object is required.
