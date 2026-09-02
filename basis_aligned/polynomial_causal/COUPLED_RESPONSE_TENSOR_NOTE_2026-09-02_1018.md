# Coupled response tensors as an identification test — 2026-09-02 10:18 UTC

This is a technical consequence of the 10:10 three-hour mathematical review, not a second scheduled review. Rung480
is still collecting its frozen attention0 responses. This note fixes what a tensor-decomposition fallback would have
to do to count as circuit progress if rung480 returns its registered scientific null.

## The target is not low rank

The program wants units that state what information is read, what computation is performed, what is written, and
which later computations use the write. Native heads and MLPs are not assumed to be the right units: a valid method
must be able to join parts of different modules that later computation treats as the same variable and split one
module when its parts have different uses. The proposed units must then predict held-out and shifted examples,
support extraction, and permit selective removal or editing without damaging unrelated circuits. Parameter count,
rank, variance explained, and reconstruction error are controls or eventual prices, not evidence for those claims.

## Exact attention0 object

For rung480, one circuit `c` and one view `v` (document half, matcher source, or independent fit) gives a response
tensor

`R[v,c] in R^(7 x 7 x 33)`.

The three axes are the two score branches and the output carried by the attention operation. Each axis has one fixed
affine coordinate, leaving varying dimensions `6 x 6 x 32`. Under legal orthogonal changes of the three varying
coordinate systems, the complete tensor changes by multilinear multiplication with the three rotations. Its entries
are therefore not individually meaningful, but a subspace in each mode can be meaningful if it transforms with the
gauge.

A coupled block-term hypothesis would be

`R[v,c] = sum_r G[v,c,r] x1 U[r] x2 V[r] x3 W[r] + error[v,c]`,

where `x1`, `x2`, and `x3` mean contraction along the three tensor axes. The core `G[v,c,r]` may change with the
downstream circuit and view, while the subspaces spanned by `U[r]`, `V[r]`, and `W[r]` must be shared. A block can
contain pieces contributed by several native heads, so this does not treat head identity as ground truth.

Coupling is essential. A decomposition fit independently to one circuit response has ordinary rotation and component
ambiguities. Requiring the same blocks to explain all discovery circuits, document halves, matcher sources, and the
independent attention-block refit supplies additional equations and can make a block decomposition unique when a
single tensor is not. The block-term literature gives uniqueness and algebraic-computation results for specified
multilinear block ranks, and coupled tensor decompositions can have uniqueness conditions that are stronger than the
conditions for any constituent tensor alone. Relevant primary sources are Domanov and De Lathauwer's
[rank-(1,L,L) block-term result](https://doi.org/10.1137/18M1206849), their
[general block-term uniqueness and algebraic algorithm](https://doi.org/10.1137/23M1557246), and Sorber, Van Barel,
and De Lathauwer's [coupled CP/block-term uniqueness analysis](https://doi.org/10.1137/140956853).

Those theorems do **not** yet solve our case. We have not established the required block ranks, genericity, full-column
conditions, or separation of the factors, and the observed attention block has a dense Tucker-like core rather than
an established CP form. Robust Kruskal-rank results such as
[Bhaskara et al.](https://arxiv.org/abs/1304.8087) become applicable only if a CP restriction is first supported. We
must not impose that restriction and then call the resulting coordinates identified.

## Executable pre-fit test

Before optimizing a block-term model, test whether the response family even supports common sectors:

1. Remove the distinguished affine coordinates; never rotate them into the varying coordinates.
2. For each mode, unfold every `R[v,c]` into a matrix `X[v,c]` and form symmetric cross-contractions such as
   `sym(X[v,c] X[v',c']^T)`.
3. Ask whether this complete family has a nontrivial common invariant-subspace decomposition. Computationally, solve
   for matrices that approximately commute with every contraction and calibrate the tolerance on planted-block and
   independently rotated controls.
4. Fit the candidate projectors on only one document/source view. Require the projectors, block sizes, and circuit
   response profiles to transfer to the other half, matcher source, and independent refit after legal gauge
   alignment. Independently permuting circuit labels is the negative control.
5. Only a transferred block earns an exact removal test. Project its contribution out inside the attention
   computation, allow all downstream layers to recompute, and require the intended circuit effect plus preservation
   of unrelated circuits on the reserved odd-root families.

The pre-fit test is useful because a trivial commutant rejects the direct-sum block hypothesis before any flexible
factor model can interpolate the data. Rung479 found a trivial/unstable common algebra for the specific MLP8/9/12
equality-state family. That closes that family, not the method for a new response tensor. Conversely, a nontrivial
commutant is only a screen; the held-out response and physical-removal tests supply the circuit meaning.

## Exact MLP0 fallback object

The MLP0 dossier already provides an exact natural-context decomposition

`MLP0 write = T(token) + C(context) + I(token,context) + S(normalization) + retained numerical remainder`.

Its SELECT Shapley CE benefits are `T=1.4983`, `C=.4177`, `I=1.5375`, and `S=.0675` nats, so the token and interaction
branches are both important and the context-only branch is smaller but not negligible. Earlier work already rejects
a universal sparse token code, whole-vocabulary interchange, response-weighted PCA, fixed low-rank quadratic
producers, and a Tucker refactor. The next MLP0 experiment must therefore collect the **downstream effect** of the
exact branches for the existing 62 circuit tags, rather than re-factor the same weights by reconstruction.

For branch `b in {T,C,I,S}`, circuit `c`, and view `v`, define

`E[b,v,c] = CE effect on c after removing branch b and recomputing every later layer`.

To split a branch further, retain its exact token/context input indices and its 1,152-dimensional write, and measure
the first-order response tensor before any candidate grouping. Shared components are then subspaces whose response
profiles agree across circuits and views, not nearby token vectors or low-rank directions by themselves. In
particular:

- tokens can be grouped only when the same downstream readers treat their branch contributions equivalently;
- one MLP0 branch can be split when its parts have different 62-circuit response profiles;
- a component shared by `T` and `I`, or by MLP0 and attention0, must use the same downstream-effect variable even if
  its native coordinates differ; and
- token-private residuals remain exact rather than being called noise merely because they resist compression.

The 62 circuits act as coupled views that can improve identifiability. A block-term or common-invariant-subspace
model is licensed only after the exact response collection shows common sectors. It is killed by a trivial
commutant, failure across document/matcher splits, or failure of selective removal.

## What tensor-network canonicalization contributes

The minimal canonical form of Acuaviva et al. uses geometric invariant theory to decide tensor-network gauge-orbit
equivalence in a broad setting: two tensors have the same minimal canonical form exactly when their orbit closures
intersect. See [Minimal canonical form of a tensor network](https://arxiv.org/abs/2209.14358). This can give us a
principled equality/duplicate check and remove arbitrary gauges before comparing two fitted circuits.

It does not choose a semantic circuit, identify which downstream task uses a block, or guarantee selective removal.
Canonicalization is therefore an invariant preprocessing or audit tool, not the objective.

## Result-conditioned action

- If all rung480 scientific gates pass, do not detour into another decomposition: run the registered exact removal
  on odd-root circuits first.
- If rung480's in-run instrument is valid and its scientific strong null fires, start the exact MLP0 `T/C/I/S` by
  62-circuit response collection. Apply the common-sector pre-test above before any coupled block fit.
- If only rung480's stored cross-session bridge fails, preserve the original failure and run the already-queued
  in-session bridge repair before interpreting its scientific statistics.

This route can change the grouping, splitting, held-out prediction, and selective-removal circuit goals. It is not
licensed to continue if its only result is a smaller rank or fewer parameters.
