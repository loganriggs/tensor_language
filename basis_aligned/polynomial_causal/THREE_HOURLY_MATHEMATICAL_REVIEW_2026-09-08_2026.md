# Three-hour mathematical tensor-network review — 2026-09-08 20:26 UTC

## Why raw writer distance is not the causal geometry

At MLP11, suppressing position indices temporarily, let

\[
h_n(x)=(\ell_n^\top x)(r_n^\top x),\qquad
y(x)=\sum_{n=1}^{4608}d_n h_n(x).
\]

For task/cell `c`, the downstream answer functional supplies reader coordinates

\[
\rho_{c,n}=g_c^\top d_n,
\]

where `g_c` is the live downstream covector at the MLP11 output.  The first-order
task-functional response is

\[
q_c(\Delta h)=\sum_n\rho_{c,n}\Delta h_n.
\]

The latest exhaustive atlas compared `Delta h` to the full native `Delta h*` in
the Euclidean metric.  That metric implicitly gives every hidden factor reader
weight one.  It is not the causal metric unless the true reader Gram is the
identity, which the H/R artifact directly refutes.

For a bank of task readers, define

\[
G=\sum_c w_c\rho_c\rho_c^\top,
\qquad
\|\Delta h\|_G^2=\Delta h^\top G\Delta h
                =\sum_cw_c q_c(\Delta h)^2.
\]

This is a positive semidefinite seminorm.  Its null space consists exactly of
writer changes invisible to every registered reader.  Raw `H` norm describes
available/occupied hidden movement; the `G` seminorm describes movement readable
by the current task family.  This explains the v21 mismatch: L9H4 has behavioral
recovery `0.2201` but raw-H projection only `0.0270`, while broad early MLP writes
receive large raw-H credit.  Across all 108 heads raw-H and behavior magnitudes are
correlated in aggregate (Spearman `0.741`) but are not interchangeable for causal
ranking.

## Exact finite writer substitution

Let an upstream component add `W_j z` to the normalized MLP11 input.  The exact
hidden-factor change is

\[
\begin{aligned}
\Delta h_n(x,z)
  ={}&(\ell_n^\top x)(r_n^\top W_jz)
     +(r_n^\top x)(\ell_n^\top W_jz)\\
    &+(\ell_n^\top W_jz)(r_n^\top W_jz).
\end{aligned}
\]

After contraction with reader `rho_c`, this becomes an exact degree-`(1,1)+(0,2)`
polynomial

\[
q_{cj}(x,z)=x^\top C_{cj}z+z^\top S_{cj}z,
\]

with

\[
C_{cj}=\sum_n\rho_{c,n}
 \left[\ell_n(W_j^\top r_n)^\top+r_n(W_j^\top\ell_n)^\top\right],
\]

\[
S_{cj}=\sum_n\rho_{c,n}(W_j^\top\ell_n)(W_j^\top r_n)^\top.
\]

These are the `cross` and `self` tensors already implemented by
`bilinear_mlp_writer_capability`.  They are weight-defined and require no corpus.
They also preserve the MLP hidden-factor gauge: under
`ell_n -> a_n ell_n`, `r_n -> b_n r_n`, and
`d_n -> d_n/(a_n b_n)`, the induced reader coordinate scales as
`rho_n/(a_n b_n)`, so every term in `C` and `S` is unchanged.

This is the correct object for the user's proposal.  Stack it as

\[
\mathcal C[c,j,i,k],\qquad \mathcal S[c,j,k,l],
\]

over task/cell `c`, causally live component `j`, context coordinate `i`, and
component-write coordinate `k`.  Unlike an activation PCA, these tensors encode
every computation the restricted weights can perform.  An activation bank supplies
only the empirical distribution of `(x,z)` used to evaluate which parts are
occupied.

## Which decomposition statements are exact

For any bipartition of modes, reshape `mathcal C` into a matrix and take its SVD.
This is an operator-Schmidt decomposition of that partition.  Its singular values,
rank, stable rank, and truncated Frobenius error are invariant under orthogonal
coordinate changes within either side of the partition.  Eckart–Young applies to
that matricization.  It does **not** say that the resulting pieces are uniquely
semantic, sparse, or optimal under another partition.

HOSVD/Tucker applies one such SVD to every mode and stores a core plus orthogonal
mode factors.  It is deterministic once sign/degenerate-subspace conventions are
fixed and gives complete reconstruction when untruncated; it is not generally the
best tensor of a prescribed multilinear rank.  Kolda and Bader survey these
distinctions and the lack of a universal tensor analogue of matrix SVD
([SIAM Review, 2009](https://epubs.siam.org/doi/10.1137/07070111X)).

Tensor Train uses sequential unfolding SVDs.  Its TT ranks are the ranks of chosen
prefix/suffix matricizations, and its rounding algorithms are stable, but the ranks
depend on mode order ([Oseledets, 2011](https://epubs.siam.org/doi/10.1137/090752286)).
Hierarchical Tucker generalizes this to a dimension tree; hierarchical ranks depend
on the chosen tree even though each node rank is an invariant of that node's
bipartition.  Hierarchical SVD provides controlled near-best approximation in the
chosen format ([Grasedyck, 2010](https://epubs.siam.org/doi/10.1137/090764189)).

For the present four-mode tensor, the scientifically meaningful trees are not
arbitrary:

1. `((task, construction), (component, local coordinates))` tests whether tasks
   share a weight program.
2. `((task, component), (context, write))` tests task-specific routing through
   common local algebra.
3. `((task, construction), component)` followed by `(context, write)` tests the
   observed six-cell writer hierarchy directly.

We must report spectra for all three before selecting a hierarchy.  Choosing only
the tree with the smallest rank after inspecting results would be an unregistered
model selection step.

## Capability, occupancy, and use as separate contractions

The clean factorization is

\[
\text{weight capability }(\mathcal C,\mathcal S)
\xrightarrow{\text{activation }(x,z)}
\text{occupied response }q_c(x,z)
\xrightarrow{\text{causal patch/reset}}
\text{used edge}.
\]

- A low-rank direction in `mathcal C` with zero observed `z` occupancy is a latent
  capability.
- A high-occupancy direction in the null space of reader Gram `G` is present but
  unread by this task family.
- A large reader-contracted response that disappears under component reset and is
  restored under rescue is a causally used interface.

Activation PCA estimates covariance under the empirical `(x,z)` distribution and
may discard low-variance, high-reader-gain directions.  An SAE or hierarchical SAE
adds a sparsity model but does not change this logical status.  Sparse dictionary
identifiability theorems require assumptions on sparse coefficient generation and
dictionary structure; for example exact recovery results for Er-SpUD concern a
specific invertible-dictionary sparse-random model
([Adamczak, 2016](https://www.jmlr.org/papers/v17/16-047.html)).  Those assumptions
have not been established for our rectangular, gauge-coupled component tensors.
Accordingly, native checkpoint atoms plus exact reconstruction remain primary;
learned sparse atoms are secondary groupings until causal interventions distinguish
them.

## The six-cell H/R result and a minimal state model

Let `c` index the six A1/A2/P construction×direction cells.  The held-out result is

\[
\hat Q_c=\sum_p \bar H_{c,p}\odot\bar R_{c,p},
\]

with pooled cosine `0.9524` and recovery `0.9177`.  Replacing each state by an
additive panel-plus-direction model lowers cosine to `0.4297`.  Therefore the
minimum currently supported discrete program has six coupled cell states; an
additive factor graph is falsified at the H/R level even though an additive model
fits normalized Q directions well.

Reader stability (`0.9537` held-out, `0.9784` cross-fold) means we can freeze
`bar R_c` and analyze writer programs with the cell-indexed seminorm

\[
\|\Delta H\|_{R,c}^2
=\sum_s\left(\sum_n\bar R_{c,sn}\Delta H_{sn}\right)^2.
\]

Writer instability (`0.6457`, `0.7873`) says not to average H globally.  The failed
gain models additionally rule out replacing the cell state by `a(h,r) Qbar_c` with
a single writer, reader, product, or joint scalar law.

## Corrected causal protocol

The v21 exhaustive atlas remains a useful discovery measurement: all 108 heads
were patched, and the strongest behavioral heads—L9H1, L9H4, L8H1, L11H3—exactly
replicate the independently identified v15 four-head set.  Its union-selectivity
terminal is invalid because it simultaneously required complete donor closure and
quiet unrelated-donor controls.

The corrected prospective v22 test must separate three arms:

1. **Closure:** all-site base self-patch is inert; all-site donor patch reconstructs
   donor.  No selectivity bar applies to a full donor reconstruction.
2. **Component causal use:** singleton/set patch effects are scored on answer margin
   and frozen reader-contracted Q.  P/C compare the same component intervention,
   not the complete unrelated donor trajectory.
3. **Capability/occupancy:** raw H, `mathcal C/mathcal S` spectra, and activation
   coordinate frequencies are reported separately and cannot veto a causal effect.

Freeze the replicated four heads and the v21 module frontier before v22 outcomes.
Run singleton and complete-set arms.  If the set is selective but no head is large,
greedy addition uses held-out reader-contracted Q plus behavior; backward deletion
tests necessity.  For at most 12 final routes, enumerate the exact source-effect
game and pair interactions to separate redundancy from complementarity.  Only after
that causal set is fixed should its literal `mathcal C/mathcal S` tensor be subjected
to Tucker/TT/hierarchical decompositions and an occupancy overlay.

This protocol directly tests the hypothesis created by the new mathematics: the
cross-corpus four-head frontier should be low rank in the reader-contracted operator
even though it is not dominant in raw-H Euclidean geometry.  Failure of Q recovery
on v22 falsifies that shared reader-weighted program.  Passing Q but failing answer
behavior identifies a downstream nonlinear mismatch.  Passing both with P/C quiet
licenses the restricted weight-tensor hierarchy and later joint composition test.
