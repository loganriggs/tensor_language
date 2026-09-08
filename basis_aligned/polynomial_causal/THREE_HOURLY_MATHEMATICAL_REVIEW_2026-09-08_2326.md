# Three-hour mathematical tensor-network review — 2026-09-08 23:26 UTC

## Exact object: a causally restricted checkpoint program

The user proposed the correct separation: use data decompositions to describe support, but use a
causally established subspace to restrict the checkpoint and then decompose the resulting weight
operator. Let the residual width be `d=1152`, head-private width `p=128`, four admitted writers
`h in {1,...,4}`, and the frozen MLP11 reader basis have width `q=4`. The existing exact cross and
self tensors are

\[
 C_{haik}\in\mathbb R^{4\times4\times1152\times128},\qquad
 S_{hak\ell}\in\mathbb R^{4\times4\times128\times128}.
\]

For normalized block-11 context `x_i`, a head-local write `z_{hk}`, and reader coordinate `a`,
the restricted MLP11 response is

\[
 y_{ha}(x,z_h)=\sum_{ik}C_{haik}x_i z_{hk}
               +\sum_{k\ell}S_{hak\ell}z_{hk}z_{h\ell}.
\]

This is an exact degree-two checkpoint contraction at the fixed MLP11 boundary: the first term is
bilinear and the second is quadratic. It is not a claim that the complete transformer suffix is a
polynomial, because RMSNorm introduces state-dependent inverse norms and attention introduces
additional multiplicative routing.

Contracting the frozen leading occupied reader mode `r_a` gives

\[
 M_h[i,k]=\sum_a r_a C_{haik}\in\mathbb R^{1152\times128}.
\]

These four maps share a physical residual/context axis but have independent private head coordinate
gauges. Under `M_h -> M_h O_h` for orthogonal `O_h`, literal columns, flattened cosines, and right
singular vectors change; each context Gram `M_h M_h^T` and the sum of those Grams do not. The MLP
factorization also has reciprocal left/right scalings and hidden-factor permutations, while the
restricted residual interface `U` has the usual basis gauge `U -> UQ` paired with transformed local
coordinates. Checkpoint parameters are tied across examples and the same MLP11/reader weights enter
all four `C_h`; head output weights differ by physical head slice.

More generally, after a causal residual interface `U in R^{d x k}` has been fixed, the exact
weight-only objects are

\[
 A_h=U^T W_h^{write},\quad B_j=W_j^{read}U,
\]

plus the restricted MLP core

\[
 T_{abc}=\sum_n(U_{out}^TD)_{an}(LU_{in})_{nb}(RU_{in})_{nc},
\]

and separate restricted QK and OV maps. `A_h` enumerates what each writer can place in the interface;
`B_j` enumerates what each downstream component can read from it; `T` enumerates the bilinear
operations possible inside it. Writer and reader families should initially be decomposed separately,
because combining them in one Frobenius objective introduces a unit/scale choice. Their composition,
principal overlap, and causal interchange are the later shared-variable test.

Allowed inputs for the exact local identity are arbitrary finite `x,z` of the stated dimensions at
that boundary. Observed activations restrict these to a reachable subset but do not alter the
operator. Outputs to preserve are the signed frozen-reader response and ultimately paired answer
margin, with P/C controls. The current approximation norm is an explicitly weighted checkpoint
Frobenius norm; causal adoption additionally requires held-out intervention metrics.

## The exact common-context problem

For positive preregistered component weights `w_h` and fixed rank `k`, define

\[
 \min_{U^TU=I_k}\sum_h w_h\lVert M_h-UU^TM_h\rVert_F^2.
\]

Let

\[
 X=[\sqrt{w_1}M_1\;\sqrt{w_2}M_2\;\cdots\;\sqrt{w_4}M_4]
   \in\mathbb R^{1152\times512}.
\]

Then the objective is exactly `||X-UU^T X||_F^2`. The top `k` left singular vectors of `X` are a
global minimizer and the minimum is

\[
 \sum_{i>k}\sigma_i(X)^2.
\]

This is the Eckart–Young low-rank approximation theorem in projector form, with Mirsky's matrix
approximation generalization ([Eckart and Young, 1936](https://doi.org/10.1007/BF02288367);
[Mirsky, 1960](https://doi.org/10.1093/qmath/11.1.50)). It applies exactly because the fixed-rank
problem has been reduced to one matrix unfolding and the loss is Frobenius. Its assumptions do not
choose `k`, `w_h`, a causal subspace, or a semantic name. The projector is unique only when
`sigma_k > sigma_{k+1}`; even then, uniqueness is mathematical, not semantic.

Independent private rotations preserve

\[
 XX^T=\sum_h w_h M_hM_h^T,
\]

so the solution is gauge invariant on the problematic head-private axes. The dense SVD costs
`O(d P min(d,P))` for `P=sum_h p_h=512`, with storage `dP=589,824` floats for the four original
maps. A common-only representation stores `dk + Pk = 1664k` floats. At `k=32` that is 53,248 floats,
about 9.0% of the original, and it is smaller only for `k<~354`. If exact private tails are retained,
there is no compression claim: the tails are present specifically to permit a causal common-core
versus private-tail intervention. For a summed common output, evaluation similarly falls from about
589,824 multiply-accumulates to `1664k` for the common part; evaluating four separate reconstructed
outputs costs `4d k + Pk`. Tail cost must be added honestly.

## Neighboring tensor results and why they do not yet solve the circuit

### HOSVD, Tucker, and hierarchical Tucker

HOSVD applies SVDs to each mode unfolding and supplies orthogonal mode factors and a core, but is not
in general the globally best tensor of a prescribed multilinear rank
([De Lathauwer, De Moor, and Vandewalle, 2000](https://doi.org/10.1137/S0895479896305696)). A Tucker
or hierarchical decomposition becomes appropriate for the restricted `T[a,b,c]` only after its input
and output causal interfaces are fixed. Otherwise low multilinear rank can merely average distinct
writers/readers or exploit an opened dataset.

Hierarchical Tucker gives controlled near-best approximation for a chosen dimension tree and storage
on the order of `O((m-1)k^3 + mnk)` in the uniform setting, but both the tree and hierarchical ranks
are modelling choices ([Grasedyck, 2010](https://doi.org/10.1137/090764189)). A scientifically useful
tree might separate `(task, construction)` from `(writer, reader)` before splitting local input/output
coordinates. Inspecting several trees and publishing only the lowest rank would be post-selection.
Neither HOSVD nor HT identifies which branch is necessary, selectively manipulable, or reused.

### CP uniqueness

Kruskal's sufficient condition for essential uniqueness of a three-way CP decomposition is
`k_A+k_B+k_C >= 2R+2`, where the left side contains factor-matrix Kruskal ranks
([Kruskal, 1977](https://doi.org/10.1016/0024-3795(77)90069-6)). There is no valid direct mapping of
the present four `M_h` to one ordinary CP tensor without first identifying the independent private
head gauges. For the MLP bilinear core, shared/tied hidden factors, scalar reader contractions, and
large candidate factor count also make the required Kruskal-rank condition unestablished and often
impossible in the contracted mode. CP may still be a probe basis, but no uniqueness theorem currently
licenses its factors as circuit units.

### Weighted automata and Hankel minimal realization

Finite Hankel rank characterizes minimal linear representations of rational series/weighted finite
automata ([Carlyle and Paz, 1971](https://doi.org/10.1016/S0022-0000(71)80005-3)). Spectral recovery
also maps weighted automata to linear second-order recurrent networks
([Rabusseau, Li, and Precup, 2019](https://proceedings.mlr.press/v89/rabusseau19a.html)). An exact
mapping would require a sequence function whose prefix/suffix Hankel matrix has finite rank and whose
state update is linear/bilinear in the required sense. The live transformer path contains RMSNorm,
soft attention routing, and bilinear MLPs, so the theorem does not apply to the whole circuit. It could
apply only after a separately validated frozen-linear restriction; treating local tangent transport as
global linear realization would violate the assumptions.

## Executable consequence and opposing causal predictions

The best exact match, Eckart–Young on the shared physical axis, is now implemented in
`optimal_shared_context_subspace`. It accepts equal or heterogeneous private widths and optional
strictly positive component weights; forms the weighted unfolding; returns `U`, every private adapter
`U^T M_h`, reconstructed common map, and tail; and reports the singular spectrum, boundary gap,
relative weighted error, and discarded-singular-value optimum certificate. Thirty-seven tests cover
heterogeneous widths, independent private gauges, explicit weight changes, rank/weight failures, and
the exact certificate. This is an exact algorithmic consequence, not a decomposition analogy.

It creates a falsifiable prospective experiment once the downstream reader is fixed:

1. Freeze `k` and `w_h` without using the opened v23 causal outcome ladder.
2. On unopened v24, intervene separately with the shared maps `UU^TM_h`, matched private tails
   `(I-UU^T)M_h`, and their exact sum.
3. Require the exact sum to close the native restricted response. Shared-core installation must
   transfer across constructions/tasks, shared-core removal must be necessary, and P/C must remain
   quiet. A matched-energy private or rotated-core control must not do the same.

The opposing predictions are clear. If multiple circuit pieces genuinely write/read one variable,
the common core should preserve signed held-out response across component-private adapters and compose
with the independently restricted readers. If the high context-Gram overlap is merely checkpoint
geometry, private tails or matched rotations will perform equally and common-core reset will not be
selectively necessary. Dataset PCA/SAE can then annotate which validated branches are occupied or
discrete, but cannot rescue a failed causal reuse test.

## Mathematical route versus live empirical route

The exact solver dominates generic PCA/SAE/CP/HT as the current *capability* analysis: it is
gauge-safe, zero-forward, and has a global optimum certificate. It does not dominate the next causal
step. Complete MLP11 rescue explains only `.15114` of the four-head removal effect, so MLP11 is a
minor mediator; spending the live GPU lane on a hierarchy inside that branch would optimize a small
side path before locating the dominant reader. The higher-information order is therefore:

1. queued exact residual/M11 factorial;
2. reciprocal downstream reader localization on the dominant residual branch;
3. queued v24 capability and fixed unfiltered OOD confirmation;
4. only then common-core/private-tail swaps and restricted reader/writer composition.

This is a strategy confirmation, not a compression detour. The mathematical result supplies the
canonical, gauge-invariant candidate and its exact reconstruction control; the empirical route must
still establish that the candidate is used, shared, and selectively manipulable.
