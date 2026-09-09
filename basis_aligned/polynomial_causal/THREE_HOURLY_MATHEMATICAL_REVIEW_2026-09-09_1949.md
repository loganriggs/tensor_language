# Mathematical checkpoint — 2026-09-09 19:49 UTC

The controlling objective is the bilinear reconstruction handoff's appended
criterion: discover an explicit reusable computation with held-out/OOD output
prediction, extraction, removals and composition, and a smaller structural
description charging all adapters and opaque weights. Exact CSE is insufficient.

## Actual object

The trained restriction remains attn4-rms-seed0: four layers, D128, four heads
of width32, vocabulary29, T51, causal product attention divided by32², absolute
RoPE, affine-free RMS and residual interpolation by .5. The head is linear;
there is no final normalization. This is not the full eighteen-layer model.
Each attention summand has two query factors and three source factors after
normalization. RMS prevents a globally polynomial residual map. Original token
projections at the first layer are fixed finite tables, however, so the extracted
document summary is affine in its position/entity incidence matrices.

The exact object for this review is `s=o+A x`, A512x1058, with two independent
24x24 bijective incidence matrices centered by 23-dimensional contrasts. The
interventions are simultaneous record permutations, acting as Ppi on x.
Preserved observables are all four 128-dimensional summaries, not only answers.
The linear closure question is whether a smaller z=Jx supports every Ppi and
decodes Ax. Norm-bounded approximate transport uses relative Frobenius error;
the fixed cap is ||T||₂<=1e6 in native summary coordinates. Contrast basis changes
are orthogonal gauges; the cap is not invariant under ill-conditioned rescaling.

The contraction graph is a source sum feeding a shared query initializer, then
three contextual layers. The summary's first-layer query is hop-only, so 24
query entities share it. Later normalized transitions remain explicit and native.
All 387968 export constants remain charged. The 512x1058 analysis operator adds
541696 values, about4.33MB FP64; it is not an adopted implementation. No tensor
exceeds256MiB and no fitted candidate feature bank is introduced.

## Literature mapping and alternatives

[CLUE](https://arxiv.org/pdf/2004.11961) supplies the closest exact algorithmic
match. Its Algorithm2 takes an observable row space and finitely many linear
operators, and returns the smallest common invariant space containing it.
PropositionI.1 proves minimality; PropositionIII.1 gives expected arithmetic
cost O(r n (T+r)), where T counts operator nonzeros. Here the operators are
record permutations and the initial rows are A. This linear-algebra subproblem
applies directly. Its polynomial-ODE reduction theorem does not apply unchanged
to our discrete RMS transformer. Minimal row space is unique, not its basis.
We derive a special group-action shortcut below instead of forming global
Jacobians or invoking ODE semantics.

[Weighted tree automata](https://proceedings.mlr.press/v51/rabusseau16.pdf)
map subtree values to multilinear states and contexts to linear readers;
Theorem2 reconstructs a minimal WTA from a Hankel rank factorization for a
rational tree function. Our finite permutation action could be encoded by
linear operators, but the full normalized model lacks this specified WTA
representation. Building a global Hankel matrix has no justified finite cost
here. The local invariant-space calculation already addresses the restricted
observable question without that construction; no full-model minimum-state
or semantic uniqueness guarantee is imported.

[Markov and Shi](https://arxiv.org/abs/quant-ph/0511069) bound simulation of
T-gate quantum circuits by T^{O(1)} exp[O(d)] for treewidth d. This motivates
controlling contraction width, but evaluating a supplied contraction graph
does not identify shared causal variables. We have not bounded the relevant
width of an expanded transformer graph, and RMS remains an explicit operation.
This alternative offers no circuit-identification guarantee for the present test.

The literature search also revisited tensor rank/identifiability, hierarchical
factorizations, polynomial identities, bilinear complexity and invariant methods.
[Robust Kruskal identifiability](https://proceedings.mlr.press/v35/bhaskara14a.pdf)
concerns CP factors under rank conditions; we have not established those
conditions for a causal tensor whose factors predict the registered interventions.
TT/Tucker factorizations or polynomial identity checking alone would evaluate
or reorganize a specified expression, not resolve its missing semantic interface.
They are demoted here, rather than treated as executable solutions without
assumptions or computational bounds.

## Executed consequence and direction

The complete self-contained derivation is
[summary_record_transport.md](explanations/summary_record_transport.md).
First, the reverse triangle inequality gives a transport obstruction without
rounding small singular values to zero. Two fixed reorderings give relative
lower bounds .11069 and .11749 at the norm cap; native correspondence3.11e-15.
Second, products of star transpositions span End(R23). Consequently the minimal
observable closure is R23 tensor the span of the 46-dimensional parity/entity
slices of A. This shortcut uses O(512*23*46²) floating arithmetic for the fixed
full-column check, followed by O(46³) finite-field arithmetic for its minor.

The CPU audit already completed .059s. A nonzero exact dyadic minor modulo
2147483647 certifies slice rank46 for the stored operator, hence closure1058.
Its singular minimum .0483 provides a numerical scale, not an interval proof
of the unrounded network. No smaller exact linear permutation-closed incidence
interface contains these stored observables. This rejects a particular reuse
abstraction without head/rank/gain sweeps, not nonlinear or finite-domain codes.

This mathematical route dominates fitting arbitrary linear transport adapters:
it settles the restricted question essentially immediately. It does not replace
the full-output experiment: downstream computation may cancel summary changes.
Next inspect the native full-distribution response to the two fixed record
permutations, preserving all worlds and queries. A semantic function-only
abstraction predicts invariance; paired Jensen–Shannon divergence lower-bounds
its teacher KL if native order dependence remains. The next bounded action
should reuse the existing full executor and paired-bound scorer, without fitting.

Next mathematical review is due22:49 UTC; hourly review is independently due19:51.
