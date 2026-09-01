# Three-hour mathematical review — 2026-09-01 18:23 UTC

## The goal and the new fact that changes the object

The program goal is an executable, predictive, manipulable tensor program for bilin18: a smaller description that
names what the model computes, reproduces the relevant behavior on fresh data, and supports causal edits. A compact
activation fit is not enough. Every proposed object must state its literal price, survive held-out and physical
interventions, and distinguish discovery, identification, and adoption.

Rung416 established a useful negative. A rank-64 residual-stream subspace of attention0 writes is stable across FIT
halves (`0.853` projector overlap), reconstructs much of the centered MLP0 interaction, and has a physical CE effect.
But it is not a shared multi-head vocabulary: the median effective writer count is `1.962`, head3 is the largest
writer for 61/64 modes, and the ordinary total-write covariance basis beats the head-pooled construction for both
the token-by-context and context-only paths. Output geometry is real; it is not yet functional identity.

The user's stronger proposal changes the equivalence relation. Two written directions should count as the same thing
when named downstream consumers cannot distinguish them. Conversely, two consumers may read the same input direction
but produce different outputs. This writer/reader dual is the object to test before decomposing query/key weights.

## 1. Exact local object: finite downstream responses, not head labels

For document-position sample `n` and attention0 head `h`, rung402 gives an exact centered MLP0 interaction write

`I[n,h] = gbar * Down(Left(delta_token[n]) * Right(delta_head[n,h])
                    + Left(delta_head[n,h]) * Right(delta_token[n]))`.

Here `delta_head[n,h]` is head `h` after its native output projection and subtraction of its FIT mean. The observed
BF16 head-sum and arithmetic closing remainder is kept as `NUMERIC`; it is not assigned to a semantic head.

Let `A1(W)` and `M1(W)` mean the actual 1,152-dimensional attention1 and MLP1 writes when the MLP0 interaction port
is set to `W`, while raw-token reinjection, attention0, every other MLP0 component, the first-value bus, RMSNorm, and
all weights remain native. Define two finite response backgrounds:

`R_single[n,h] = [A1(NUMERIC+I_h)-A1(NUMERIC), M1(NUMERIC+I_h)-M1(NUMERIC)]`,

`R_drop[n,h]   = [A1(FULL)-A1(FULL-I_h), M1(FULL)-M1(FULL-I_h)]`.

Each bracket is a 2,304-dimensional vector. The first asks what a head path does when added to the empty semantic
interaction boundary. The second asks what it does at the native full-interaction operating point. Their difference
measures nonlinear interaction with the other head paths. “Head `h` and head `k` compute the same service” now means
their response signatures are close on held-out contexts, consumers, and both backgrounds—even if their raw writes
are not close. This is distribution- and consumer-relative, not an absolute identity of vectors.

The dual reader question is separate. If attention1 and MLP1 are sensitive to the same residual input subspace but
map it to different output directions, the shared object is the input subspace plus consumer-specific output maps.
It must not be collapsed into “same output.”

## 2. Tensor-network environment: the exact analogy and its limit

In a linear tensor-network contraction, removing one tensor exposes an environment: a linear map from the open bond
to the final state. For a fixed pure state and Hilbert-space squared norm, the Schmidt/reduced-density spectrum gives
the optimal retained subspace, and discarded eigenvalue mass is the truncation error. This is the mathematical core
behind DMRG ([White, 1992](https://doi.org/10.1103/PhysRevLett.69.2863)). It explains why an environment-defined
basis is more relevant than local covariance.

The exact mapping for this project is:

- open bond: the 1,152-dimensional MLP0 interaction write;
- environment: the downstream block1 consumers, later suffix, and loss;
- candidate bond states: the nine `I_h` writes and their combinations;
- relevance metric: finite changes in named consumer writes or ultimately logits/CE.

But the DMRG theorem does **not** certify our proposed quotient. The bilin18 suffix contains RMS normalizations and
bilinear maps; the environment changes with the injected bond state. CE is not Hilbert-state squared error. We use a
finite natural-text distribution rather than one globally specified state. Therefore a response SVD is exactly
optimal only for the sampled response matrix in Frobenius norm, not for the full nonlinear model or CE.

## 3. Balanced realization and observability: useful vocabulary, bounded claim

For a stable linear dynamical system, controllability measures which state directions inputs can reach and
observability measures which directions outputs can distinguish. Balanced realization makes the two Gramians equal;
truncating small Hankel singular directions has system-level meaning
([Moore, 1981](https://doi.org/10.1109/TAC.1981.1102568)). In our writer/reader language:

- the covariance/action span of the `I_h` paths is a finite controllability analogue;
- the map from those paths to attention1/MLP1 responses is a finite observability analogue;
- a low-dimensional shared service must be both written and discriminated downstream.

Empirical balanced truncation extends this by simulating perturbations of nonlinear systems, but it is a heuristic
outside the linear case ([Lall, Marsden, and Glavaski, 2002](https://www.cds.caltech.edu/~marsden/bib/2002/06-LaMaGl2002/LaMaGl2002.pdf)).
The prior global loss-gradient observability quotient already found ranks 677–825, or 59–72% of the stream. That
closed the claim that one small global first-order stream quotient solves the model. The proposed test is different:
it uses finite, exact, isolated `I_h` interventions and immediate named vector consumers, not infinitesimal CE
gradients over every state direction.

Hermann–Krener nonlinear observability gives a local rank condition using outputs and their Lie derivatives under all
admissible inputs ([Hermann and Krener, 1977](https://doi.org/10.1109/TAC.1977.1101601)). Bilin18 is finite-depth,
tokens are discrete, and we sample two intervention backgrounds rather than all controls and Lie derivatives. Hence
failure of this finite consumer bank means “not equivalent for these measured consumers/backgrounds”; success means
held-out empirical indistinguishability. Neither is a theorem of globally minimal nonlinear realization.

## 4. What linear algebra can solve exactly here

After collecting the finite response tensor, flattening samples, backgrounds, and consumers gives a response matrix
`Y` and a corresponding action matrix `X`. The singular value decomposition gives the best rank-`r` approximation to
either matrix in squared Frobenius error ([Eckart and Young, 1936](https://doi.org/10.1007/BF02288367)). Reduced-rank
regression gives a best linear predictor of sampled response from sampled action at a stated rank. These are exact
finite-matrix optimization statements.

They do not establish semantic uniqueness. A downstream metric `G = average(J^T M J)` from Jacobians gives only a
local seminorm around one operating point. The finite two-background response tensor is therefore the first assay;
a tangent Gramian is a later explanatory approximation only if it predicts these finite responses.

## 5. Bilinear block decompositions and gauge

For a bilinear tensor `W[out,left,right]`, a rank-`(L,L,1)` block has many left/right input combinations sharing one
output direction. That is the “different inputs, same output service” side. A Tucker/block term with a shared left
or right input factor and a multi-dimensional output factor captures “same input vocabulary, different partners or
outputs.” Block-term decompositions formalize these structures and can be essentially unique under rank and
independence conditions ([De Lathauwer, 2008](https://doi.org/10.1137/070690729)); CP has its own sufficient
Kruskal-rank uniqueness condition ([Kruskal, 1977](https://doi.org/10.1016/0024-3795(77)90069-6)).

Those uniqueness conditions are hypotheses to check, not facts about MLP0. Arbitrary factor matrices remain gauge
dependent. Stable subspaces, complete contracted bilinear maps, and downstream response classes are the safer
objects. A canonical tensor-network gauge can identify equivalent network representations in its own setting
([Acuaviva et al., 2022](https://arxiv.org/abs/2209.14358)), but bilin18's data-conditioned RMS/bilinear residual
program is not directly the tensor-network-state object of that theorem.

For attention, each head score is a product of two bilinear query/key matches. With rotary position encoding, one
half is a relative-offset family `Q_h^T R_delta K_h`, not just `Q_h^T K_h`. The later weight-level object should be

`score_h = sum[r,s] core[h,r,s] * form1_r(query,key,delta) * form2_s(query,key,delta)`.

This can reveal one QK half reused across heads with different partners. Yet shared score factors are not complete
functional equivalence: different patterns may retrieve downstream-equivalent values, and identical patterns may
write distinguishable values. QK factoring follows, rather than replaces, the response-defined writer/reader test.

## 6. Other exact-looking directions reconsidered

- **Polynomial identity testing.** After clearing RMS denominators one could, in principle, compare rational suffix
  functions by polynomial identities; randomized identity testing has formal guarantees
  ([Schwartz, 1980](https://doi.org/10.1145/322217.322225)). BF16 arithmetic, approximate rather than exact semantic
  equality, large degrees, and data-relative equivalence make this a poor next experiment.
- **Weighted-automaton/Hankel minimization.** It is exact for rational sequential series with a complete prefix/suffix
  basis, but no new sequential state object is supplied here and prior token-splice probes were out-of-distribution.
- **Another global Fisher/gradient quotient.** Already answered negatively at the global stream: stable but broad.
- **More covariance rank tuning.** Rung416 already showed that covariance geometry can be causal without identifying
  shared producers. More ranks would optimize the rejected object.

## 7. Protected decision: rung417 finite-response head-service assay

The next rung will collect the exact two-background response tensor above on the frozen 96 FIT and 96 SELECT
documents, scored positions 64:256, without opening FINAL. It will stream sufficient statistics where possible.

The first falsifiable question is intentionally coarse: can the response of dominant head3's `I_3` path be predicted
from the other eight head paths on held-out documents more accurately than its raw `I_3` write can? This directly
tests redundant producers of one downstream service. A fixed document/position-shuffled response relation is the
negative control. The assay also compares action and response head-mode spectra, FIT/SELECT Gram stability, and
agreement across singleton/removal backgrounds and attention1/MLP1 consumers.

Frozen high-level bars, made literal in the preregistration, are:

1. exact rung402 interaction closure and exact native block1 replay;
2. response head-mode is materially more compressed and stable than action head-mode;
3. held-out head3 response reconstruction from the other heads is at least `R2=.50`, at least `.20` above action
   reconstruction, and at least `.30` above the shuffled response control;
4. the equivalence is not created by averaging incompatible consumers or operating points.

This is mechanism identification only. It stores no deployed substitute and licenses no compression. If it passes,
the next object is a shared response basis plus the dual reader factorization, followed by physical grouped
interventions. If it fails at attention0, that does not refute copy-head redundancy elsewhere; it moves attention0 to
the cross-head double-QK shared-half vocabulary and repeats the response definition at attention1.

## Operational consequence

Rung417 is higher-information than an immediate weight factorization because it first defines what “the same” means
at the behavioral interface. It is also cheap: block1 is evaluated directly after exact MLP0-port edits, so the GPU
budget is dominated by 96+96 documents and nine heads rather than 18-layer CE forwards. Only after the instrument and
held-out equivalence bars survive should we fit block-term or QK vocabularies to explain the identified classes.
