# Hourly strategic review — 2026-09-01 19:35 UTC

## Goal

Compile bilin18 into a smaller predictive and manipulable tensor program. For the present attention0/MLP0 arc,
identify gauge-safe input features, their bilinear pairings, and output effects without mistaking architectural heads,
private coordinates, tensor-fit quality, or a favorable allocation artifact for mechanisms.

## What changed this hour

1. The July ticks169–187 prior-art audit is now §14 of the current user explanation. It already provides a validated
   per-head sparse `(key1,key2,value)` CP pipeline and corrected nulls; no reason remains to rebuild it as new work.
2. Rung419 rejects a discrete shared K256 output-payload vocabulary. The tiny global/private advantage is reproduced
   by token-row permutation and raw geometry; head3 consumes61% of private centers. Downstream transport is real but
   cannot rescue the vocabulary claim.
3. Rung420 rejects the proposed single rank24 global QK carrier. The average-projector modes are reproducible and
   connect Q, K, and MLP0's linear token path, but the spectrum is graded (`lambda_1≈.90`, `lambda_24≈.45`) and
   carrier removal leaves14.3% pairwise overlap instead of the .3% null floor.
4. The positive result is narrower and stronger than “nothing shared”: Q/K carriers overlap61%; their overlap with
   leading MLP0 L modes is73–76% and with normalized token input z is84–85%, while overlap with nonlinear Q is≈0.
   The common object is broad token-input geometry, not one finite vocabulary or one low-rank carrier.

## Mathematical diagnosis

If every branch were `S_e=C direct-sum P_e` with a common rank24 subspace C and mutually generic private spaces,
then the top24 eigenvalues of the average projector would be near1, carrier-only SELECT overlap would be near1, and
residual overlap would be near the random floor. Instead the average-projector spectrum declines smoothly from .90
to .45 and the residual remains .143. The data fit a graded factor model

`P_e ≈ sum_r loading[e,r] * c_r c_r^T + private_e`,

with non-binary loadings and likely several partially shared blocks. Calling the first24 directions “the carrier”
discretizes a continuous sharing spectrum and is therefore the wrong model class.

The very high z alignment also explains why raw weight-space decompositions can look shared without identifying a
downstream mechanism: all early token functions inherit the same embedding manifold. A useful decomposition must
condition on the realized QK score and named downstream response, not merely recover that common input manifold.

## Ranked next directions

### 1. Coupled continuous QK1 × QK2 × OV blocks in a realized common token basis

Reuse the exact folded tables and old asymmetric-core solver, but replace per-head sparse-code coordinates by an
observable common token basis derived from normalized token input plus the validated downstream U16 output interface.
Fit partially shared blocks, not fully tied atoms:

`edge_h(q,t,delta) ≈ sum_b (phi(q)^T A[h,b,delta] phi(t)) * (psi_h(t)^T C[h,b]) * u_b`.

The block index may share one query/source mode or one output mode while retaining head-specific partners. Compare
against equal-price independent-head blocks and a token-permuted-factor transplant. Score on held-out offsets/tokens,
natural-edge response, and CE. This directly answers the user's “same input, different output” and “different input,
same output” cases. It must explicitly avoid ticks207/208's closed naked weight-space CP: use the realized token
measure and downstream output quotient.

Before a large fit, register a small rank ladder and literal storage formula. Selection should use FIT; SELECT chooses
only among preregistered arms; FINAL remains closed. A tensor-fit pass licenses an intervention, never adoption.

### 2. Decompose MLP0's nonlinear token remainder Q

R420 says the shared early-token carrier is almost orthogonal to Q. That makes Q a cleaner target for the user's
request to group tokens by extra features written beyond their broad linear identity. On exhaustive tokens, compare
block-sparse, hierarchical, and DAG dictionaries in output-action space, with equal-price PCA/SAE controls and
downstream MLP1/attention1 response. This is likely the shortest path to a readable MLP0 token-only dossier, but it
does not solve the attention interaction path.

### 3. Move the functional-equivalence test to attention1

Rungs416–419 show attention0 is unusually individuated, especially head3. Copy/retrieval redundancy is a stronger
prior at attention1. Apply the exact finite response quotient there, below whole heads and with Q/K score halves
separated. This is high-value if the attention0 coupled blocks fail, but it should not interrupt the now well-defined
layer0 continuation.

## Decision

Proceed with direction1 as the next registered identification screen. In parallel planning, specify direction2 as
the MLP0-focused fallback. Do not tune K, carrier rank, whole-head clusters, input metrics for MLP0-p448, or naked
weight-space CP; each is now closed by controlled evidence. Preserve FINAL.
