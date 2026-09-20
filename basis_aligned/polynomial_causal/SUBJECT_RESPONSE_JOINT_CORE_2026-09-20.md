# Joint-tensor response compiler and the next baseline

The six-MLP conditional chain is an executable fidelity baseline, not a simple
circuit. v646 tests whether its quadratic **response** products are necessary,
retaining the exact changed RMS denominator. It is queued behind a confirmed live
shared-lane fitting job; no native v646 outcome is asserted here.

## Exact finite-response identity

For P0=D[(Lh)*(Rh)], s0=mean(h^2)+eps and s1=mean((h+d)^2)+eps,

```
delta = d + {D[(Ld)*(Rh)+(Lh)*(Rd)+(Ld)*(Rd)]}/s1 + P0*(1/s1-1/s0).
```

Bias cancels locally but must remain in the baseline recurrence. The CPU control
checks a multiblock response against independent complete evaluations, including
bias and explicit epsilon. The omission experiment removes only `(Ld)*(Rd)`;
it is not a Taylor approximation to RMS. Its registered limits are5% effect error
against the conditional dense chain and10% against the full native effect in every
one of eight now-opened position/direction cells.

## Joint projected tensor

Let d=Pz and let Q contain output readers. Write C=Q^T D, A=LP, B=RP. Contract
the full bilinear object into two fixed tensors:

```
K[a,p,j] = sum_k C[a,k]*(A[k,p]*R[k,j] + B[k,p]*L[k,j])
G[a,p,q] = sym_pq sum_k C[a,k]*A[k,p]*B[k,q]
```

Then the projected response, including its residual carry, is exactly

```
Q^T P z + (K[z,h] + G[z,z])/s1 + m0*(s0/s1-1)
m0 = Q^T P0/s0
s1 = s0 + (2*(P^T h) dot z + z^T(P^T P)z)/1152.
```

The compiler [projected_bilinear_response.py](projected_bilinear_response.py)
keeps K, G, carry and norm geometry. It is exact for any specified P,Q; approximation
enters when an evolving residual response is constrained to their spans. P need
not be orthogonal. A float64 test uses nonorthogonal input features and arbitrary
output readers, checks an independent dense native contraction, verifies reciprocal
L/R gauge invariance, and detects omission of the quadratic response. Together
with the chain test, two tests pass. No native subspace-fidelity result exists yet.

## Sparsity and cost

For input/output width8, G has36 distinct unordered products shared across eight
output coordinates; dense storage is512 coefficients (288 would suffice if packed
symmetrically). K has73,728 coefficients and64 background-dependent linear
couplings. Those dense background readers dominate the core. A narrow latent
dictionary is not synonymous with few mixed interactions or sparse input features.
The present compiler stores83,584 values per block: K, full symmetric G, carry,
norm Gram and input basis. Six blocks plus the final basis, two answer readers and
six residual scales would store513,030 values, roughly2.05MB at float32. This is
an algebraic price, not a validated compressed program.

The price must additionally charge six background h vectors, six projected
baseline writes m0, the input response and final baseline state. Producing these
still requires the native model. The compiler does not remove those ports. It
should only be judged against the dense chain at this same conditional boundary.

## Registered next native discriminator: width-eight shared response spaces

Use the128 original subject-number prompts as calibration only. Obtain seven
width-eight orthonormal response bases from the baseline-versus-frozen-attention
state differences after blocks11–17. Freeze these before testing on the48
position-shift prompts (already opened, not newly held out). This is a data-informed
basis with exact weight contraction, not weights-only discovery or parameter training.
Compare with a fixed-seed matched-width random basis and the dense conditional chain.
Do not sweep widths or select a winning basis using the test panel.

The full exact contraction is the replay gate (relative/absolute error1e-4 where
meaningful). The reduced chain must meet5% conditional effect error and10% full
native effect error in every cell. Report target norms, cosine, calibrated versus
test error, all native background ports, fixed tensors and sparse index costs.
A failure means this width-eight response-space hypothesis failed; it is not a
bound on other shared DAGs. A success nominates a causal/selective test and genuine
new OOD prompts, not immediate circuit adoption. Native implementation remains the
next step; the joint compiler and its independent tests are already executed.

## Executed follow-up and numerical red team

v646's exact finite-response replay passes at4.07e-6 maximum margin error.
Omitting response quadratics gives7–21% conditional and11–23% full native effect
error; both substantive gates fail. These interactions cannot be dropped on this
panel. This says nothing about whether their joint tensor admits a simpler basis.

v647 implements the width-eight comparison but is **numerically invalid under its
registered absolute gate**: relative local replay1.42e-6, absolute replay.00412.
The projected baseline-write ports were sourced from native float32 writes while
the independent contraction uses float64 polynomial numerators. The original
receipt remains intact. v648 recomputes these ports directly in float64 with the
same data, bases, ranks and thresholds; no gate is relaxed. The repair has now
passed: absolute replay3.64e-10, while both effect gates still fail. See the
[subsequent observable-basis audit](SUBJECT_OBSERVABLE_BASIS_AUDIT_2026-09-20.md).
The original v647 instrument remains invalid and is not itself promoted.

A planted four-block circuit with a known three-feature invariant response space
is recovered exactly by the compiled recurrence despite arbitrary biases and
background states. A matched unrelated space fails its10% gate. This provides a
positive implementation control independent of native success. Together with the
finite-response and single-block tests, three CPU tests pass.

## Reusable sufficient background coefficients

`prepare_context` contracts K with the baseline h once. Each subsequent response
evaluation then needs only the resulting8x8 linear map,8 overlaps P^T h,8 projected
baseline writes and one baseline squared RMS:81 floating values per block. A
similar exact contraction closes final RMS/softcap to two baseline raw logits,
eight basis overlaps and one squared RMS. Across six blocks plus initial latent
coordinates this is505 per-example floating context values, excluding IDs and
metadata. Their production still requires the native background generator and the
mixed tensors. End-to-end cost is not reduced merely by renaming these inputs.

`evaluate_prepared` and `readout_prepared` consume those coefficients without full
residual vectors. CPU checks verify their equality with the original contraction,
including nonorthogonal norm geometry and reused contexts at edit scales
-.5,0,.5,1.5. This is algebraic reuse, not yet independently validated reuse of a
semantic feature by multiple circuits. The prepared execution coefficients would
store3,926 fixed floating values (dense cores, carry/Gram matrices, scales and final
projected readout); the producer tensors still cost513,030 values and remain charged.
