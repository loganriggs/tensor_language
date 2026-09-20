# Finite observable folding and dual response spaces

v651 uses calibration endpoint derivatives to select six directions in addition
to two fixed answer readers. The complete nonlinear forward core is retained.
It improves postposed conditional-effect error to1.8–8.3%, but fails the5%
all-cell conditional gate and10% full-native gate. Local derivatives are not
finite-effect predictions.

## Exact finite-reader fold

A CPU counterexample makes the distinction concrete: h=x(2-x), f=h^2. Both
endpoint derivatives at x=0 and x=1 vanish, while f changes from0 to1. This is
a composition of two bilinear operations with a constant coordinate.

For a normalized bilinear map, write P(h)=D[(Lh)*(Rh)], s_i=mean(h_i^2)+eps,
and m=(h0+h1)/2. Quadratic polarization and symmetric divided differences give
the exact finite-response reader pullback:

```
q_previous = lambda0 * [q + .5*(1/s0+1/s1)*J_P(m)^T q
                            - m*(q dot (P(h0)+P(h1)))/(d*s0*s1)].
```

Final RMS uses its symmetric reciprocal-square-root divided difference; each
softcap uses its own scalar secant. [Implementation](bilinear_chain_readers.py)
requires both endpoints. Its reader dotted with the state change equals the final
margin change at every boundary. It is therefore an exact calibration/attribution
object, not a predictor with free access to an unseen edited endpoint.

Three CPU tests verify an independent autograd derivative, finite-effect closure
at all boundaries, the repeated-endpoint derivative limit, and the quartic
counterexample. Native v652 finite-reader closure is2.99e-6 maximum absolute error.
Its eight-feature forward model still fails: conditional error3.7–5.3% preposed,
1.4–9.0% postposed. Exact observable accounting does not guarantee that an orthogonal
projection onto a few reader directions preserves the dynamics.

## v653: snapshot-balanced trial and test spaces

Let columns of X be calibration response snapshots and columns of Y be the exact
finite readers. For Y^T X = A Sigma B^T, take

```
V = X B_r Sigma_r^(-1/2)       # decoder / trial basis
W = Y A_r Sigma_r^(-1/2)       # encoder / test basis
W^T V = I.
```

This exactly retains the truncated cross-snapshot coupling; it does not guarantee
the finite nonlinear intervention error. The compiler contracts input slots with V
and output readers with W. It keeps V^T V for RMS geometry rather than pretending
the decoder is orthonormal. Initial encoding uses W0 and final decoding uses V6.

[balanced_response_frames.py](balanced_response_frames.py) is tested for
biorthogonality, equality to the truncated cross-snapshot SVD, and a planted example
where a small observable response must be kept instead of a large unobserved one.
v653 native duality error is5.77e-14 and local absolute contraction error2.34e-11.
Its extra initial encoder costs9216 values, giving522,246 fixed values rather than
513,030. This is matched width, not identical total storage.

Conditional errors are1.37% calibration,1.3–2.2% preposed and4.7–10.6% postposed.
Full-native postposed errors remain12–17%. Both substantive gates fail. The
balanced construction improves the represented input/output coupling, but has not
produced a circuit satisfying the requested properties.

## Next discriminator: calibration coverage, not a larger rank

All preceding bases learned only from subject-final calibration. v654 keeps the
balanced width8 and every threshold, adding earlier-subject versions of the original
128 noun/template calibration rows. The48 evaluation prompts and disjoint evaluation
nouns stay unchanged; this panel is opened. Calibration cost rises to256 examples,
and16prefix+20suffix calls plus8 endpoint-chain batches are charged. Native
calibration capability is reported. The run is queued; no outcome is asserted.

This tests whether the failure is tied to the restricted calibration interface.
It cannot establish prospective OOD success: even a pass needs genuinely new
evaluation, selective intervention and useful reuse. Native background generation
and the505 context coefficients remain explicit dependencies. No failed run is
discarded and no gate is relaxed.
