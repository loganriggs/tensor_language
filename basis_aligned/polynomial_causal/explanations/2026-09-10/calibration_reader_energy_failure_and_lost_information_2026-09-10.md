# Vocabulary-score spread loses information needed by the calibration scalar

The proposed vocabulary-spread computation fails both tests: its weights do not
match the folded calibration quadratic, and it does not reproduce that scalar's
effect on real text. Replacing the scalar with this formula causes almost as much
prediction damage as removing it. We have closed this particular explanation,
without fitting new coefficients to the evaluation examples.

The mathematical follow-up is more informative than the failed approximation.
Using the saved weight matrices, we constructed two continuous inputs with the
same norm and the same vocabulary-score spread, but calibration values differing
by approximately248,000. The spread discards a relative-sign interaction that
the calibration quadratic reads. These are constructed local inputs, not natural
text examples or a proof about the complete model's reachable states.

## What was tested

The existing scalar is q(u)=u^TQ u+beta, obtained by folding the final bilinear
MLP's weights into the original frozen output direction. U is the model's
unembedding matrix, which maps a residual vector to vocabulary scores.

We tested the explicit operation

\[
E(u)=\operatorname{Var}_{v\in\mathrm{vocabulary}}[(Uu)_v]=u^TKu,
\qquad
K=\frac{(U-\bar U)^T(U-\bar U)}{V},
\]

where bar U is the mean vocabulary row and V=50,304. This is the variance of
scores across vocabulary entries, not entropy and not an activation-variance
preservation objective. The candidate producer was

\[
q_E(u)=\alpha E(u)+\gamma\|u\|^2+\beta.
\]

We determined alpha and gamma entirely from the weights, before evaluation.
After removing each matrix's trace component, alpha is the least-squares
coefficient matching Q0 to K0; gamma restores the trace. The resulting values
are alpha=590.361 and gamma=−7.53044. No activations or labels fit these numbers.
The explicit norm term retains the native normalization's epsilon dependence.

We checked the global matrix relation and the native-input approximation
separately. A matrix mismatch alone need not disprove an approximation on a
restricted distribution of real inputs.

## Both levels fail

The global residual ||Q−alpha*K−gamma*I||/||Q0|| is **0.9905**. The registered
numerical compatibility threshold was1e-10. The Gram computation itself is
accurate: direct score variance and the quadratic contraction agree to5.4e-16
relative error on the fixed control inputs.

On the reused42 FineWeb rows and16 Pile documents:

| Measurement | FineWeb | Pile |
|---|---:|---:|
| Relative scalar error | 0.970 | 0.965 |
| Full-vocabulary replacement error, relative to complete removal | 0.979 | 0.973 |
| Prediction-loss increase from q_E | 0.322 nats | 0.363 nats |
| Prediction-loss increase from complete q removal | 0.338 nats | 0.380 nats |

All native sufficiency requirements fail by wide margins. Mean predicted q_E
is432/608, compared with native q means29,756/31,080. This is not a near miss.
The first stability experiment's native/removal/mean controls replay within
the registered tolerance. Native online projection replacements also agree
with the formula within4.19e-5 maximum absolute logit error and5.86e-7 relative
error. The failed explanation is therefore not due to a broken editing instrument.

The run used18 model-body forwards and70 sequence instances in6.36 seconds.
All545,902,902 native parameters remain; U alone has57,950,208 parameters.
Calling U a shared reader does not remove those costs. No circuit was adopted.

## An explicit equal-energy counterexample

Because K is symmetric, choose orthonormal eigenvectors e_i,e_j with eigenvalues
lambda_i,lambda_j. This uses the standard spectral decomposition of symmetric
matrices; see [MIT's symmetric-matrix notes](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/resources/mit18_06scf11_ses3-1sum/).
For radius r, define

\[
u_+=\frac r{\sqrt2}(e_i+e_j),\qquad
u_-=\frac r{\sqrt2}(e_i-e_j).
\]

Both have squared norm r² and energy r²(lambda_i+lambda_j)/2. But

\[
q(u_+)-q(u_-)=2r^2e_i^TQe_j.
\]

Whenever this cross coefficient is nonzero, energy and norm alone lose the sign
information needed by q. In exact arithmetic, the two inputs disprove any
single-valued decoder q=f(E,||u||²) on a domain containing them, including a
nonlinear decoder. This is our direct sign-flip construction, not an assumption
that diagonalizing a matrix discovers semantic circuits.

For the saved matrices, the largest off-diagonal coefficient in K's eigenbasis
connects its two largest-eigenvalue directions. We used r²=576:

| Quantity | u+ | u− |
|---|---:|---:|
| Squared norm | 576 | 576 |
| Vocabulary-spread energy | 80.05355 | 80.05355 |
| Calibration q | 191,220.32 | −56,774.36 |

The computed q difference is247,994.68 and agrees with the analytic expression.
The eigendecomposition residual is2.5e-15; energy agrees exactly in the reported
FP64 calculation, and relative norm difference is2.0e-16. These are numerical
checks on the stored FP64 Gram, not an exact rational certificate for every
original weight contraction.

The vectors can be obtained by the stated RMS normalization of continuous raw
inputs: choosing a=sqrt(2*epsilon)*u gives RMSNorm_eps(a)=u when ||u||²=576 in1152
dimensions. The inverse check agrees to1.8e-15. This establishes a legal local
continuous-input construction. It does not establish that either vector occurs
on natural text or that the full model can reach it without an intervention.

## What the failure changes

The fixed weight-derived score-spread formula is closed. The counterexample
shows that a future input explanation must preserve relevant interactions, not
just an aggregate energy. It does not license a new gain fit, vocabulary subset,
rank sweep or a claim that the scalar is confidence or entropy.

The earlier conditional removal and donor results remain valid. Stable
identification, semantic meaning, independent upstream extraction and reduced
structural description remain missing. Any proposed signed-reader computation
would need its own specified inputs, causal tests and full cost accounting.

Evidence: [preregistration](../../CALIBRATION_READER_ENERGY_V1_PREREGISTRATION.md),
[native result](../../CALIBRATION_READER_ENERGY_V1_RESULT.json),
[continuous-input witness](../../CALIBRATION_READER_ENERGY_WITNESS_V1_RESULT.json),
[witness implementation](../../calibration_reader_energy_witness_v1.py).
