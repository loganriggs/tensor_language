# Exact final-readout root over approximate upstream fields

Native instrumentation passes: exact root replay5.33e-15, projected first/second
derivatives1.60e-14, native/double differences7.44e-6, prior native replay0.
All quadratic denominators stay positive. The all-method10%number/5%modal
prediction gate nevertheless fails; preserving the final nonlinearity is not
by itself sufficient for every method.

| Full-radius method | Output quadratic | Exact root, quadratic fields | Output cubic |
|---|---:|---:|---:|
| Original5 |7.773%|8.347%|3.416%|
| Full6 |17.987%|10.257%|7.208%|
| Substituted5 |16.751%|9.569%|6.385%|

Numbers are maximum number-effect errors over16cells per method. Every modal
error remains below5%. Substituted5 passes its field-program subgroup, but the
registered all-method gate is still false. These are opened diagnostics and
do not replace the original failed prospective quadratic result.

The program folds the final residual state through eight unembedding rows,
one shared RMS denominator and eight explicit softcaps. Nine quadratic fields
cost27coefficients per chosen ray including constants, versus16for the cubic
output jet. Direction storage and native field generators remain additional.
The native field experiment takes12prefix,112double and88native calls,
216first+216second full-suffix reverse calls and96+96cheap root reverse calls.
It uses no native third derivatives but does not win the coefficient-price
comparison. Lower derivative order does not mean fewer derivative calls.

## Third-order partition across this interface

For F(t)=R(z(t)), the chain rule is

    F3 = DR(z0)[z3] + 3 D2R(z0)[z1,z2] + D3R(z0)[z1,z1,z1].

Here zk means the kth derivative. The last two terms equal the third derivative
of R(z0+t*z1+t²*z2/2), available from the saved field program. Subtracting it
from native F3 infers the transported upstream-field term. It is not a separately
measured z3. A planted nonzero-z3 control closes this identity to7.89e-18; CPU
field-error replay agrees with the native receipt within3.41e-13.

In the originally failing opposite/subject/beside_subject-singular cell, the
substituted5 root term has norm0.555times the total third derivative and the
upstream-field remainder0.568times. These ratios need not sum to1: vectors
can align or cancel. Original5 has root ratio1.299, demonstrating cancellation.

Adding the inferred residual cubic term to the exact-root field response yields
maximum number error6.319% and modal2.371%, passing every full-radius cell.
This costs31coefficients per ray and needs both native third derivatives and
field generation. It is an attribution control, not a simpler extracted circuit.

## A more faithful next folding boundary

The field interface calls mean(h²) an upstream field. If h(t) is approximated
quadratically, its norm is **quartic**, not quadratic. Thus part of the inferred
upstream-field third term can be norm geometry computable from h1 and h2,
without a native h3. Do not attribute all of it to earlier modules yet.

For h(t)=h0+t*h1+t²*h2/2, fold the three state coefficient vectors through the
eight unembedding rows and a thin QR factorization. With phi=(1,t,t²),
mean(h(t)²)=||R phi||². Three shared quadratic features feed the squared norm;
the exact RMS and softcap then remain explicit. Numerators need24coefficients
and triangular R needs6, versus storing3*1152state values. This is a local
conditional readout price, not a whole-model saving; compare against the cheaper
16coefficient output cubic and charge native state-jet generation.

Native receipt: `source_ood_v2_readout_fields_result.json` in the followup folder.
CPU partition: `READOUT_CUBIC_PARTITION_V1_RESULT.json` and its audit script here.
