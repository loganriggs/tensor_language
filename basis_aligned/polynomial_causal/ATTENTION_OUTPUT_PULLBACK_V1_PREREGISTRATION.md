# Entire centered unembedding through MLP17 and attention17 output map

Registered before native execution,10September. Prior-art: backward unembedding
fold through MLP16 held attention17 fixed; QK1/QK2 task splitting did not factor
this terminal path. Last-attention specialists and MLP17's existing dossier
are known, not rediscovered here. No new token data or outcome labels.

Let b be the residual before the attention write, z the concatenated9×128
head outputs before c_proj, and O=c_proj.weight. The raw MLP17 input is
b+Oz. With Q_v the quadratic numerator from centered U folded through MLP17,
the exact polynomial numerator is

$$
b^\top Q_v b+2b^\top Q_v O z+z^\top O^\top Q_v O z.
$$

Divide by the native MLP-input RMS squared including epsilon; preserve the
MLP bias, residual readout, final RMS and tanh. This experiment studies coefficient
structure of the three displayed ports, not the whole end-to-end logit function.
z still contains live QK1/QK2 routing, value mixing and token history; none is
assumed constant or independently explained. b and z need not be independent
on actual inputs. Coefficient energy uses formal independent coordinates and
does not measure their in-distribution energy or causal importance.

Compare three fixed maps:

- Native O: pull both quadratic input slots back through attention output.
- P O: random orthogonal P, seed77119. Preserves O's singular values and column
  Gram, but scrambles its residual-space alignment to MLP17. Controls metric
  anisotropy versus alignment, not natural-text distribution.
- O P: same P on the right. Preserves the pulled tensor's full Frobenius energy
  and output-mode spectrum exactly, while scrambling the nine head-coordinate
  blocks. This is an algebraic head-partition control, not a replacement model.

For each map compute exact centered coefficient output spectrum and9×9
input head-pair energy table via native hidden-product Gram contractions.
Off-diagonal matrix blocks counted twice in the full symmetric tensor norm.
Also compute residual/residual and mixed residual/attention energy separately;
do not silently omit mixed ports. Common vocabulary output remains separate.

Predictions:

- A: dense toy pullback/head/mixed-port/normalized-polynomial controls pass;
  native head-energy sum matches output spectrum trace, no material negative
  eigenvalues, native centered residual total matches prior receipt, and O P
  preserves native-O output spectrum and total, relative errors<=1e-10.
- B: native-O top128 output capture is at least5percentagepoints above the
  original centered tensor, and at least2points above P O. These are coefficient
  sharing screens only, not circuit identification or a matched physical metric.
- C: native-O within-head diagonal blocks contain at least50%of attention/attention
  coefficient energy, and exceed O P by at least10percentagepoints. A failure
  rejects this concentrated-head prior, not cross-head/open circuits.

Zero body forwards, FP64, managed runner only,900second alarm. No optimization
required: spectra and fixed-coordinate energies are exact computations. Intermediate
hidden Grams use about5GB; persist only compact JSON. All native weights, residual,
normalizations, QK/value computation and discarded approximation remainders stay
charged. A gain explained equally well by P O is not attributed to learned alignment.
Nulls receive metric/alignment/head-partition red-team interpretation before selecting
any new factorization; no arbitrary head labels or independent circuit claims.
