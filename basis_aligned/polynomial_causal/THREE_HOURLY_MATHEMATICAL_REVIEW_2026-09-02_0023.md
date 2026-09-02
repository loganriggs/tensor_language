# Three-hour mathematical review — 2026-09-02 00:23 UTC

## Goal and current decision

The goal is a smaller independently executable tensor program for bilin18 whose structure predicts fresh/OOD behavior,
supports circuit extraction and selective removal, and composes without damaging unrelated mechanisms. Storage and local
reconstruction are candidate simplicity measures, not the goal itself.

The immediate frontier changed at rung454. Vocabulary-program rank perfectly orders preservation of an attention16
removal effect, but the registered normalized composition ratio anti-orders reproducibly and triggers its strong null.
The correct response is not to retro-pass vocabulary. It is to recognize that the two consequence labels are mixed
finite differences of a tensor program and audit whether the composition denominator makes destructive candidates look
artificially good. Preserve rung454, freeze a mathematically uniform fixed-scale companion metric, and require independent
rows before using that metric to reconsider any teaching target.

## 1. Exact mixed-difference algebra

For the vocabulary family, let `L(U,h)` be the vector of per-token cross entropies after applying output vocabulary map
`U` to final hidden state `h`, including the model's `30*tanh(logits/30)` soft cap. Let `U_P` be candidate `P`, `h` the
native hidden states, `h_R` the states after the fixed attention16 removal, and `h_Q` the states after installing the
fixed MLP16 partner.

The discrepancy in preserving removal is exactly

`D_R(P) = [L(U_P,h_R)-L(U_P,h)] - [L(U,h_R)-L(U,h)]`.

The non-additive composition interaction is exactly

`D_Q(P) = L(U_P,h_Q) - L(U_P,h) - L(U,h_Q) + L(U,h)`.

Both are rectangular mixed finite differences: `Delta_P Delta_R L` and `Delta_P Delta_Q L`. They are not analogies to
an interaction; they are the discrete second-order interaction coefficient on the two-variable intervention lattice.
For candidates elsewhere in the model, replace `U_P` by the corresponding program state and the same identity holds.

Rung454's removal error uses the fixed denominator

`||D_R(P)|| / ||L(U,h_R)-L(U,h)||`,

so every candidate is measured in the same units. Its registered composition error instead uses

`||D_Q(P)|| / ||[L(U_P,h)-L(U,h)] + [L(U,h_Q)-L(U,h)]||`.

The latter denominator depends on `P`. A very destructive low-rank vocabulary map produces an enormous candidate-only
effect, inflating the denominator and making its normalized interaction appear small. This is not hypothetical. Along
uniform independent rank label0→512, the raw interaction norm falls `82.15→58.10`, while the additive denominator falls
faster `934.15→468.80`; the ratio therefore gets worse `.08794→.12392`. Across all14 frozen adjacent-rank edges, raw
interaction magnitude improves14/14 while the registered ratio improves only5/14. Rung454 remains false as written,
but its failure exposes a target-definition problem rather than document noise.

A candidate-independent companion is

`C_fixed(P) = ||D_Q(P)|| / ||L(U,h_Q)-L(U,h)||`.

The denominator is the fixed native effect of the same partner `Q`, exactly parallel to removal. On the already-open
TEACHING outcomes this post-hoc diagnostic orders all14/14 vocabulary rank edges, the five MLP0 ranks, and the MLP-PCA
8+17 rank ladder. Those facts are discovery only; vocabulary requires independent outcome rows before this metric can
carry confirmatory weight.

## 2. Tensor-network environment interpretation

Cut the computation graph at a local program tensor `T` and contract everything else—including inputs, downstream
readers, softmax loss, and an intervention—to form its **environment**. To second order, the mixed difference is

`D_Q(T + delta T) approximately delta T contracted with H_(T,Q) contracted with delta Q`,

where `H_(T,Q)` is the cross block of the loss Hessian or an empirical response/Fisher approximation. In ordinary
tensor-network language, a local tensor should be truncated under the metric induced by the contracted environment,
not under its raw Frobenius norm. This is the same mathematical lesson repeatedly measured here: covariance/projector
rank alone can fail while a response-weighted quotient succeeds.

This connection is established tensor-network methodology:

- second renormalization improves local truncation by including the tensor environment rather than using a local SVD
  alone ([Xie et al., Physical Review Letters 2009](https://doi.org/10.1103/PhysRevLett.103.160601));
- full and neighborhood updates explicitly differ by how much of the surrounding environment enters the truncation
  error metric ([Dziarmaga, Physical Review B 2021](https://doi.org/10.1103/PhysRevB.104.094411));
- all single-tensor environments of a closed network can be evaluated with controlled shared contraction cost
  ([Evenbly and Pfeifer, Physical Review B 2014](https://doi.org/10.1103/PhysRevB.89.245118));
- differentiable tensor-network programs provide gradients and higher derivatives through contraction graphs, including
  stable differentiation of decompositions and fixed points
  ([Liao et al., arXiv:1903.09650](https://arxiv.org/abs/1903.09650)).

The mapping is not exact in every physical assumption: bilin18 has nonlinear RMS normalization, softmax/tanh loss,
finite text sampling, causal masking, and interventions rather than a closed quantum-state norm. But the environment
construction itself is exact for any differentiable contraction graph; what changes is the scalar/vector objective and
the approximation used for its Hessian.

## 3. Highest-information next computations

1. **Fixed-scale consequence audit, then independent confirmation.** From stored bundles, publish raw interaction norm,
   candidate-dependent denominator, and fixed-native-partner normalization for all three teaching families. Label this
   explicitly post-hoc. Before any new vocabulary outcome, freeze the formula and bars on the existing independent192-
   document role, then rerun the output-only vocabulary contraction there. Do not alter rung454 or count family3 from
   TEACHING alone.
2. **Environment predictor for vocabulary.** Because candidates change only `U`, compute the mixed Hessian-vector action
   between output-map error `delta U` and each intervention hidden-state change. Compare this local quadratic prediction
   with the exact mixed finite difference, with large-damage candidates separately reported to expose Taylor failure.
   This yields a mechanistic structural feature, not merely rank/bytes.
3. **General tensor-program environments.** For MLP0 and MLP-PCA, use automatic Jacobian-vector and vector-Jacobian
   products to contract the downstream environment around each changed core. Test whether environment-weighted residual
   energy orders removal and fixed-scale composition better than ordinary activation variance, price, or rank under
   leave-one-whole-family-out validation.
4. **Environment-guided truncation, only after prediction.** If the environment feature predicts held-out families,
   solve a generalized low-rank approximation under that metric—analogous to a full tensor-network update—and compare
   it prospectively with local SVD/PCA at matched price. The sealed attention0 family remains the untouched discriminator.

## Protected decisions

- Rung454 remains a registered strong null for its candidate-dependent normalized composition ratio; no threshold or
  verdict changes.
- Do not fit the consequence predictor with only MLP0 and MLP-PCA, and do not open attention0.
- Do not call the fixed-scale diagnostic confirmatory on TEACHING. Freeze it and test vocabulary on independent rows.
- If fixed-scale composition also fails independent rank/wave reliability, add a genuinely new outcome-free teaching
  family rather than tuning another denominator.
- Even if the fixed metric passes, the ultimate simplicity claim still requires leave-family-out prediction and the
  sealed attention0 test; an environment-weighted compression is not adopted from local fit alone.
