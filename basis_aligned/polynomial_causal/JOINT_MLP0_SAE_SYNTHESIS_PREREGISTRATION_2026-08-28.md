# Joint MLP0 sparse-program synthesis: staged preregistration

Date: 2026-08-28  
Status: Stage 0 executable queued; later stages depend on its frozen decision and the
sealed finite-response backend.

## Why the older result is not enough

The historical weight-action SAE established a promising compressor, not a recovered
causal program. Its original real-layer run used one seed per sparsity and a fixed
step budget without a held-out convergence curve. A source audit corrects an earlier
claim in this preregistration: `Down` itself is bias-free and its forward hook fires
before the containing MLP adds the separate `Down_bias`, so the historical hook did
**not** omit native `Down_bias`. Its learned mean vector was a compressor intercept in
addition to the unchanged native bias. Later experiments found mean atom match only
about 0.40 across four seeds, while rank-64 decoder-subspace overlap was about 0.65. The jointly sparsified
`Down_0 -> Left_1` algebraic coupling also failed its strict causal prediction:
downstream-effect correlation was 0.217 rather than 0.4, and high- versus low-coupling
atom removal had nearly identical CE harm.

Accordingly, “an SAE fits” and “the SAE atoms are the model's causal components” must
remain separate claims.

## Target program

The desired object is an executable, jointly priced program

\[
g_0 \xrightarrow{\text{sparse producer}} z_0
\xrightarrow{\text{physical write and live residual/attention}}
\{\text{sparse consumer responses}\},
\]

where:

- the producer reconstructs MLP0's bias-free `Down` weight action with an explicitly
  priced learned intercept, while the separate native `Down_bias` is retained exactly;
- sparse coordinates may overlap rather than impose disjoint lexical clusters;
- downstream reader structure is selected jointly, not inferred from reconstruction;
- the intervening RMSNorm, attention, residual addition, and MLP1 bilinear map remain
  explicit nodes;
- program cost includes encoder, decoder, bias, sparse routing, reader parameters,
  indices, and actual product/linear operations;
- promotion requires held-out CE/KL, finite edits, OOD transport, selective removal,
  collateral damage, and seed/subspace stability.

Because top-k is nonlinear, a general `GL(P)` atom mixing is not a free gauge: it
changes supports. Encoder normalization, permutation/sign conventions, and stable
subspaces are the appropriate canonical checks. Full `GL` minimum-norm balancing is
used only on declared *linear* internal bonds.

## Stage 0 — optimizer and model-class discriminator (queued)

Executable:
`basis_aligned/bilinear_quotient/ops/mlp0_weight_sae_optimizer_discriminator.py`

Compare, at matched `P=512`, `k=32`:

1. historical positive hard-top-k with a learned intercept and exact external
   `Down_bias` retained;
2. signed top-k with unit-normalized encoder rows;
3. the signed model trained with calibrated 3% covariance-diagonal input noise.

Use 128 fit and 64 untouched evaluation documents, three seeds, 2,400 steps, and
held-out reconstruction curves. Freeze each decoder and directly refine held-out
sparse codes with iterative hard thresholding. The oracle refinement cannot earn
simplicity credit; it decides whether expensive alternating MOD/K-SVD is justified.

- Oracle gain at least 0.05: the amortized encoder/optimizer is a material bottleneck;
  next run is alternating sparse coding plus decoder update, followed by a separately
  trained executable encoder.
- Oracle gain below 0.05: classical inference cannot close much of the flat dictionary
  gap with the learned decoder; move to joint causal consumers or hierarchy rather
  than spending more optimizer budget.
- Noise is retained only if it gains at least 0.03 noisy-input R2 for at most 0.02
  clean-R2 loss.

## Stage 1 — joint producer and physical consumers

After Stage 0 freezes the producer family, compare:

1. independent producer/reader dictionaries;
2. the old algebraic edge penalty as a control;
3. a joint causal-response loss measured after the real RMSNorm/attention/residual
   path;
4. an equal-byte dense low-rank control.

The smallest executable joint grammar is

\[
u_0=p_0E_0+c_0,\quad a_0=\operatorname{TopK}(u_0),\quad w_0=a_0D_0,
\]
\[
u_1=p_1E_1+c_1+a_0C,\quad a_1=\operatorname{TopK}(u_1),\quad w_1=a_1D_1.
\]

Here `C` must actually execute; using it only as a training regularizer does not make
an executable sparse DAG. Compare six paired arms: native/native, program/native,
native/program, independently fit program/program, jointly fit program/program, and
jointly fit with the producer codes deranged. Every program arm must make zero calls
to the replaced native component and pay for all factors, indices, and routing.

The broader reader set includes both MLP1 bilinear banks and attention-1 query/key/value/output
responses that consume the post-MLP0 stream. A direct matrix product such as
`E_Left1 @ D_Down0` is only an algebraic diagnostic because attention and RMSNorm lie
between the two sites.

The stage passes only if its response operator predicts held-out finite edits better
than the independent and algebraic controls, while maintaining matched CE and cost.
The already-preregistered sealed 22-arm finite-response transaction supplies the
decisive causal currency.

## Stage 2 — hierarchy or DAG

Only if Stage 1 yields response-stable groups, compare:

- flat top-k atoms;
- group/tree-sparse codes;
- a DAG with shared parents and multiple lexical memberships;
- matched dense and shuffled-hierarchy controls.

The hierarchy is useful only if it reduces total producer-plus-reader description
length or executable operations at equal causal fidelity. Co-activation containment
alone is insufficient: previous activation hierarchy was real, while the executable
hierarchical native-Down replacement failed its internal-interface gate.

## Stage 3 — promotion

Evaluate on fresh large natural-text rows and at least one OOD/task distribution.
Require:

- nested data-doubling stability;
- whole-program CE/KL and top-1 agreement;
- response prediction under unseen finite edits;
- selective feature/group removal with bounded collateral harm;
- extraction success for named behaviors;
- serialized bytes and measured runtime/product-count savings;
- stable invariant subspaces across seeds, with individual atoms interpreted only
  when their stability is separately established.

Until these gates pass, the SAE is a promising compression grammar, not a completed
reverse engineering of MLP0.
