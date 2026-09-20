# Joint folded tensor: sparsity and instrument audit

The active object is C=UD, A=LE, B=RE, E=[I,lambda17[0] Down16,O17].
The native checkpoint gives z width **6912**, not the 7872 in v614's prose.
The tensor is symmetric in its two input indices. It is the last MLP's
unnormalized logit contribution; it excludes residual readout, final RMSNorm,
and softcap. Those operations must remain explicit during behavioral tests.
The native MLP also has `Down_bias`: its constant contribution is outside
C,A,B. It must be preserved separately, or represented by a homogeneous
constant coordinate. See the v622/v623 bias instrument correction in
[the mechanism audit](OUTPUT_COMPONENT_MECHANISM_AND_SHIFT_2026-09-20.md).

## What was wrong with the previous experiment

v614 divides its targets by mean(h²), but its Tucker predictor does not.
It therefore does not fit the declared normalized quadratic family. Its
held-out failure cannot be interpreted solely as an optimizer problem.
The fit also discovers from sampled activations rather than weights, and the
random token split within 40 sentences cannot establish OOD generalization.
Keep its receipt as historical evidence; do not reuse its pass/fail verdict as
a test of the corrected decomposition family.

Unit rows of W give vocabulary rows a fixed norm, not unit output atoms.
This restricts the model differently from normalized columns. Even normalized
columns alone do not remove basis rotations, correlations or duplicate atoms.
Counting abs(G)<1e-4 is not a scale-independent definition of circuit sparsity.

## Operational definition

For the first comparison, use orthonormal columns in W and P. This chooses a
restricted, well-defined family; it is not a claim that all useful dictionaries
are orthogonal. Optimize or compare frames separately from core support.

* Interaction sparsity: number of retained (alpha,p,q) with p<=q. Off-diagonal
  terms occur twice in tensor Frobenius energy, but need one s_p*s_q product.
  In fixed orthonormal frames, select support by G² times this multiplicity.
  Hard zeroing gives exact support and measurable reconstruction loss.
* Feature sparsity: support of P in explicitly declared original source ports.
  Dense P must be charged in full, even when G is sparse. QR reduction is exact
  algebra, not free feature extraction: the executable expanded P is charged.
* Output sparsity: support and storage of W. Dense vocabulary adapters count.
* Reuse: count distinct pair computations separately from edges to output
  features. A shared product is computed once but each coefficient is stored.
* Full simplicity: count factor values, core coefficients, support metadata,
  products, adapters, and all upstream operations needed to produce z. The
  current local price excludes those upstream operations, norm and readout;
  it is not the cost of a complete extracted circuit.

Report error versus executable cost, not sparsity percentage or equal rank
alone. Weight-tensor Frobenius error uses Euclidean coordinates in this specific
z space. Rescaling source ports changes that metric; source gauges must stay
fixed across methods. Behavioral error uses the actual normalized computation.
On-manifold accuracy cannot be inferred from the ambient tensor error in either
direction. Freeze candidates before evaluating unseen documents, distributions,
and interventions. Selective removal, restoration/interchange, and reuse need
their own controls; no tensor score establishes them.

## Exact reference and native screen

`joint_folded_tucker.py` implements symmetric projection, blockwise exact tensor
inner products, mode Grams, sparse-core truncation and denominator-aware
execution. Eight CPU tests compare against dense tensors, native RMSNorm,
factor rescalings, and the user's two-layer cancellation example.

`../bilinear_quotient/ops/run_joint_folded_tucker_weights_v615.py` runs through
the managed queue. Thin QR of U and E.T reduces output and input ambient modes
to 1152 without approximation. It compares joint HOSVD with independent SVD at
ranks 16/32/64, using full tensor reconstruction error, not an activation fit.
It reports lower bounds from the unfolded mode spectra, as well as support
budgets and prices. HOSVD is an initializer; no iterative optimizer or sparse
basis discovery is claimed. The result JSON and queue log are authoritative.

The first run gives rank-64 full-core relative error 0.972741 versus 0.984126
for independent SVD, at 3,794,944 versus 4,988,928 stored floating values.
Exact ambient-reduction replay error is 7.69e-7; mode-energy agreement is
1.81e-7. These establish a functioning instrument and poor low-rank fidelity,
not circuits. The completed follow-up gives rank-16/32/64 error lower bounds
for the symmetric shared-input Tucker family W G(P,P):
0.955650/0.932177/0.891714. Error <=0.10 requires output rank >=1088
and input rank >=1089 (necessary, not sufficient). Low-rank optimization
alone cannot solve this global tensor. This is a bound in the declared ambient
weight metric, not a bound on normalized-model behavioral fidelity.

## Next discovery step

Follow-up: the [native-atom frontier and spectral bound](NATIVE_ATOM_SPARSE_FRONTIER_2026-09-20.md)
also rule out sufficiently accurate pruning/scalar refitting in the fixed
native dictionary at the tested budgets. The next search must change features
or shared output structure rather than only select native terms.

Inspect the mode lower bounds before investing in low-rank optimization.
Compare output-sharing block terms and sparse rotations in larger retained
subspaces at matched error/cost. Avoid a large dense k³ core merely to reach
the required ranks: stream projected blocks or use implicit block terms.
Then jointly refactor selected two-layer compositions. Match the symmetrized
input difference, while allowing the stored coefficient representative to
remain unsymmetrized and compact. Requiring a fully symmetric representation
can inflate tree ranks for the same polynomial; see the user's
[HT and shared DAG refinement](HIERARCHICAL_TUCKER_SHARED_DAG_DIRECTION_2026-09-20.md).

Hierarchical Tucker is a tree representation, whereas a computation DAG can
reuse a node under multiple parents. Keep those as distinct candidate grammars.
The reference [Grasedyck hierarchical SVD paper](https://epubs.siam.org/doi/10.1137/090764189)
provides hierarchical rank truncation and error analysis; it does not supply a
causal-circuit or sparsity guarantee. For normalized paths, keep denominators
and attention operations as explicit graph nodes throughout.

The full OOD/extraction/removal/reuse/simplicity objective remains unachieved.
