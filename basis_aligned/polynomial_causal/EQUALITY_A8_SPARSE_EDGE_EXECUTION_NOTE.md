# Equality A8 sparse-edge execution and red-team note

The pre-MLP9 three-edge factorial passed all registered predictions.  On the
opened natural panel, the attention8 write alone (`100`) recovered `.9268297`
of the copy-token removal effect.  MLP8-only recovered `.1423781`,
attention9-only `.0233874`, and every pair/third-order recovery interaction was
below `.029` absolute.  Exact native/absent corner and write replay errors were
zero.  This selects a one-edge oracle support; it does not by itself extract
the edge.

Frozen transfer to the separately sealed 192-document code-OOD role was a
valid calibration null, not an instrument failure.  A8 remained dominant,
positive in every cell, and stable across halves (`1.17405/1.18067`), with
`.00660` nat noncopy mean damage.  Aggregate recovery was `1.17756`, however,
versus `.92683` on natural text, missing the registered absolute-drift ceiling
of `.15`.  MLP8-only and attention9-only were negative on code OOD.  Thus edge
identity and sign transfer, but a natural-text scalar does not calibrate its
code-OOD behavioral magnitude.

Two zero-new-parameter extraction implementations then separated algebra from
deployed arithmetic:

1. V1 computed `bmm(score * support, projected_payload)`.  Its bilinear
   composition identity passed (`1.68e-7`), and installed recovery nearly
   matched the oracle (`1.17664`), but projecting before summing changed BF16
   operation order.  Term error was `.002368` and logit replay error about
   `.016--.017`; V1 is therefore a valid projected-payload executor null.
2. V2 computes the deployed order exactly: contract the raw head payload, then
   apply the native output-projection slice.  Term error, removal-logit error,
   and removal-MLP9 error are all exactly zero; its FP32 bilinear composition
   error is `2.87e-7`.  Reinstallation is not bit-exact because BF16 subtraction
   is not invertible: `(full - term) + term` need not equal `full`.  Installed
   recovery is still `1.17494` versus oracle `1.17756`, but logit relative error
   `.01582` fails the frozen `2e-6` exact-install gate.  V2 is consequently also
   a valid null at the stronger exact bidirectional-executor claim.

The retained result is precise: L8H4 now has a sparse, zero-new-parameter,
compositionally reusable edge executor with exact removal and close behavioral
reinstallation on the tested code-OOD contexts.  Exact reinsertion needs either a
canonical graph arithmetic boundary or an explicit rounding/remainder port.
OOD scalar calibration also remains unsolved.  Neither issue licenses changing
the failed thresholds after seeing outcomes.

V3 resolves the reinsertion issue without changing the semantic edge or either
null.  Its graph boundary exposes a separately typed FP32 roundoff-correction
port returned by `split(full, term)` and consumed by `merge`.  On all code-OOD
documents, semantic term, split removal, merge, removal logits/MLP9, and
installed logits/MLP9 all have exactly zero measured replay error.  Canonical
recovery equals the oracle exactly at `1.1775565`.  The correction is nonzero in
`9.43%` of elements but carries only `.001215` of full-write norm and `.002712`
of semantic-term norm.  This is an exact reversible sparse-graph boundary with
zero new learned parameters; the implementation port is not evidence for an
additional semantic circuit edge.

The next upstream port also transfers.  Replacing the L8H4 score by the frozen
natural-scale L5H5 score inside the exact-order node recovers `.97287` on code
OOD, compared with `.96873` for the older projected-payload arithmetic.  The
preregistered L7H3 wrong-score donor is strongly harmful (`-1.84448` recovery),
the target is stable across halves (`.99883/.94979`), and noncopy mean damage is
`.00240` nat.  The target score cosine on equality edges is `.82849`.  This is
exported as a one-scalar, zero-new-parameter adapter.  The adapter is a reusable
graph edge; production of the L5H5 score from residual/QK inputs remains native.

The opaque L5H5 score port is now split into its actual zero-parameter
multiplicative node: four post-projection, post-RMS, post-rotary ports compute
`causal_mask(dot(q1,k1)/128 * dot(q2,k2)/128)`.  On all 192 code-OOD documents
this reconstruction has exactly zero score and donor-logit error relative to
factor capture, preserves `.9728678` recovery, and expands into the registered
`2x2` half-coordinate product grid at `1.07e-7` relative error.  The initial V1
receipt is deliberately retained as a red-team null: it omitted the causal mask,
reported `1.0932` score error, yet had zero logit error because the erroneous
future-key entries were later excluded by induction support.  The corrected V2
therefore converts that apparent negative result into a verified extracted
node.  The remaining upstream boundary is residual-to-Q/K projection followed
by normalization/rotation; L8H4 raw-payload production is still a separate
native port.

That boundary is now moved one step farther upstream.  A zero-learned-parameter
executor takes the single pre-L5-attention residual, applies the four frozen
L5H5 `128x1152` Q/K weight slices, retains both RMS operations and rotation,
and reproduces the native score and donor logits with exactly zero measured
error.  A natural-only exhaustive screen over the exact additive provenance
groups `E,L0,...,L4` selected `L2+L3+L4` (three of six): equality-edge score
error/cosine was `.14462/.99197`.  Frozen on code, it reached `.15268/.99233`,
recovered `.90458` of the removal stake, stayed at `.00172` nat noncopy damage,
and had `3.94e-9` complete Möbius-graph closure.  Removing the `.00365`-norm
roundoff port selected the same support, slightly improved score error to
`.15255`, and changed recovery by only `-.00020`; an equal-norm position roll
changed recovery by `.00287`.  Thus the three-source graph is not an artifact
of the correction direction.  Its current ports remain the native frozen
layer-2/3/4 writes: recursively extracting their producers is the next boundary,
not another L5 score fit.

The first recursive split distinguishes scientific sparsity from implementation
failure.  V1 flattened the six writes and changed BF16 addition order; its
all-six replay error was `.0004195`, so the exactness gate correctly marked it
invalid.  V2 preserves `(A+M)` within each layer and then adds layers 2, 3, and
4 in parent order, giving exactly zero all-six replay error.  Under that lawful
arithmetic the frozen natural rule retains `M2+A3+M3+A4+M4`: natural
parent-score error/cosine is `.05038/.99925`, code is `.03875/.99934`, recovery
is `.88553`, noncopy damage is `.00132` nat, and the complete interaction graph
closes at `9.60e-9`.  Because five writes survive, the preregistered claim of at
most four is a valid null.  The five-write executor is retained as a useful
boundary, not relabeled as a sparse positive.  `A2` is the only safely omitted
module write at this resolution; recursive extraction should next split a
retained producer using downstream score and fresh causal transfer rather than
source norm alone.

That recursive M4 test now separates basis sparsity from interaction sparsity.
Exact native product atoms required 4,096/4,608 channels, despite `.02914` code
parent-score error and `.89166` causal recovery.  A weight-only SVD of the M4
Down map contracted with all four L5H5 Q/K readers reduced the selected boundary
to rank 256.  After rejecting a float32 SVD receipt (`1.26e-4` bridge error), a
float64 correction certified the operator at `2.06e-13`; the deployed node has
`.03818/.99927` code error/cosine, `.89159` recovery, `.00155` nat noncopy
damage, and a `.18060` recovery advantage over its equal-norm position roll.
This is useful mode compression but fails the prospective rank-64 compactness
bar and retains all 4,608 native product evaluations.  A natural-only
most-additive split improved score composition from `.49047` to `.13323` OOD,
but direct behavioral composition remained `.42108`.  Preserve those nulls and
make the child/remainder cross-difference an explicit graph node next; do not
force an additive interpretation.

The explicit interaction graph resolves that representational failure.  Above
the bias-only baseline it stores the rank-128 child effect, rank-128 remainder
effect, and their Möbius cross-difference.  Natural and code graph closure and
downstream composed replay are exactly zero-error.  Removing only the
interaction changes copy-token NLL by `.30096` of the joint-vs-baseline effect
overall and `.26961–.32079` in every subtype/half, with `-.000345` nat noncopy
mean change.  A one-query roll preserves interaction norm within `1.17e-7` but
has only `.52158` behavioral-effect cosine to true removal.  Direct and composed
recovery both equal `.89159`.  This is a zero-learned-parameter four-trait OOD
graph at the declared boundary; its live efficiency handoff is replacing the
current four-score cross-difference executor with a cheaper direct kernel.

That cheaper-kernel attempt provides a sharp caution.  A 12-projection shared
executor has only `.00376` joint-score error but `.10063` interaction error;
changing the arithmetic gauge does not improve it (`.10069`).  One selected K2
correction produces a 13-projection executor that passes the score gate at
`.09956` and nearly preserves aggregate recovery (`.89071` versus `.89159`).
It still fails behavioral equivalence: tokenwise replay error is `.258`, and
its interaction-removal vector has `.294` cosine and `1.198` relative error to
the exact graph.  Keep the exact 16-projection interaction node as the
behavioral authority.  A future fold must target the removal vector directly;
global score error and aggregate recovery are insufficient certificates.

A behavior-selected native-order follow-up closes the miscoding loophole more
strongly.  It tested all 15 proper subsets of native child `Q1,K1,Q2,K2`
projections on natural data, with the full four-map path as a positive control.
The control has zero replay and removal-vector error in every natural/code cell.
No proper subset passes.  Best diagnostic `Q1+Q2` has `.846` code removal-vector
error and `.650` cosine despite exactly matching authority recovery and composed
replay.  Therefore the four child ports are conjunctive at this boundary;
replay closure cannot certify the interaction-removal counterfactual.  Continue
with a four-map Möbius decomposition rather than treating any raw map as
independently removable.
