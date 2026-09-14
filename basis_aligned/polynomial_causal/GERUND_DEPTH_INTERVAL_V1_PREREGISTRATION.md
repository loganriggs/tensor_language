# Gerund cumulative depth-interval interchange

14 September 2026. The gerund token/norm program requires the actual-token
MLP17 context gate and final-state RMS moment. The prior all-36 singleton output
screen found no source above .371 transfer, while attention7, attention11,
MLP11 and MLP16 recur near the top in both target frames. The carried terminal
complement is also closer than MLP17 alone. This test therefore changes the
operator: it interchanges fixed contiguous depth intervals of complete native
module outputs at the semantic position. It does not search a new singleton or
head subset.

For each A1, A2 and G row, the cyclic donor is the next row in the same panel.
The final token is identical within every cyclic pair. Cache the 36 native
attention/MLP outputs, then patch the recipient with the donor outputs for these
fixed arms:

* cumulative prefixes through blocks 7, 11, 13 and 16;
* the complete causal prefix through attention17 (`pre_mlp17`);
* contiguous suffix bands from blocks 7, 8, 11 or 14 through block16, each also
  including attention17; and
* attention17 alone as a direct-input sentinel.

All later modules recompute normally. MLP17 is never patched. The fixed reader
is the existing answer-minus-foil context reader
`k_perp = 2 Q17(v)e - 2(e^T Q17(v)e)e`. Report aggregate transfer and relative
error for its gate `tau = k_perp^T u17`. Also report transfer and error for the
final pre-RMS second moment `mean(h^2)`, because the executable token-score
program needs both numerator and normalization state. Aggregate norms avoid
rowwise division by small references.

Predictions, frozen before native scoring:

* **A — instrument.** Exactly 36 bodies/576 sequences and 1,296 module-hook
  visits; all 48 native answer endpoints beat their foils; cyclic gate and norm
  reference norms exceed 1e-4; own-cache no-op logits have max absolute error
  <=1e-3 and relative L2 <=1e-5; `pre_mlp17` reproduces donor MLP17 input and
  final state with the same tolerances. Parent input hashes must match.
* **B — distributed target band.** `band7_pre17` has gate transfer >=.75 and
  relative gate error <=.50 on both A1 and A2.
* **C — cumulative onset.** `prefix7` gate transfer is <=.50, while `prefix11`
  is >=.50 and `prefix13` is >=.65 on both A1 and A2. This opposing prediction
  distinguishes mid-depth buildup from an already-established early state.
* **D — norm-state sufficiency.** `band7_pre17` has final norm-moment transfer
  >=.50 and relative error <=.75 on both A1 and A2.
* **E — collateral control.** `band7_pre17` changes correct-token CE on G by
  mean absolute <=.15 nats.

The conditional port price counts a full 1,152-vector for every patched module
and row. `pre_mlp17` uses 35 upstream module outputs; `band7_pre17` uses 21, a
40% reduction in this intervention interface. Native weights, donor-state
generation, every token position, MLP17 and the suffix remain charged external
dependencies. A pass is a depth-band producer/interface result on these opened
frames, not a static model compression, semantic grammar label, natural-text
OOD result or independent circuit extraction. Failure closes this fixed band;
do not rescue it with a learned boundary, head mask, rank sweep or denominator
change.
