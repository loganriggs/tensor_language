# Equality L5H5 M4 contracted-mode graph V1

This is the prospective positive red-team of the native-product sparsity null.
That instrument replayed exactly, but required 4,096/4,608 native product atoms.
The null may reflect a poor coordinate basis rather than distributed causal
structure.

Let `W` stack the four frozen 128x1152 L5H5 Q/K matrices and let `D` be M4's
1152x4608 Down matrix.  Compute the weight-only SVD

`W D = U diag(s) V^T`.

For ranks `0,1,2,4,8,16,32,64,128,256,512`, project the native M4 product
vector onto the leading rows of `V`, map those coefficients through `D V`, add
the exact native bias, and use the resulting write in the already-frozen
five-write score graph.  These are dense bilinear product modes: a rank-r node
has r scalar quadratic readers and r writer directions, but currently evaluates
all 4,608 native products.  It is writer/mode compression, not yet product-cost
compression.

Frozen gates:

1. **Lawful bridge.** The inherited native-product instrument has exact write
   and score replay, and all 512 SVD modes reconstruct `W D` within `2e-5`
   relative L2.
2. **Compact extraction.** The smallest natural rank with five-write
   parent-score error at most `.10` and cosine at least `.995` is at most 64.
3. **OOD prediction.** Frozen code parent-score error is at most `.15`, cosine
   at least `.98`, and both half errors are at most `.20`.
4. **Causal use.** Code donor recovery is in `[.80,1.05]`, within `.08` of the
   five-write parent, with all registered copy cells and halves above `.55`.
5. **Selective direction.** Noncopy damage is at most `.01` nat, while a
   one-token roll of the bias-free reconstructed mode write (equal norm) lowers
   copy-positive recovery by at least `.05`.
6. **Composition.** Alternating frozen singular modes form two components whose
   separately measured score effects predict the joint effect within `.10`
   component-relative L2 on code OOD.

Failure of gate 1 makes the receipt invalid.  Other failures are valid nulls.
No natural activation, behavior label, code row, logit, or fitted parameter
defines the basis.  M4's input state and native L/R product computation remain
external/native costs and are counted explicitly.

Price: one checkpoint load, one natural geometry pass, one code geometry pass,
and the same six code behavior arms as the native-product experiment; no
gradients, parameter updates, or new text.
