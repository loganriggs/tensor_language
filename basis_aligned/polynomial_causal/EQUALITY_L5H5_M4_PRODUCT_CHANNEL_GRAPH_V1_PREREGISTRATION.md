# Equality L5H5 M4 native-product graph V1

This prospectively refines the retained `M4` port in the canonical five-write
boundary.  M4 is represented by its 4,608 exact native bilinear atoms

`D[:,i] * (L[i] z) * (R[i] z)`

plus the native bias.  No behavioral labels, logits, or code-OOD rows select
atoms.  On 192 natural documents, atom `i` is ranked by the RMS of its native
product activation on equality-edge incident positions times the norm of its
writer after contraction with the four frozen L5H5 Q/K matrices.  This is a
one-layer, open-context DCT moment score; it proposes atoms but does not certify
them.  Candidate widths are `0,16,32,64,128,256,512,1024,2048,4096,4608`.

Frozen predictions and gates:

1. **Lawful instrument.** Direct execution of all 4,608 atoms in native order,
   including bias and layer-5 residual scaling, reproduces the captured M4
   write and five-write parent score within `2e-6` relative error.
2. **Compact extraction.** The smallest natural width with parent-score error
   at most `.10` and cosine at least `.995` uses at most 512 atoms.
3. **OOD prediction.** With no reselection, code parent-score error is at most
   `.15`, cosine at least `.98`, and both 96-document half errors are at most
   `.20`.
4. **Causal use.** Frozen code-OOD donor recovery is in `[.80,1.05]`, differs
   from the five-write parent by at most `.08`, and every registered copy cell
   and half exceeds `.55` recovery.
5. **Selective direction.** Noncopy mean damage is at most `.01` nat.  Rolling
   the selected bias-free M4 product write by one token preserves its norm but
   reduces copy-positive recovery by at least `.05`.
6. **Composition.** Splitting selected atoms by alternating frozen rank into
   two components, separately measured score effects predict their joint score
   effect with at most `.10` component-relative L2 error on code OOD.

If gate 1 fails, the receipt is invalid and no scientific null is interpreted.
If gate 1 passes but compactness, transfer, causality, specificity, or
composition fails, preserve the corresponding valid null.  In particular,
full-width success cannot be relabeled compact.  Bias is always retained as a
typed constant; selected native M4 input state, L/R rows, D columns, and bias
are declared ports/weights rather than learned parameters.

Price: one checkpoint load; two natural prefix passes; one code geometry pass;
192-document code behavior with native, absent, exact-score, five-write,
selected-product, and rolled-product arms; no gradients, parameter updates, or
new text.
