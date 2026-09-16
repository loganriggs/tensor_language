# Equality L5H5 M4 corrected shared-kernel behavior V1

The natural-selected K2 correction passes score fidelity with 13 instead of 16
Q/K projections.  This prospectively tests whether it preserves the explicit
interaction's behavioral removal and composition certificates.

Frozen code arms: native, late edge absent, four-score authority additive,
four-score authority direct, shared bias-only baseline, shared additive
(interaction removed), and shared composed/direct.  No reselection occurs.

Gates:

1. Inherited authority and corrected shared score receipts remain lawful.
2. Shared composed NLL replays authority direct within `.05` of the authority
   joint-vs-shared-baseline effect on copy-positive, every subtype, and both
   halves.
3. Shared interaction-removal NLL effect matches authority removal within `.20`
   relative L2 and cosine at least `.90` on the same cells/halves.
4. Shared removal magnitude remains at least `.20` of the joint-vs-baseline
   effect in every cell/half, with incremental noncopy mean change at most `.01`
   nat.
5. Shared composed recovery is in `[.80,1.05]`, differs from authority direct by
   at most `.02`, every donor term is nonzero, and the executor retains 13 Q/K
   projections and zero learned parameters.

Failure after lawful inherited receipts is a behavioral compression null; it
does not overturn the exact 16-projection interaction graph.

Price: one checkpoint load, 192 frozen code documents, seven behavior arms, no
natural rows, fits, gradients, parameter updates, or new text.
