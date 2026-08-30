# Previous-token fixed-tensor discovery findings

**Executed:** 2026-08-30 04:18 UTC

**Runtime:** 77.79 seconds on a shared GPU

**Receipt:** `previous_token_tensor_discovery.json`

**SHA-256:** `0853acf3a339ae470131c8f7f30f6859b65f9420a43ba106d71364ae9864f588`

## Result in plain language

The previous-position component of layer-0 head 3 is an excellent **extractable
primitive**, including on token bigrams absent from the FIT documents. It is not a
selectively removable “previous-top behavior circuit.”

The candidate uses the fixed shift tensor

$$
S_{qk}=\mathbf 1[k=q-1]
$$

inside the head's exact quadratic attention pattern. It has no TopK, argmax, parser,
or input-dependent router. `Argmax` was used only to label evaluation positions.

## What passed

- Recomputing all of layer-0 attention from frozen weights reproduced native logits
  bit-for-bit: maximum logit error and KL were both zero.
- Every analytical arm made zero calls to native layer-0 attention and its Q/K/Q2/K2/V/O
  submodules. Native made the expected 24 calls to each.
- Restoring only the previous-position tensor in an L0H3-deleted background recovered
  **94.21%** of the head's CE effect on previous-top positions, with document-bootstrap
  95% interval **[91.33%, 97.07%]**.
- Recovery on 9,262 unseen-bigram positions was **94.17%**, versus **94.42%** on 3,422
  seen-bigram positions.
- The fixed shift-minus-two null recovered only **15.29%**; the causally masked
  shift-plus-two null recovered zero.
- Every named cell had at least 710 tokens across at least 95 documents.

## What failed

Selective removal failed. Removing the previous-position tensor changed CE by
**+0.06249 nat** on previous-top positions, but also about **+0.06321 nat** on self-top
positions. Their difference was -0.00071 nat with 95% interval
**[-0.01311, +0.01156]**. Thus the fixed component is important even when it is not the
largest-magnitude attention edge.

Global removal cost was also **+0.06191 nat**, with 95% interval
**[+0.05503, +0.06897]**, far above the preregistered 0.01-nat collateral ceiling.

The candidate therefore failed 2 of 9 scientific gates: target-minus-self specificity
and global collateral. It is not eligible for a fresh terminal run under this behavior
definition, and the behavior definition will not be changed after seeing the result.

## Interpretation

This cleanly separates three desirable properties that had been conflated:

1. **Algebraic/executable extraction:** strong pass.
2. **Unseen-input transport:** strong pass for unseen bigrams.
3. **Selective behavioral removal:** fail.

The fixed previous-token path is a broad computational service used across many
positions, not a narrow circuit that activates only when its attention edge is the
largest. It remains a shared primitive for induction, copied entities, and numeric
formatting, but its parameters and causal effect must be counted once and its removal
cannot be advertised as behavior-specific.

The campaign should next target a mechanism with a naturally selective tensor—ordered
successor or bracket closure—rather than trying to rescue this result by post-hoc
gating. An external “remove only when argmax says previous” condition would itself be
a discrete router and would no longer test the fixed tensor circuit.
