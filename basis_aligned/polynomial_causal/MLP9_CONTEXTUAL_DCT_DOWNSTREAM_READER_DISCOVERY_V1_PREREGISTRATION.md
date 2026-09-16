# MLP9 contextual DCT downstream-reader discovery V1

## Question

The rank-16 contextual DCT package predicts MLP9 mixed responses on a second
fresh context panel, but no downstream behavior is known to read those
responses.  Use the already opened 64-prefix rank-16 panel only as a discovery
panel.  For each of the frozen 4×4 ordered input pairs, propagate both the exact
local MLP9 Hessian response and the packaged response through the exact native
block-10--17 suffix Jacobian to final-token logits.

This discovery may select one ordered pair and one positive/negative token
contrast.  It may not claim OOD behavior, selective removal, or a complete
circuit.  Any causal promotion must freeze the selected pair, tokens, scale,
and thresholds before opening new contexts.

## Predictions

- **A — instrument:** native MLP9 and suffix state replay are within `2e-6`.
- **B — downstream prediction:** packaged versus exact local-Hessian logit
  responses have at most `.02` aggregate relative L2 and at least `.999`
  cosine.
- **C — stable reader candidate:** the selected exact-logit contrast has at
  least `.90` sign agreement across all 64 prefixes and mean absolute effect at
  least `.005`.
- **D — package predicts the reader:** on the selected contrast, packaged
  versus exact responses have relative L2 at most `.03`, cosine at least
  `.999`, and identical aggregate sign.
- **E — no padded-token artifact:** both selected token ids are below the GPT-2
  tokenizer vocabulary size `50257`, distinct, and their decoded strings are
  serialized.

Passing is only a frozen-reader candidate for a prospective causal test.

## Price

One checkpoint load; 64 already opened score-blind prefixes; 16 exact local
Hessian and 16 packaged response tangents per prefix propagated in one batched
suffix JVP; no new text, finite intervention, fitting, parameter update, or
removal.
