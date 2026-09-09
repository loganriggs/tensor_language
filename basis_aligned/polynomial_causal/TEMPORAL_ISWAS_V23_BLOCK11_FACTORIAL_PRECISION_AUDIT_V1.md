# Block-11 factorial precision audit v1

The original v23 block-11 residual/MLP factorial remains `invalid_instrument`: prediction D required
the maximum of block-11 output, final hidden state, and logits restoration errors to be at most
`1e-4`, while the final-hidden maximum was `0.03125`. This audit does not relabel that prediction.

The planned downstream-reader atlas needs a narrower interface: it reconstructs the residual
correction at the block-11 output, then independently replays the residual branch and validates every
reader intervention. For that successor only, the necessary predecessor conditions are:

- original authority/hooks/closure/price prediction A passed;
- residual-dominance prediction B passed;
- component closure and self-patch error are at most `1e-4`;
- joint block-11 output restoration is at most `1e-4`;
- joint output-logit restoration is at most `1e-4`;
- joint signed behavioral projection is at least `0.99`;
- no component or threshold was selected after observing the outcome.

The final-hidden diagnostic is retained verbatim and explicitly fails its original bar. It is not
used to license the reader atlas because it is outside that atlas's installed block-11 correction
interface and the full logits already close at `1.28746e-5`. The successor remains fail-closed on
its own residual replay (`1e-4`), hook counts, finiteness, FIT/HOLDOUT split, reciprocal transfer and
reset, and P/C controls. Thus this is a scoped downstream experiment license, not retroactive
validation of the factorial and not evidence that any reader exists.

