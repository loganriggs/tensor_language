# Exact upstream port closure for the coupled value operator

## Motivation

`city_mlp8_coupled_value_v1` is installed and behaviorally replayed, but still takes native `z8` and the upstream attention8 `delta`. The existing attention7/head8 package is not a valid closure: it approximates RMS8, leaves native prefix state external, and its FineWeb removal direction is only 88/102 positive. Do not compose that approximation into the coupled operator or borrow its Pile selectivity result.

## Frozen target

Build an exact residual6-to-attention8 upstream generator using the complete bilinear attention7 readers and head8 product, with the edited RMS8 generated from the same native factors. Keep the coupled value operator unchanged. The generated `delta` must be compared to an independently captured native attention8 city intervention before any suffix evaluation.

## Gates

1. **Prediction:** fresh 20-document FineWeb panel, 40 sequences, 240 probes; per-sequence upstream delta replay error at most `0.05` against the independent native intervention.
2. **Extraction:** residual6, token IDs, city, and destination are the only declared upstream inputs; installed and isolated execution receipts required. Report every native vector/scalar port and stored value.
3. **Selectivity:** full coupled operator plus generated upstream delta must preserve the fresh direction gate and beat 16 same-boundary random equal-norm corrections; at least three unrelated readers stay below `.5` of target.
4. **Composition:** test native upstream generator + coupled value operator with the five-arm Möbius battery. Require global and every nonempty native-reversed subgroup interaction at most `.35` of the smallest piece. Empty subgroup is untested, not a pass.

## Interpretation

The current approximation is a retained falsified route. A successful upstream fold would close one native port; it would not establish token-only execution or simplicity. A failure retains the coupled operator as the best declared-boundary object and leaves the port explicit.
