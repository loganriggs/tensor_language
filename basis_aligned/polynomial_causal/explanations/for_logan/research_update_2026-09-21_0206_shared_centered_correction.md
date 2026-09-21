# A compact context correction: a small native tradeoff

21 September 2026, 02:06 UTC.

We compressed the [centered-operator correction](research_update_2026-09-21_0205_centered_operator.md) into shared linear features and tested it inside the model. The 32-feature version slightly improves swap-effect fidelity, but increases replacement loss and storage. It is an optional tradeoff, not a clear replacement for the existing graph.

## Executable representation and price

The graph keeps its existing512 products. A centered product can be reconstructed from those same products and linear terms:

$$
(a-\bar a)(b-\bar b)=ab-a\bar b-\bar a b+\bar a\bar b.
$$

The correction reads32 linear combinations of these centered products and writes their output effects. We verify its algebra against a direct centered evaluation, and verify the full-rank factorization against the dense correction. The existing program also replays identically in the extended native executor.

At rank32, weight coefficients rise from1,291,264 to1,344,512: **4.1% extra**, rather than the dense compiled correction's2,031,616. There are also1,024 additional stored input means. Constant multiplications and linear sums remain real costs; keeping512 variable-by-variable products is not a whole-program speed claim.

The rank32 approximation leaves69.2% of the dense correction's weighted norm unexplained. This is error relative to the correction, not relative to the whole model function. The correction is not extremely low-rank under this metric.

## Native diagnostic results

Same reused32 FineWeb and16 related-code documents; no fitting on these panels. Native final normalization and softcap remain active.

| Metric | Existing512 graph | Add32 correction features |
|---|---:|---:|
| FineWeb replacement CE added | 0.01098 | 0.01126 |
| Code replacement CE added | 0.02758 | 0.02867 |
| FineWeb removal-effect error | 23.74% | 23.73% |
| Code removal-effect error | 16.14% | 15.88% |
| FineWeb same-token swap-effect error | 31.66% | 31.45% |
| Code same-token swap-effect error | 27.57% | 27.00% |

Registered instrument, relative-preservation and CE-threshold checks pass. Those checks establish preservation, not improvement. FineWeb swap error still exceeds the older absolute30% bar on this reused panel.

A successor paired10,000-document bootstrap gives swap-error ratio intervals of0.9926–0.9944 on FineWeb and0.9777–0.9813 on code. Thus the small swap improvements are consistent across this panel. The CE-difference intervals cross zero in both domains; the point estimates worsen, but the panel does not resolve their population direction. These are descriptive intervals on repeatedly used documents, not fresh confirmation or familywise guarantees.

The8-feature version was also tested; it offers smaller swap improvements and larger replacement-loss increases on this panel. No intermediate feature has been assigned a semantic identity.

This establishes that a small shared graph correction can carry some of the centered objective's benefit through native nonlinear operations. Next, test the specific source-only/context interaction in native logits: whole-contribution swaps can still conceal which part of the computation improved.

Evidence: `MIDPOINT_CENTERED_CORRECTION_V1.json`, `MIDPOINT_CENTERED_CORRECTION_NATIVE_V1.json`, `MIDPOINT_CENTERED_CORRECTION_BOOTSTRAP_V1.json`; exported executable graphs in `MIDPOINT_CENTERED_CORRECTION_GRAPHS_V1.pt` under `direct_tensor_match`.
