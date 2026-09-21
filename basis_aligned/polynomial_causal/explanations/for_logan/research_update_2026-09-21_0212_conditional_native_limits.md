# Native conditional interventions expose a substantial remaining gap

21 September 2026, 02:12 UTC.

The shared correction improves the interaction it was designed to target, but that interaction remains poorly reconstructed on the native diagnostic panels. The corrected program has **55.8% FineWeb and40.1% code context-only effect error**. Whole-contribution swap results had understated this limitation. Final normalization and softcapping do not cause the gap: the corresponding linear contribution errors are even larger.

## Interventions and scope

The folded function is bilinear in normalized midpoint input $n$ and source input $m$. We use the existing same-token, cross-document donor mapping, take $\Delta m=m_{\mathrm{donor}}-m_{\mathrm{recipient}}$, and test:

$$
\Delta F_{\mathrm{source}}=F(n,\Delta m),
$$

$$
\Delta F_{\mathrm{context}}=F(n-\bar n,\Delta m),
$$

where $\bar n$ is the fixed calibration mean. Each exact or approximate write is inserted separately into the same native final residual state, followed by native final RMS normalization and logit softcap. We compare resulting vocabulary-centered logit changes.

Holding $n$ fixed means holding the midpoint interface input fixed. It is not the same as physically replacing an upstream source, recomputing attention/normalization, or holding every other original residual contribution fixed. Likewise, although source and context components have a linear decomposition before final operations, their final-logit effects need not add.

A CPU check compares the exported graph's direct differences against the new source/context evaluation helper, with relative errors below $1.3\times10^{-14}$. The previous native intervention families reproduce their earlier results within the registered tolerance. No fitting occurs on these reused32 FineWeb and16 related-code documents.

## Native effect errors

| Intervention and domain | Existing512-product graph | Add32 shared correction features |
|---|---:|---:|
| Source-only, FineWeb | 27.72% | 27.23% |
| Source-only, code | 23.58% | 22.69% |
| Context-only, FineWeb | 57.12% | 55.82% |
| Context-only, code | 42.13% | 40.10% |

The registered predictions—instrument replay, context improvement in both domains, and source-effect preservation—pass. Those were relative tests. They do not make the large absolute context errors acceptable for an identified, accurately reusable circuit.

## Is the gap caused by downstream nonlinearities?

A successor run measures the same interventions before and after final native operations, in the same vocabulary-centered norm:

| Corrected program | Before final operations | After final operations |
|---|---:|---:|
| Source-only, FineWeb | 29.53% | 27.23% |
| Source-only, code | 23.23% | 22.69% |
| Context-only, FineWeb | 58.78% | 55.82% |
| Context-only, code | 42.04% | 40.10% |

The nonlinear operations reduce these relative errors modestly. The substantial miss is already in the folded approximation under this conditional test.

For the corrected context effect, predicted-to-native norm ratios are0.834 on FineWeb and0.913 on code; pooled cosines are0.830 and0.916. Thus this is not merely a small overall amplitude mismatch. These pooled geometric measures are not evidence of semantic alignment.

Earlier CPU context-interaction errors around31–32% used the full uncompressed correction on calibration inputs and cyclic donors. The present test changes the correction rank, input panel, donor rule and context length. Those earlier numbers cannot isolate which change caused the difference. The new matched before/after measurement does rule out final nonlinearities as the cause of the high errors in this test.

The next research decision should address this conditional approximation gap and its input support, rather than continue polishing aggregate replacement loss. We still need stable intermediate identities, behavior-specific interventions, extraction under stated interfaces, composition and genuinely broader OOD evidence. The complete circuit-discovery objective remains open.

Evidence: `MIDPOINT_SOURCE_CONTEXT_NATIVE_V1.json`, `MIDPOINT_SOURCE_CONTEXT_NATIVE_V2.json`, `MIDPOINT_SOURCE_CONTEXT_LINEAR_AUDIT_V1.json`, and their raw per-document records under `direct_tensor_match`. Shared helper: `midpoint_program.py`; audit reproduction: `audit_source_context_linear.py`.
