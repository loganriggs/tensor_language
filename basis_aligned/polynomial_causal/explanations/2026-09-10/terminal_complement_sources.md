# Last-layer output versus incoming context — 18:50 UTC

**Both complementary sources are needed in the tested physical readout.** Replacing either one by its baseline value misses the registered 20% error limit. This tests the lexical side effect of the fixed grammatical command; it does not make that command selective.

Let $h=r+m$ be the final state: incoming residual $r$ plus last bilinear output $m$. With the fixed grammatical direction $e$, let $P=I-ee^\top$. We hold the edited final scalar $s_1=e^\top h_1$ and build

$$
h_{ij}=s_1e+Pr_i+Pm_j,\qquad i,j\in\{0,1\}.
$$

Subscript 0 denotes baseline and 1 denotes the grammatical edit. Every state gets its own physical final RMS normalization and score cap. The $10$ state also matches an actual native intervention restoring the last MLP's complementary output. Thus the source comparison has a native intervention check.

| Cell | Only MLP complement changed: $01$ | Only carried complement changed: $10$ |
|---|---:|---:|
| Frame 1, original verb | 71.02% | 34.22% |
| Frame 1, alternate verb | 80.32% | 36.85% |
| Frame 2, original verb | 82.12% | 37.28% |
| Frame 2, alternate verb | 83.33% | 36.26% |

These are relative errors in predicting the actual edit's two lexical-margin changes. They are not additive causal shares. Paired 95% intervals from 4,000 resamples of the sixteen rows also stay above 20% for both source-only predictions in every cell. They characterize these opened panels, not an OOD population.

The first run at18:47 failed an instrument check comparing the real-arithmetic folded reader directly with FP32 native output after a scalar edit. It remains invalid. The repaired18:50 run separates the FP64 contraction identity, exact captured native Down-plus-bias execution, and measured native arithmetic/scalar-cast differences. No scientific arm or .20 bar changed. This is a post-outcome instrument repair, not independent confirmation. Shared saved states are bitwise equal across the two runs. Native reader rounding accounts for up to .00390 in unnormalized reader units; the additional scalar-cast contribution is at most .000344. These are not final token-score errors.

Valid run:16 native forwards,256 sequences,1.42758 measured executor seconds. Full-state/readout and actual-clamp bridges held. CPU paired/replay/rounding audit completed in .0464seconds without a checkpoint. Both strong source-sufficiency hypotheses failed. The last layer is not the sole producer of the contextual information required by the command.

The user's next direction is the **entire unembedding composed with the bilinear layer**, retaining signed shared subterms. This is distinct from the earlier518-token search for proportional whole functions. The present null does not rule out a useful weight-derived shared product decomposition.

Receipts: [V2 result](../../TERMINAL_COMPLEMENT_SOURCE_V2_RESULT.json), [CPU audit](../../TERMINAL_COMPLEMENT_SOURCE_V2_AUDIT_RESULT.json), [preserved invalid V1](../../TERMINAL_COMPLEMENT_SOURCE_V1_RESULT.json).
