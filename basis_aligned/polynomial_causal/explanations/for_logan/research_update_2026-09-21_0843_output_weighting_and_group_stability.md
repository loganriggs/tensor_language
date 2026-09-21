# Output weighting makes groups repeatable but does not preserve accuracy

21 September 2026. This is a follow-up experiment, not a replacement for the [overall review](research_update_2026-09-21_0104_full_coverage_and_shared_baselines.md).

We tested whether weak output combinations were being neglected by coefficient fitting. Increasing their weight made four groups of quadratic computations much more repeatable across restarts, but worsened accuracy on the original model components. The result is a tradeoff, not circuit adoption.

Let T contain four quadratic source reads and let its output Gram matrix be K = T_flat T_flat^T. We fit B T with an invertible output transform B and undo that transform before executing the fitted program:

$$
K=U\operatorname{diag}(e)U^\top,\qquad
B=U\operatorname{diag}(e^{-p/2})U^\top.
$$

Here p=0 is the earlier coefficient fit, p=0.5 partially balances output directions, and p=1 gives equal relative weight to all four orthogonal output combinations. The input covariance metric is unchanged. These are different losses; their percentages are not directly comparable.

| Measurement | Partial balance | Full balance |
|---|---:|---:|
| Original coefficient error | 9.58% | 13.76% |
| First two component value errors | 2.88%, 2.68% | 5.14%, 3.59% |
| Weakest original-feature cosine | 0.944 | 0.987 |
| Weakest feature relative reconstruction error | 33.09% | 15.83% |
| Minimum group cosine across restarts/rates | 0.979 | 0.996 |

Full balance passes the 0.99 restart-agreement diagnostic but fails the original-feature 0.99 fidelity gate and the relative component-value gate. Partial balance retains the value gate but fails group fidelity. Neither improves both conditional component Jacobians by the registered 10%. All arms keep the third component unchanged; its known fresh-panel failure is not repaired.

Each power used Adam rates0.02/0.05, seeds816/1816 and2000cosine steps per fit. Input directions start randomly; readout coefficients are solved analytically. All programs have512products and897804floating coefficients. Native z and h remain supplied inputs; diagnostics reuse448opened sites. There were no native-model forwards or new OOD panels.

Independent exports, dense reconstruction and centered affine/mean replay pass below1e-8 for all eight programs. The private third branch is unchanged. Five synthetic loss/gradient/invertible-output checks passed; those are metric checks, not five new recovery runs. At fixed product directions, inverting the output transform leaves the optimal readout unchanged: improvements must come from moving the feature directions.

An all-six-output audit reveals another weak combination involving the third component. It carries0.0307%of total coefficient energy, yet the current approximation has112.95%relative error on that feature. Fitting only the first four outputs cannot address this. A global383-product graph has896262coefficients, below the partial graph's storage. Its exporter and executor have passed a CPU preflight; its fit is a next comparison, not a result of this report.

Evidence: [partial fit](../../direct_tensor_match/SOURCE_OUTPUT_BALANCE_HALF_V1.json), [full fit](../../direct_tensor_match/SOURCE_OUTPUT_BALANCE_FULL_V1.json), [independent audit](../../direct_tensor_match/OUTPUT_BALANCE_PROGRAM_AUDIT_V1.json), [all-six contrasts](../../direct_tensor_match/ALL_SOURCE_CONTRASTS_V1.json), and [global export preflight](../../direct_tensor_match/GLOBAL_SOURCE_EXPORT_PREFLIGHT_V1.json).
