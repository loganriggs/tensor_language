Joint feature learning under coefficient and shifted-Gaussian matching

Target remains the original MLP16→MLP17 pure quartic branch in16fixed data-informed output readouts. The candidate remains512quartic CP terms,1536variable products and2385920stored float coefficients. Both archived coefficient-fit starts1001/1002 are used. No extra features, task labels, connection changes or output directions.

For seed s, fix lambda_s to the Lagrange multiplier of the preceding registered budget1 readout solution: approximately789.24 and1031.49. Those multipliers depend only on native weights and calibration input mean/covariance, not evaluation labels. Optimize

$$
L_s(\theta,C)=\frac{L_G(\theta,C)+\lambda_s L_F(\theta,C)}{1+\lambda_s}+10^{-6}\|C\|_F^2.
$$

L_G is exact function-squared error under N(calibration mean,calibration centered covariance); L_F is exact symmetrized coefficient-squared error. Constant teacher norms are omitted. Re-solve C exactly each step and backpropagate the envelope gradient into four learned input-factor matrices, normalized rowwise. Lambda stays fixed as features change. **This is a fixed mixed objective, not a maintained coefficient-deterioration constraint and not a convex optimization over directions.** Report actual coefficient-objective change as well as the sampled normalized diagnostic. Initial mixed readout must replay the archived guarded point within1e-5, allowing itsFP32 storage.

100Muonsteps per start, rate .1sqrt(4/1152), cosine schedule to1%floor, zero weight decay, normalized initial absolute objective. Same optimizer/rate convention as CP400; best point selected by training objective only. No evaluation metric used for selection. Baseline is the same fixed-feature guarded readout at step0. No hyperparameter winner selected from the opened panel.

Pred_a_integrity: initial readout replay<1e-5 and physicalFP32 export<1e-4, bothstarts. Finite values and gradients mandatory; first native backward memory must be<26GiB on32GBGPU. Pred_b_learning: bothstarts improve omitted-constant objective by at least1%of initialabsolute objective and text error<=1.1initial. This is not1%relative teachererror improvement. Pred_c_component: root1same-token response<=10% and sensitivity-weighted error<=1.1*.145391145 forbothstarts. Preserve every failedgate; zero prediction is100%relative error. Native finite removal/OOD/semantic promotion is not claimed from these metrics.

Independent controls:15small cases, five planted structural teacher families and mixtureweights0/.1/1000. Compare against all dense3^4coefficient entries and exact5^3Gaussian quadrature; maximum profiled gradient discrepancy1.47e-13. Dynamic Gaussian Gram reuses55subset moments/201recurrence products instead of764fully expanded partitions; values agree with the old expansion. Actual512-term,16-output CPU backward/reporting branches pass beforeenqueue. Clear recurrence caches to avoid retaining autograd graphs acrosscalls. Run through managed GPU queue only; do not mutate live helper files.

Evaluation remains4096coefficientqueriesseed951,1024standardGaussianpointsseed939,2048openedtextstates,30directed/20independent same-tokenconstraints. Sensitivityweights use actual final normalization/softcap at the original model state. Calibrationmean/covariance from6144states. These are explicit reuse diagnostics, not untouched confirmation.

If learning the metric improves training but not component transfer, the result falsifies this attempted bridge at this budget; it does not prove no compact arithmetic circuit exists. Next structural alternative is an adaptive shared hierarchy, rather than another readout-only sweep.
