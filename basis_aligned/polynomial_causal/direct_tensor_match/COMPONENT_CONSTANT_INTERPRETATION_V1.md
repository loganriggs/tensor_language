**Original-fit constants do not repair code transfer**

The graph and both baselines independently received three constant offsets, fitted as mean component errors on exactly the original448 fitting states. No code-panel values were used to fit these offsets. This adds three coefficients and three additions per site, with no additional products. Scalar derivatives and donor differences would be unchanged; nonlinear endpoint differences need not be.

Fitting error decreases as required by least squares. Graph component-three error changes only from8.3463% to8.3362%. Its calibrated mean is0.9377, whereas the opened code/spaced-word natural means are21.6387 and30.3182. The covariance baseline's calibrated mean is1.4038 versus11.9446 and17.6426 in code. The apparent cross-code-panel transfer of mean offsets therefore does not generalize from the original fitting distribution.

Using exact saved error sums and energies, the graph/covariance-baseline natural ratios remain1.15275 and1.13646 after both programs receive their own correction. Hybrid ratios are1.04822 and1.15873. No previously passing scalar comparison becomes failing, but the existing failures are not repaired. Some other cached component errors also increase slightly. **Reject this candidate; do not launch a full endpoint run merely to pursue an already-failed scalar repair.** No program artifact has been modified.

The next analysis moves from aggregate read errors to explicit native computation paths. Write $h=m+c$, with $m$ the folded MLP16 contribution and $c$ everything else in the residual supplied to MLP17. If $q_a=m^\top A$ and $q_b=m^\top B$, then

$$
a=\frac{c^\top A+\tfrac12q_a}{s}-\alpha.
$$

The second-read error contribution becomes

$$
\frac{a\delta_b}{s}
=\frac{(c^\top A)\delta_b}{s^2}
+\frac{q_a\delta_b}{2s^2}
-\frac{\alpha\delta_b}{s}.
$$

This distinguishes a cross-boundary carry interaction from the within-source product and normalized centering. Combined with the analogous first-read expansion and the error product, there are six exact terms. Retaining their signed cross terms is necessary. The carry still combines multiple sources; a dominant carry term would motivate splitting named earlier residual/attention contributions, not declaring a single circuit.

[Calibration results](COMPONENT_CONSTANT_CALIBRATION_V1.json) · [Path diagnostic preregistration](NATIVE_PATH_ERROR_PLAN_V1.json).
