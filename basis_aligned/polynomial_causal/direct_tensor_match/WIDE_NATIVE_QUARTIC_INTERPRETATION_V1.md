**More features and longer fitting lower training error without improving transfer.**

The wider dictionary targets the true native pure quartic path, rather than an archived approximation. Width4 has26products/48,384coefficients; width32 has656products/903,168coefficients. Both use four bilinear products per quadratic feature and all unordered pair products. The physical residual writer includes the QR inverse and target scale; the original unembedding remains a common external operation. These programs still omit residual cross terms, biases and normalization from the reconstructed polynomial path.

| Features / initialization |100step fit|100step second panel|1000step fit|1000step second panel|
|---|---:|---:|---:|---:|
|4 / inherited|8.46%|12.99%|6.15%|12.93%|
|4 / random|11.63%|19.41%|6.64%|14.92%|
|32 / inherited|2.97%|12.87%|2.16%|15.05%|
|32 / random|3.66%|19.48%|2.49%|19.89%|

Both native runs pass numerical/export checks and fail the registered width-improvement and5%both-panel targets. The1000step run takes34.85s. Best checkpoints are selected by training objective only and occur at the final step. Longer fitting improves the narrow random start but worsens both wide starts' second-panel errors. No further extension of this same empirical objective is justified merely because training is still improving.

Five-family optimizer controls establish numerical correctness and local recoverability, not easy global discovery:300stepAdam recovers1/10random starts versusMuon0/10;3000stepAdam recovers6/10, covering four families below1%. The remaining family's best error is1.071%. All five perturbed-teacher starts pass below.06%. Augmented least-squares and envelope-gradient controls agree below1e-9.

A CPU diagnostic of the frozen100step readout designs rejects its proposed strong extrapolation threshold. Wide models' mean second-panel leverage is1.25and1.55times training, below the registered factor2. Their conditional readout effective dimensions are near528. Narrow ratios are.80and.74. Some wide second-panel rows have high leverage, but this does not establish a causal explanation for aggregate transfer error. These quantities condition on learned features and do not count the parameters used to learn them.

The next distinct comparison should constrain the broad dictionary using folded weights. The existing exact quartic teacher-to-feature contraction can be batched over root pairs without expanding the full coefficient tensor. A new batched helper matches both values and gradients of the validated unbatched contraction on three sizes, including partial final batches. This enables a controlled fixed-dictionary output refit in isotropic and covariance-weighted coefficient metrics. It must be labeled weight-based readout fitting of a data-informed dictionary; it is not end-to-end data-free discovery. No such native refit result is claimed here.

[Short pilot](WIDE_NATIVE_QUARTIC_V1.json) · [Long pilot](WIDE_NATIVE_QUARTIC_LONG_V1.json) · [Readout leverage diagnostic](QUARTIC_FEATURE_LEVERAGE_V1.json) · [Batched contraction controls](BATCHED_QUARTIC_CROSS_CONTROLS_V1.json).
