**Full-target metric sensitivity after failed low-rank prediction**

Repeat exact unfolding bounds for three input metrics: folded concatenation Euclidean E=[I,lambda D16,O17]; original residual Euclidean I; covariance square root estimated from2048historical lastMLP inputs after RMS normalization. Center covariance, floor eigenvalues at1e-8times mean eigenvalue and disclose count. Keep full unembedding metric. No student fitting, no activation-output regression.

Predict residual output-rank128floor<20%; predict covariance output-rank128and input-rank512both<20%. Replay original folded values to numerical precision; native Gram traces<1e-9, PSD>-1e-10relative. Full-rank bounds do not preclude sparse arithmetic circuits; metric differences must not be called structural success. Centered covariance coefficient norm is not the empirical fourth-moment functional loss and excludes mean-shift lower-degree terms. No causal or semantic claim.
