# Variable projection for native bilinear features — 2026-09-20 15:55 UTC

The teacher-channel baseline proves an optimization gap at fixed width:1024 selected native channels with exact output refit attain82.0% Frobenius error versus95.5% in the prior random-start sweep. This does not prove a small exact decomposition exists.

Test random-start variable projection: given input factors A,B, solve the output factor D at every step through the exact quadratic-feature Gram matrix. Optimize only the input factors. Use tiny relative1e-6 diagonal regularization for numerical conditioning; evaluate the unregularized error of the actual solved D. Differentiate through the solve. Because regularization is nonzero, the solve is a ridge solution, not an exactly unregularized minimizer.

Eight fits: widths128/512, Adam/Muon, two restarts, learning rate0.005,400steps. Frobenius objective only. Compare to prior same-width Frobenius joint fits and teacher-channel baselines; unequal iterations and eliminated output optimization mean this is an algorithm comparison, not a controlled optimizer ranking. Record initial and final loss, Gram condition, normal equation residual, independent implicit-score agreement, both metrics and compute time.

Predictions: implicit replay agrees with Gram loss within2e-5 relative squared error; every fit improves its initial objective; at least one width512 fit beats0.8981 (teacher selected-channel/refit baseline). Null: variable projection at these settings does not close that gap. Output factors remain dense and full frame cost remains explicit. No native forward or causal claims.
