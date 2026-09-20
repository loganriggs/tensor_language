# Quartic sparse-basis replication — 2026-09-20 15:51 UTC

Replicate the first-context gauge result over all16 existing native quartic groups, first row of each. Select each starting width4 shared-bank fit by its Frobenius error in the existing three-restart sweep, reconstruct that fit deterministically, and check its error agrees with the saved receipt. No new context is selected by sparsification success.

For each bank, run Muon rate0.05 with two gauge restarts for1500steps. These hyperparameters are chosen from the first-context pilot. Preserve the starting polynomial under the invertible feature basis change; normalize each quadratic feature by its Gaussian norm; minimize group sparsity over root pairs. Refits impose2/4/6/8/10 shared products. Compare against exactly the same original bank and root-refit procedure.

Predictions: (a) reconstruct saved starting fits within1e-8 error; (b) gauge transformations preserve functions within1e-8; (c) at least12/16 contexts improve four-product error by a factor2; (d) all-product refit error remains invariant within1e-8. Null: pilot sparsity gain is exceptional. Record per-context and worst-case results, conditioning, and actual exported factors. This is independent fitting per context, not one circuit transferring across contexts. Differences between polynomial numerator approximation and full normalized model remain explicit.
