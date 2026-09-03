# Rung 534 terminal receipt: the target remainder is interaction-dependent

**Completed and independently audited:** 2026-09-03 13:20 UTC

## Registered outcome

The run is valid. Predictions A and B pass; C through F fail. The preregistered interaction-only strong null is true.

The shared score `S = gamma * P_source` reproduced its inherited code/source-absent premise in both document halves.
The target remainder `R = P_target - S` did not act autonomously: on copy-positive document effects it failed the
60% relative-error bound in both code halves (`66.5%`, `77.8%`), even though its directions were similar to the
true marginal effect (cosines `92.2%`, `89.4%`). It passed the matched-negative autonomy rule in both halves and beat
the key-reversed and sign-flipped controls in all eight code/source-absent cell-control comparisons.

Thus `R` contains relation-specific information, but its useful copy effect depends on composition with `S`. Do not
adopt `R` as a standalone circuit component. The next analysis should expose the exact factorial interaction
`E_native - E_S - E_R` across corpus, token group, and source background.

## Execution and audit

- model forwards: exactly `1,440`;
- backward passes and fitted vectors: `0`;
- runtime: `34.58` seconds;
- instrument A: pass;
- inherited shared premise B: pass in `2/2` code/source-absent halves;
- private autonomy C: pass in `0/2` code/source-absent halves;
- independent audit: recomputed all reports, gates, strong-null decision, and forward count from the saved
  per-document sufficient statistics.

Frozen artifact hashes:

- result: `8804dca2cbd0203a6ef9517a15ec7a4186ed5e69ec8c284b854967c8e13197a7`;
- sufficient-statistics bundle: `77ca551a19004abade5ec5dcc79023a01f3d9c5d97ca693c012ca74f512cef80`;
- managed log: `e0e58eeb8b578423446dc956bcbd2b00825c5ee7c24cb177319ee380109d75f4`;
- independent audit: `16242c79f0b42a85cb9db1c12c2d4bc8d955356ab5266bcd2ac8d67cb7a2cff7`.
