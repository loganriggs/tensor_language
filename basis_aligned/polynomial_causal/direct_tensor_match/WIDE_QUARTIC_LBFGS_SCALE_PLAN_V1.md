# Initial-objective scaling for the wide L-BFGS control

22 September 2026, 04:55 UTC. Original10fits complete: five below5%, three stop after one evaluation. Installed LBFGS source stops when the directional derivative exceeds negative tolerance_change, so small absolute gradient scales can terminate before meaningful recovery.

Repeat identical10cases, starts, learning rate, line-search, tolerances and iteration/evaluation budgets. Divide the already target-energy-normalized objective by max(abs(initial objective),1e-10), a fixed scalar chosen on the first closure. This preserves the minimizer and mirrors the initial-objective scaling convention of the pending native learner. It changes optimizer numerical behavior and may change the first search direction scale. Record the scale and all evaluation counts.

Predict no one-evaluation stops and at least8/10 final value errors below5%. Both are tests, not assumed repairs. Preserve original outcomes and first251-evaluation scores. No queued native modifications or new native algorithm selection from this small control alone.
