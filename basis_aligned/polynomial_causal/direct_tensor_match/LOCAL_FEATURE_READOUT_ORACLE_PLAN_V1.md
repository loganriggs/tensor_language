# Diagnose the new feature dictionaries after failed native recovery

22 September2026,06:01 UTC, before execution. Existing CP512 frozen-span oracle does not answer this question: these are newly learned96 atoms, not the old512 dictionary.

Use all six completed local corrections (Adam/Muon/L-BFGS, two starts), parent fixed. On the opened16384-state panel, fit oracle coefficients directly to the parent's residual in outputs4–15. Compare two readout classes: (a) each output uses only its own8 learned atoms; (b) all12 outputs can reuse all96 learned atoms. No feature directions change. Also solve independently on2494 response differences; do not claim the independently optimal value and response readouts are one model.

Primary diagnostic prediction: allowing all96 atoms reduces equal-small-output value AND response RMS by at least15% relative to the output-local oracle in both Adam starts. This tests whether output-local assignment restricts reusable computations. Report absolute errors and every optimizer/start regardless. A failure demotes readout connectivity as an explanation; a pass would justify testing learned shared readouts, not claim a generalizing circuit.

Both fits use evaluation answers intentionally. They are finite-panel best-case diagnostics, not deployable candidates or fresh validation. Dense sharing adds1056 readout coefficients (12x96 versus12x8) without new quartic products. No refit of the original parent, no offsets, no export. Normalize design columns only for numerical conditioning; this does not change their spans. Require float64 SVD/QR residual agreement<1e-8 and normalized residual orthogonality<1e-10; report ranks/conditioning. Singular matrices require rank-aware projections rather than full-column QR.
