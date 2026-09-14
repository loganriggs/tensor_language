# Shared-bank executor native boundary

Freeze the twelve contexts and eight recipients from
`DENOMINATOR_FACTORED_NATIVE_V1_ROWS.json`. For each recipient, evaluate zero
and seven off-grid coupled routing/value strengths:
`(-.75,.25),(.25,-.75),(.125,.875),(.875,.125),(1.25,.75),(.75,1.25),(-.5,1.5)`.
Mixed strength is `a*b`. Compare the exported shared-bank executor with the
native block9/attention10 path through the complete suffix.

- `pred_a`: post9, post10 and all-vocabulary score relative error are at most
  `1e-5` in every case.
- `pred_b`: baseline-subtracted target/control effect discrepancies are at most
  `1e-5 + 1e-4*abs(native effect)`; zero controls are exact and every nonzero
  intervention has at least one native readout effect above `1e-5`.
- `pred_c`: bank plus all caller contexts and runtime source are at most 89MB,
  reproduce the shared-native portfolio tensor formula apart from serialized
  framing/runtime source, and are at most35% of the all-projected portfolio.

The run has12 pristine captures and `8*8*2=128` comparison forwards:140 body
forwards,180seconds. It reuses the exact exported runtime source. Original
prefix/context generators, MLP9 weights, MLP10 and the later suffix remain
native and are explicitly reported as retained dependencies. This tests native
conditional extraction, intervention fidelity and reuse across existing
contexts. Off-grid strengths are not unseen text; no semantic identification,
runtime speedup, arbitrary-text interface or whole-model compression is claimed.

