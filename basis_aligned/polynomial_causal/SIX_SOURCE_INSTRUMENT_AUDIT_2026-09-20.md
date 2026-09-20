# Six-source negative-result audit

The original six-source run remains **invalid under its registered instrument gates**.
Its original receipt is preserved. This CPU audit reuses saved coefficients and
does not rerun or reinterpret native outcomes as validated evidence.

The old five-source derivative subblock agrees within 2.22e-16 when both saved
arrays are loaded as float64. The runner accidentally constructed the comparison
tensor in float32, producing 5.68e-8 error against a 1e-8 gate. Reproducing that
cast reproduces the reported error exactly: this failure was an implementation bug.

A separate failure remains: casting individual source differences to float32
before summation produces a maximum absolute closure error of 1.72e-5, above
the registered 1e-5 limit, despite relative error 1.71e-8 and exact pre-cast
state closure. Finite-difference checks pass (gradient 0.109%, Hessian 0.299%).
Fixing one comparison therefore does not validate the whole instrument.

Exploratory native cell counts are 45/48 for six sources versus 42/48 for the
five-source control. Those counts are not promoted: instrumentation failed,
three strength failures remain, and adding an intervention direction increases
capacity. Both datasets are opened, and the intervention edits only one site's
residual while retaining baseline cache and other positions. Neither a full
input edit nor an extracted reusable circuit has been established.

The next implementation should preserve differences in float64 for the reference
and construct the native summed edit before the final cast, explicitly measuring
the latter's rounding. Declare that changed numerical interface before rerunning;
do not silently loosen the existing gate. Compare the same five-source control
and report source-generation, derivative and optimizer costs.

Reproduce the audit with `python audit_six_source_instrument_v1.py`; its JSON
receipt records hashes of the native and comparison artifacts.
