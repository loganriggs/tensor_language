# Subject-number native head-response coordinate discovery V2 correction

V1 stopped before evaluating either head condition or writing a result. The runner attempted to read `.dtype` from the captured MLP8-input component dictionary rather than from an assembled raw input tensor, raising `AttributeError: 'dict' object has no attribute 'dtype'`.

V2 makes one implementation-only correction: after assembling $x_b$, it constructs the frozen native axis with `dtype=xb.dtype`. The V1 preregistration, four designs, leave-one-construction-out split, thresholds, authority, artifacts, and price are unchanged. The V1 failure log remains preserved.
