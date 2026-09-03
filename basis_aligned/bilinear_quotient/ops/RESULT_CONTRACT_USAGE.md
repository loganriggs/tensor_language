# Generic saved-result contract

`result_contract.py` is a standard-library-only, side-effect-free validator for
the boundary between a preregistered experiment and a saved JSON result. It is
not a scientific gate: each experiment still defines its own metrics and
thresholds.

Before interpreting a result, construct a `ResultContract` from frozen
pre-outcome facts, then call:

```python
summary = validate_result_contract(result, flat_raw_rows, authority_rows, contract)
```

The call fails unless all of the following hold together:

- the raw table contains exactly every authority row in the opened split(s),
  with the same split and declared grouping fields and no duplicates;
- the result's split declaration, observed raw splits, and preregistered opened
  splits agree, while forbidden splits remain closed;
- result, raw evidence, and authority are literal finite JSON values that can be
  encoded with `json.dumps(..., allow_nan=False)`;
- fields declared as scalar, string, number, list, dict, or null have exactly
  that shape;
- forward calls stay inside the frozen price (and equal it when requested),
  backward calls match, and the weight-update flag matches;
- every required provenance entry is a lowercase SHA-256 digest and every
  frozen expected digest matches.

Flatten nested evidence to one record per authority row. If a result has several
arms or directions, validate each arm/direction table separately against its
corresponding authority subset; do not concatenate duplicate row IDs and weaken
the identity check. Use multiple group fields when membership is defined by more
than one label, for example `group_fields=("group_id", "family_id")`.

Run the focused regressions with:

```bash
pytest -q basis_aligned/bilinear_quotient/ops/test_result_contract.py
```

They include named failures for a singleton list in a declared scalar field, an
omitted group/row census, and non-finite values hidden in a null-result path.
