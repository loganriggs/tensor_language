# Rung 560 v2 implementation correction

**Frozen:** 2026-09-03 17:52 UTC, after v1 crashed and before any R560 result existed

V1 completed FIT model evaluation and then crashed before writing a result. The in-memory raw dictionary had shape

```text
raw[split][family][direction][arm][source_kind]
```

but `score()` correctly expected the already-selected split:

```text
raw[family][direction][arm][source_kind].
```

The only v2 change is for the evaluator wrapper to return `raw[split]`. No model value appeared in the run log, no
result file exists, and SELECT/FINAL_TEST/OOD were not opened. The dataset, interventions, source positions, metrics,
thresholds, FIT choice, SELECT rule, and all hashes in the scientific preregistration are unchanged.
