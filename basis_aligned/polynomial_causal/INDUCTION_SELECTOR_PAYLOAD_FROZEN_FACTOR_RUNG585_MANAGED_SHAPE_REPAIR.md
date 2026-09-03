# R585 managed-run shape repair

**Status:** prospective execution-only repair after a pre-outcome managed failure  
**Failure time:** 2026-09-03T23:17:29Z  
**Scientific outcomes opened:** none  
**Result, receipt, and evidence namespaces:** absent after failure

The independently approved iteration-6 adapter entered the managed GPU runner, loaded the checkpoint, and stopped on
its first model forward. The traceback was:

```text
RuntimeError: tokens must have shape (4, 256)
```

R585 preregistered batches of 32 examples and deliberately mixes prompt lengths to test padding. The observed-model
facade's `require_production=True` option instead enforces the separate fixed validation shape `(4, 256)`. Thus the
approved implementation's literal batch schedule and its facade flag were mutually inconsistent. No endpoint logits,
intervention scores, result, receipt, or evidence package was published.

The prospective repair changes only the three R585 scientific calls to `forward_with_dispatch`: capture/replay,
independent native comparison, and intervention evaluation now pass `require_production=False`. This flag relaxes the
facade's tensor-shape assertion; it does not change the loaded model, weights, attention/MLP dispatch, frozen batch-32
schedule, interventions, thresholds, row census, or 459/231 forward budget. Checkpoint identity and model structure
remain validated by the unchanged model-loading path.

A new owner test parses all three scientific collectors and requires exactly one facade call in each with an explicit
false shape flag. This prevents a future dry run from approving the fixed `(4, 256)` option for the registered batch-32
experiment. The owner and adapter suites pass (`72 passed` combined), both gates pass, and the managed adapter dry run
remains model-free.

This repair is not authorized for GPU execution until a fresh independent reviewer approves its exact committed bytes.

