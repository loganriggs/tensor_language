# Terminal-copy selection v1: attempt-2 recovery ruling

Frozen: 2026-08-29, before the second execution attempt and before any
terminal-copy model outcome was observed.

## What happened in attempt 1

The first execution authority is spent.  Its receipt-last lifecycle published
`terminal_copy_selection_v1_failure.json` and explicitly forbids a retry under
that authority.  Execution stopped while hashing the freshly loaded model,
before construction of the intervention dispatcher and before the first model
forward.  PyTorch rejected `view(torch.uint8)` on a zero-dimensional bfloat16
buffer.  No ledger, result, manifest, passer receipt, or scientific-negative
receipt was written.

## What attempt 2 may change

Attempt 2 is a new authority and a new create-only output namespace.  The only
executable repair is to flatten a tensor before viewing its bytes:

```python
tensor.reshape(-1).view(torch.uint8)
```

This preserves the raw bytes and separately hashes the original dtype and
shape.  It changes neither model computation nor any scientific choice.

## What attempt 2 must not change

The 192 natural documents, 64 descriptive synthetic rows, eight candidate
interventions, fitted means, masks, causal arithmetic, batch schedule,
estimands, 10,000 joint document bootstraps, thresholds, tie-break, and
final/OOD opening rule remain exactly those of attempt 1.  Attempt 1 disclosed
no model outcome and therefore supplies no basis for adapting any of them.

The new authority must truthfully record that the selection rows and masks, fit
bank, and checkpoint were already deserialized, although no model forward or
outcome occurred. It must bind the exact first authority and terminal failure,
pin every absent attempt-1 ledger/result/manifest/decision-receipt/lock path,
protect all of that state against mutation, use a distinct lock and output
namespace, bind a new pushed source closure, and receive a fresh outcome-blind
independent audit.
The old failure remains immutable.  A failure in attempt 2 spends attempt 2;
there is no same-authority retry.
