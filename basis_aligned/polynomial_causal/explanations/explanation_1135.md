# Plain-English update — 2026-08-31 11:35Z

(Yardstick: damage = extra prediction error above the real model; LOWER IS BETTER.)

## Where we are
Best full-model replacement: 1.88 nats added (or 1.66 with a richer front) — down from 2.67 this morning,
at 184M fewer stored values. Both points preregistered and reproduced. Circuit certificates remain 0/62.

## The name mystery
The single worst-predicted class is NAMES (+3.1 nats). We spent the morning eliminating suspects, and every
single-family fix failed: exact early layers (−9%), real mid-attention (−18%), richer per-class maps (−2%),
and even handing every name position its real attention output (−26%). Conclusion: the damage is
COMPOSITIONAL — spread across component families and their interactions.

## The new idea running right now
Our "perfect attention" test had a subtle flaw: the attention module was real, but it was reading a
CORRUPTED stream — like a perfect librarian searching a vandalized library. The new experiment splices in
what attention would have returned on the UNCORRUPTED stream. If names recover, the fix is stream fidelity
(keep the library clean), not fancier retrieval modules. If not, the damage is somewhere stranger still.

## Also this hour
Two instrument bugs were caught by their own tripwires and numbers (a self-targeting fit and a double-free),
both filed as reusable rules. The failure ledger is doing its job: every void run so far has been detected
before any conclusion rested on it.
