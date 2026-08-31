# Plain-English update — 2026-08-31 23:33Z

(Damage = extra prediction error above the real model; LOWER IS BETTER. A "certificate" = one of 62
behaviors kept demonstrably close to the original: damage on that behavior's text under half the effect
of deleting its key component.)

## Tonight's arc: from "compress everything evenly" to a designed allocation
- We found WHERE the compressed model still hurts: the early attention blocks (2–5), where the model's
  local-grammar machinery lives. Making just those exact recovers most of the remaining error.
- The retrieval tail (blocks 10–17) is the opposite: shrinking it further barely moves average error but
  quietly kills certificates — its last bit of precision matters only to a few specific behaviors.
- Best registered model now: exact early/motif attention maps, half-size tail maps — 0.061 extra error,
  9 of 62 behaviors certified, at the same price as the old uniform version but strictly better.
- Strangest finding: the 9th certificate exists ONLY when the whole motif stack is exact — no part of the
  stack "owns" it. Behaviors can depend jointly on many pieces so that no per-part accounting predicts
  them. We saw the same shape earlier in the failure direction (costs compounding when composing parts).
- Also closed tonight by their own preregistered tests: static per-token lookup tables inside the
  compressed model (they compose badly), learned cluster indices at depth (plateau far above the dynamic
  gate), circuit-targeted rank (zero certificates).

## Where this leaves the program
A two-tier menu, both official or in registration: an economical shape (0.068 error, 8 certificates,
smallest yet) and a certificate shape (0.061, 9). The open scientific question: which behavior is the
joint 9th certificate, and what do the remaining 53 need?
