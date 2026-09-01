# Plain-English update — 2026-09-01 00:33Z

(Damage = extra prediction error above the real model; LOWER IS BETTER. A "certificate" = a behavior kept
demonstrably close to the original.)

## The night ended with a law about why compression stops
Every attention head in this model computes its "where should I look" scores with a set of directions in
weight space. The big directions carry the obvious structure and compress beautifully. But the SMALL
directions - the last ~25% - turn out to hold one shared, extremely delicate cancellation trick used by
the whole network at once. Break it anywhere - remove a few small directions at a few heads in the front,
or the back, or anywhere - and you pay the SAME ~0.055 error with the SAME per-behavior fingerprint. Keep
it fully intact and you pay nothing. It is all-or-nothing, everywhere at once.

Consequences, all measured and registered:
- The best compressed model keeps 104 of 128 directions per head everywhere: 0.057 extra error, 11 of 62
  behaviors certified, ~1/3 the original size in attention maps. That is the floor of this approach - not
  because we ran out of ideas, but because we proved any further pattern compression pays the fixed toll.
- The MLP side is the opposite: each token really uses ~1/4 of the units, selecting per token is nearly
  free, and its cost composes additively with everything else (four measurements within 0.001).
- Value maps (what heads copy, rather than where they look) price smoothly - no delicate trick there.

## What's next
The program's third goal - showing the compiled model can be MANIPULATED like the real one (delete a
part, get the same predicted effect) - is now the live test on the GPU.
