# Plain-English update — 2026-08-31 19:35Z

(Yardstick: damage = extra prediction error above the real model; LOWER IS BETTER.)

## The evening's discovery: one grammar to rule the attention
For months the compiled model's attention used a patchwork: fixed copy-patterns ("motifs") in the middle,
class dictionaries at the tail. Tonight both were RETIRED. The replacement is a single rule applied to all
148 replaced heads: keep the model's own attention computation, but compress each head's pattern-forming
maps to rank 16 by plain SVD (no fitting, no training). Result, preregistered and reproduced:

  best config this morning: 1.95 nats added  ->  tonight: 1.27 nats  (fresh-text metric: 1.88 -> 1.35)

The whole day's registered path halved the error: 2.67 -> 1.35.

## Why rank 16? A little theorem told us
This model's attention squares two attention-score matrices together (a product of two bilinear forms).
Matrix rank multiplies under such products, so rank-r factors give pattern rank ~r^2 - and each head needs
rank ~128. Prediction: the critical factor rank is sqrt(128) ~ 11. Measured: rank 8 breaks badly, 10 is
poor, 12 works, 16 is the plateau. The theory called the knee's location before we measured it.

## Still true, and still the deep puzzle
Even at 1.27 nats, ZERO of the 62 circuits pass their fidelity certificates - a day of halving aggregate
error moved circuit-level fidelity barely at all (circuits kept ~86% of their damage through the last
grammar change). The two currencies remain decoupled; the circuit repertoire documents exactly how.

## Running now
The top of the rank curve (does rank 24 help?), a free-win test (a legacy component that costs 0.14 nats
and compresses nothing), the front-layer re-price, and the new frontier's per-circuit bookkeeping.
