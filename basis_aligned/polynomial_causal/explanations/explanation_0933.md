# Plain-English update — 2026-08-31 09:33Z

(Yardstick: damage = extra prediction error above the real model; LOWER IS BETTER.)

## The frontier fell twice this morning
Our best full-model replacement improved from 2.67 nats of added error to **1.88** — while getting 184
million stored values SMALLER. Two moves did it: (1) replacing the first four layers' lookup tables with a
pruned copy of the model's own neurons; (2) refitting the tail-attention dictionaries to aim at the REAL
model's outputs rather than their own drifted stream ("trajectory teaching"). Both were preregistered with
bars and reproduced exactly.

## Then the same trick failed twice — informatively
Applying trajectory teaching to two more component families made things WORSE (one catastrophically). The
pattern so far: it helps components that were already approximations (dictionaries), and hurts components
that carry the model's own exact weights — where refitting trades exactness for steering and loses. Two
control experiments now running will tell us whether that's a real law or an instrument bug. Until they
land, no conclusion is claimed; the frontier config itself is untouched by the failures.

## The sobering constant
Circuit-level certificates remain stuck near 0 of 62 for every combined replacement, even as aggregate
error halves. Aggregate quality and circuit fidelity are different currencies; the "direction problem"
(all our replacement errors damage the same shared subspace the circuits need) is the deep open question,
now assigned to the mathematical review.
