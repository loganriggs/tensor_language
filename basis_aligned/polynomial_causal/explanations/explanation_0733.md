# Plain-English update — 2026-08-31 07:33Z

(Yardstick: damage = extra prediction error added above the real model; LOWER IS BETTER.)

## The morning in one paragraph
We discovered the lookup tables replacing the first four MLP layers were the program's biggest damage source
(~1.75 of ~2.86 nats), falsified that whole table grammar (no amount of correction capacity or clever fitting
objectives fixes it), and replaced it with something better: keep the model's OWN neurons, just fewer of them
("CP pruning"). Per-layer this crushes the tables at a fraction of the storage. But combining the four pruned
layers costs 3.3× the sum of the parts — the errors compound.

## What we learned about the compounding (three experiments, one story)
1. **It's a ladder**: each deeper pruned layer pays a growing multiplier (1.0/2.5/5.8/5.8×) for the drift it
   inherits.
2. **It's directional, not big**: in raw state-space the four layers' errors are nearly perpendicular and
   partially cancel; the middle layers even shrink them 2.6×. Yet the prediction error superadds. The damage
   is about WHICH directions get corrupted, not how much.
3. **You can buy the aggregate back** — more neurons (+0.119 at 75% kept) or smarter refitting (+0.437 at no
   extra storage) — **but not the circuits**: of 62 known circuits, no combined config keeps more than 2
   working. Single layers keep 8–12. Something about combining replacements kills circuit-level fidelity.

## The new instrument: a composition calculus
We measured every subset of the four pruned layers (15 configs) and built an interaction table (like a
many-body expansion in physics). Pairwise terms explain 74% of the compounding; three-way terms follow
ADJACENCY (only touching runs of layers interact); the expansion predicts any subset to ~±0.07 nats. For the
first time we can predict a config's cost before running it.

## Running right now
- Does the calculus extend across component types? (predicting the full front including the attention tables)
- Do circuits die already at PAIRS of pruned layers, or only at 3+?
- The big one: the full-model config with the pruned-neuron front swapped in — if the front really carried
  1.75 nats, this beats every full config we've ever built while storing 184M fewer values.
