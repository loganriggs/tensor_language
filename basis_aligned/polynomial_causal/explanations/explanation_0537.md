# Plain-English update — 2026-08-31 05:37Z

(Reminder of the yardstick: every damage number is "extra prediction error added above the real model";
LOWER IS BETTER. The whole-model replacement currently adds ~2.86 nats on the census evaluation.)

## The big result of the night: the certificates all fail, and we now know why
We built a "certificate" test: for each of the 62 known circuits in the model, a replacement config passes
if it damages that circuit's own positions by less than half of what ablating the circuit's top component
does. Result: NO config we have ever built passes for even one circuit — not the full frontier config, not
gentler ones. Chasing this down produced a decomposition: the biggest single source of damage is the oldest,
most trusted part of the program — the LOOKUP TABLES replacing the first four MLP layers (~1.75 nats of the
~2.86). The fancy attention replacements everyone worried about are smaller (~0.36 and ~0.29).

## A correction, caught by reading code
Two entries briefly mislabeled which pieces were installed (the tail-MLP dictionaries ride along in every
config). Per our correction rule, we ran a physical control before fixing the story: the control confirmed
the tables really do carry the bulk (+1.75 of +1.92), so the headline survived with refined numbers.

## Three crisp new facts
1. **One replacement is perfect**: the layer-0 attention value table is EXACT (zero damage, passes all 62
   certificates) — because that module only looks at the current token. The certificate test can be passed.
2. **The first MLP table alone breaks 61 of 62 circuits** while adding only +0.25 nats — circuit breakage
   happens long before aggregate damage looks bad. Aggregate CE and circuit fidelity are different currencies.
3. **No single culprit**: each of the four front MLP tables, installed alone, kills 60+ of 62 certificates.
   And the errors are context-signal — not fixable by matching means or by per-class corrections.

## What's running now
The two live repair hypotheses, one experiment each: (a) CAPACITY — does making the table's correction term
8× bigger fix layer 2? (b) OBJECTIVE — does refitting layer 0's correction with 10× weight on circuit
positions fix the certificates? If both nulls win, the table grammar itself is falsified at the front, and
the next chapter is a new grammar.
