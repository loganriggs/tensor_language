# Plain-English update — 2026-08-31 17:00Z

(Yardstick: damage = extra prediction error above the real model; LOWER IS BETTER.)

## The circuits program's first day, concluded in five findings
1. **Circuits are ~5-component objects** (of 16 measured) — but keeping exactly those 5 while removing the
   rest rescues NOTHING (worse than keeping 5 random ones, barely better than keeping none). Necessity
   without sufficiency: the 62 circuits are facets of one deeply coupled computation, not detachable parts.
2. **Their identity is positional, not directional**: the "special subspace" each circuit seemed to use
   turned out to be the component's ordinary principal variance — computable without knowing the circuit at
   all. What makes a circuit a circuit is WHERE it fires, not a dedicated coordinate system.
3. **The surgical tool is counterfactual swapping at member positions** (~250x more selective than
   deletion) — but the "member positions" gate is not cheaply computable (not linear, not class-based, not
   both). Testing one last hypothesis now: individual leaves may be separable even though families aren't.
4. **Partial repair is real**: inside our best compiled model, restoring a circuit's own 5 components to
   real behavior buys back 30-56% of that circuit's damage. Half the kill is circuit-local, half is
   stream-wide. (One control arm hit a numeric-overflow bug — caught by its own NaNs, fixed, re-running.)
5. **Everything is in the repertoire**: circuits/REPERTOIRE.json now holds, per circuit: components, refs,
   minimal-set size, removal profile, which simplifications preserve it, variance-basis shares, and the
   honest verdicts (extraction FALSE at component grain; OOD blocked at leaf recomputation).

## Where this leaves the big picture
Aggregate compilation and circuit fidelity remain different currencies (best config: 1.88 nats added,
0/62 circuits certified). The day's structural results explain why: the circuits share substrate, live in
positions, and can't be repaired by any additive or subspace patch. The plausible paths forward are
whole-substrate fidelity (the frontier program) with documented circuit costs, or intervention at a finer
grain than components.
