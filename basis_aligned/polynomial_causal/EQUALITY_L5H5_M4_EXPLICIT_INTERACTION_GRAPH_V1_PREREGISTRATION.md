# Equality L5H5 M4 explicit interaction graph V1

The certified rank-256 M4 node is causal and OOD-predictive, but its best
prospective 128/128 split has `.13323` code score composition error and `.42108`
behavioral NLL composition error.  Rather than force additivity, this experiment
promotes the cross-difference to a first-class graph node.

For bias-only baseline score `S0`, rank-128 child score `Sc`, complementary
rank-128 remainder score `Sr`, and direct rank-256 joint score `Sj`, define

`C = Sc-S0`, `R = Sr-S0`, `I = Sj-Sc-Sr+S0`.

The graph output is `S0+C+R+I`.  `I` is an executable contextual interaction
node with the same declared M4-input, mode-program, and L5H5 score ports as its
parents.  It currently costs four score evaluations; this experiment does not
pretend it is already a minimal closed-form kernel.

Frozen gates:

1. **Lawful graph.** The inherited float64 bridge passes; graph output replays
   `Sj` within `2e-6` relative L2 on natural and code, including both halves.
2. **OOD prediction.** Direct `Sj` retains the inherited code five-write-parent
   error at most `.15` and cosine at least `.98`; graph score and donor NLL replay
   direct joint within `2e-6`.
3. **Selective removal.** Removing `I` yields the additive score.  Its NLL change
   relative to the joint-vs-baseline effect is at least `.20` on copy-positive,
   every copy subtype, and both halves, while incremental noncopy mean damage is
   at most `.01` nat.
4. **Directional null.** Rolling `I` by one query position preserves its score
   norm exactly.  Its copy-positive NLL-effect cosine with true interaction
   removal is at most `.90`; equal norm is checked directly.
5. **Causal use and composition.** Direct and composed joint recovery are both
   in `[.80,1.05]`, differ by at most `2e-4`, all donor terms are nonzero, graph
   closure passes, and the node has zero learned parameters.

Natural rows only confirm graph closure and interaction magnitude; no selection
or refit occurs.  Code rows test all frozen claims.  Failure of inherited bridge
or graph replay is invalid; other failures are valid interaction-node nulls.

Price: one checkpoint load, one natural score pass, one code score pass, and
seven code behavior arms (native, absent, baseline, additive, composed, direct,
rolled-interaction); no gradients, fits, parameter updates, or new text.
