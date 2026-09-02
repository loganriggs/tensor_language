# Hourly strategic review — 2026-09-02 12:31 UTC

## What changed since 11:31

- Rung483 showed that local derivatives do not predict complete removal of MLP0's token-only (`T`) or
  token-by-context (`I`) terms. The exact finite effects nevertheless distinguish T from I at attention1 and MLP1,
  so the useful signal is nonlinear and branch-specific rather than a single local linear reader.
- Rung484 therefore measured the complete finite attention1 computation. Its arithmetic and replay checks passed,
  but no proper attention1 subpath explained both branches. T needs the full interaction-heavy path. I has a stable
  near-result through score side B plus carried value, but it is generic rather than equality-specific and narrowly
  misses independent reselection. The stable T/I path profiles are anti-aligned; that is anatomy, not yet a circuit.
- Rung485 is implemented, CPU-tested, committed, and queued through the managed runner. With attention1 restored to
  its native value, it measures all four finite native/absent combinations of MLP1's Left and Right activations. It
  asks whether either physical side predicts the complete downstream route and whether input token identity predicts
  T's route on held-out occurrences. Equality-positive results are descriptive, not a reused task-selectivity gate.
- In the independent equality-MLP arc, the orthogonal-complement mirror test is currently running ahead of rung485.
  The runner serializes them, so this is coordination rather than competing GPU use.

## The seven goals remain the acceptance test

1. Give a human-readable account of what information is read, what operation is performed, what is written, and
   which later computation uses it.
2. Group pieces across heads or MLPs when downstream computation treats them as the same variable.
3. Split a native head or MLP when different pieces perform different computations.
4. Predict activations and causal effects on held-out data and genuinely shifted input distributions.
5. Extract a named computation as an executable object, or specify a precise interface plus background.
6. Remove, swap, or edit that computation while preserving unrelated circuits and accounting for redundancy and
   interactions.
7. Recover stable, reusable components that compose into more than one larger circuit and survive equivalent gauges.

Rank reduction, quantization, byte count, reconstruction error, or average CE alone does not meet these goals. Rank
can later price an already identified circuit or provide a matched-capacity control. It must not choose the semantic
units. Rung485 is worth running because its variables are exact architectural sides and exact downstream effects,
selection is held out, T and I can split, and a token-effect pass only licenses clustering followed by physical
interchange. It claims no compression.

## Alternative explanations and result-conditioned routes

1. **A proper MLP1 side exists.** If one side predicts each route and the relation validates, use exact interchange
   to test whether T and I share that side or use different sides. This advances extraction and selective editing.
2. **No proper side exists.** Preserve the full finite T/I responses and form a coupled token-by-context response
   object across attention1 and MLP1. Decompose it by held-out downstream equivalence, not by product rank or another
   subset search. This treats nonlinear cancellation as signal rather than noise.
3. **Token identity predicts T but a side does not.** Group tokens by their frozen downstream-effect profiles, then
   require within-group interchange to work better than between-group interchange. The 698-label lookup itself is
   only a diagnostic and cannot be the explanation.
4. **Token identity fails.** Condition the reader on live context/state. This would mean that “token-only at MLP0”
   does not imply “token-only downstream role,” because later nonlinear computation uses the same write differently.
5. **Path profiles change across halves.** Retain per-example effects and model the context-dependent finite response;
   do not average away the instability or tune thresholds.
6. **Independent geometric mirror test passes.** Keep its natural/code register geometry as a separate composition
   result. It does not replace the MLP0 decomposition and should not be used to relabel rung485 outcomes.

The highest-information next action is therefore unchanged: score rung485 literally when it lands, then begin the
specific successor selected above. No receipt, explanation, commit, empty queue, or achieved subgoal is a stopping
condition while the compiled predictive/manipulable tensor-program goal remains open.

## Cadence and liveness

- Managed GPU: orthogonal-complement mirror test active; rung485 queued immediately behind it.
- Latest mathematical review: `THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-02_1114.md`; next due after 14:14 UTC.
- Next hourly strategic review is due after 13:31 UTC if the chain remains active.
- Safe boundary after rung485: scored receipt plus a genuinely started result-conditioned successor.
