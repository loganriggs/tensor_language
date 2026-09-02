# Hourly strategic review — 2026-09-02 14:33 UTC

## Circuit targets and full goal

The target remains a smaller executable tensor program that is predictive on fresh and shifted text, composes when
several replacements are installed, supports selective removal/swapping/editing, and is cheaper under literal
storage, compute, edge, state, and program prices. A circuit decomposition must eventually establish:

1. what information is read, what operation is performed, what is written, and what later computation uses it;
2. grouping across heads or MLPs when downstream computation treats their parts as one variable, and splitting a
   native module when its parts do different jobs;
3. held-out and OOD prediction of activations and causal effects;
4. an executable extracted circuit, or an explicit interface plus its necessary background;
5. selective manipulation that changes the intended behavior while preserving unrelated behaviors and accounting
   for redundancy and interactions;
6. predictable composition and reuse of shared and task-specific pieces; and
7. stable identification across document/corpus splits, plausible gauges, and refits, or by operational downstream
   equivalence.

Lower rank, quantization, reconstruction error, and average CE are not circuit identifications. They may price an
already identified object or supply a matched-capacity control; none is the discovery target here.

## What changed since 13:33

- Rung488 passed every registered discovery and held-out clause, validating exactly the bidirectional T--I midpoint
  graph. A pre-adoption red team found that C is also a strong midpoint donor within T and I targets; every midpoint
  contains the same large native MLP1 state. The graph remains factual, while the special T/I-reader interpretation
  was guarded before adoption.
- Rung489 directly tested that confound. The native-state term predicts T and I better than their registered
  cross-midpoints, so the T/I-specific midpoint interpretation is false. A single all-branch reader also fails because
  C misses the `.90/.45` effect criteria. T and I nevertheless pass strongly at `.97--.98` effect cosine.
- Rung490 prospectively validated the narrower branch signature on previously unopened NATIVE/CURVATURE intervention
  outcomes. T and I retain `.97--.98` native-state prediction, C remains `.84--.88`, and both finite curvature and the
  suffix's nonlinear response reproduce the strict RMS order `T>I>C` in both quarters.
- Rung491 was preregistered, implemented, gated, committed, and launched. It expands the native MLP1 state into exact
  embedding/skip, attention0, fixed MLP0 remainder, MLP0 T/C/I/S, attention1, and numerical-remainder sources. For
  every target, physical singleton and leave-one-source-out suffix recomputations test whether any named source is
  necessary for both T and I. It is still live in the managed GPU runner.
- The full since-02:19 explanation now includes the guarded interpretation, the held-out corrected rule, its exact
  computation, and a percentage graph.

## Does the current route remain highest-information?

Yes, through rung491's result boundary. It asks which real residual source makes the identified T/I response work,
so it can turn an opaque whole-state predictor into a computationally named modulator. It directly advances
computational specification, cross-branch grouping, held-out causal prediction, and stable identification. It is not
a search for a smaller coordinate basis.

However, a rung491 pass is not yet adoption. Its physical arms inject source-level MLP1 output terms into a common
branch-absent background. The next stronger test must edit the selected state source at the MLP1 input in both native
and branch-absent runs. The resulting difference-of-differences asks whether disabling the same reader source changes
T and I as predicted while preserving C/S and unrelated behaviors. This moves from output attribution to an
executable interface edit.

## Confound audit

- **Common-background dominance:** rung489 already killed midpoint donor specificity. Rung491 must name a source
  within the common native state; it cannot relabel the whole state as a T/I circuit.
- **Normalization bookkeeping:** RMS normalization multiplies all unnormalized residual sources by one scalar per
  token. A separately retained numerical remainder makes their deployed normalized-state sum exact; it may not be
  selected as semantics.
- **Suffix nonlinearity:** source necessity is measured by full physical suffix recomputation. Singleton CE effects
  are never algebraically added to predict a multi-source CE effect.
- **Post-selection:** source thresholds and the complete source vocabulary were frozen before discovery. Validation,
  if licensed, must reproduce the entire shared source set; no best-source replacement is allowed.
- **Shared token difficulty:** comparisons hold the target branch fixed and contrast actual source terms; sixteen
  shifted-position source states are controls for generic magnitude or easy-token effects.
- **Dead interventions:** every non-numerical source arm must produce a nonzero physical effect. The numerical term is
  measured separately because deployed rounding may erase it.
- **Precision and temporal drift:** float32 source closure, roundoff-derived BF16 bounds, in-process baselines,
  deterministic fingerprints, and exact call counts remain required. No old sub-.1-nat cross-process bridge is used.
- **Causal-interface gap:** a necessary injected MLP1 output term is not automatically the same as an upstream source
  intervention. The planned dual-state input edit is required before selective-manipulation language.
- **Evidence scope:** rung490 and any conditional rung491 validation are held out by intervention outcome on an
  already used corpus, not new-corpus OOD evidence.

## Genuinely different next moves, ranked

1. **Follow rung491's frozen fork.** If a named shared source validates, disable that source at MLP1 input in both
   native and branch-absent states and measure T/I difference-of-differences plus C/S and unrelated-behavior
   preservation. This advances extraction and selective manipulation. It dies if the exact state-edit identity fails,
   either T or I does not change in the registered direction, or unrelated effects move comparably.
2. **If rung491 finds no stable named source, preserve the whole native-state interface and test the site-graded T-I
   difference path.** Remove the T-minus-I component at attention1 and MLP1 separately. This advances within-path
   splitting and asks whether the empirically shrinking difference signal is genuinely consumed by depth. It dies if
   both sites show proportional effects or shifted/random matched controls behave the same.
3. **Cross-context source interchange after a named-source pass.** Swap the chosen source state between examples
   matched on the branch change but differing in context, and predict the target response from the donor source. This
   advances reuse and OOD-style factorization. It dies if donor identity adds nothing beyond token/position controls
   or the effect does not follow the swapped source.
4. **Equality-query single-index causal readout.** The parameter-free singleton-sum ordering is strong on code and
   moderate on natural text; a held-out monotone readout can test whether three MLP components compose through one
   scalar variable. This advances composition, but is secondary to the live MLP0/MLP1 decision. It dies if held-out
   subset effects do not beat permuted-site and additive baselines.
5. **Predictive-state causal quotient across the 62 circuits.** Group internal components only when their signed
   intervention-response vectors remain interchangeable under held-out downstream readers. This is the broader
   cross-module route if named residual sources stay distributed. It dies if group membership is unstable across
   row/circuit splits or selective interchange fails.

Rank reduction and quantization are rejected from this list because none of these open decisions asks how many
coordinates reproduce variance. The decisions ask which computations are the same, which are different, and what
can be selectively edited.

## Live continuation

Rung491 is confirmed live under the managed runner, with sustained GPU utilization rather than only a queue marker.
Its receipt selects move1 or move2. At landing it must be scored exactly as preregistered, recorded in the ledger and
backlog, and followed by an append-only board claim plus an actually started successor. The next mathematical review
is due after the 13:10 review's three-hour interval.
