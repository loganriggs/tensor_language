# Hourly strategic review — 2026-08-29 05:40 UTC

## What changed since the 05:18 review

This interval produced three concrete advances, but only one new model result:

1. The recovered Family-F experiment is now fully interpreted.  Selecting 512 MLP3
   products by downstream consequence improves fit-distribution suffix KL relative to
   random support and the earlier Family-A support.  It still fails the registered
   local-write reconstruction gate by a large margin.  Keeping the native decoder is
   better downstream than refitting the decoder, even though refitting improves local
   write error.  This is evidence that local Euclidean reconstruction and behavioral
   equivalence can point in opposite directions.
2. A prospective test is frozen for the native-decoder anomaly.  It tests ordinary
   held-out substitution, both signs of the observed error, and finite edits in four
   frozen physical directions.  This can distinguish a transferable behavioral port
   from one-sided downstream compensation.  It has not run and earns no new credit.
3. The real 36-site shared-output factorization now has an executable,
   source-closure-ready
   runner and adversarial CPU tests.  It fits every program choice before evaluation
   rows are deserialized and replaces all 36 native attention/MLP writes.  This is an
   implementation advance, not an E2 result; the GPU experiment has not run.

The proposed finite L8→L11→L14 transport triangle was also audited.  Its computation
is genuinely different from the failed infinitesimal tangent-rank screen: it predicts
a finite response through two learned maps without reading the true intermediate.
It cannot launch on the existing cached receipt, because the nominal 96/96/192 rows
come from only 33/33/105 unique documents.  Treating chunks as independent documents
would make confidence estimates invalid.  A new unique-document cache and a complete
source-closed lifecycle are required.

## How much of the model is actually explained

The strict answer has not increased:

- **Structural accounting:** 36/36 attention and MLP write interfaces can be captured,
  replaced, and composed by the execution facade.  This says where computation enters
  the residual stream, not what it means.
- **Strict parameter removal:** 5.3481% of the original parameter storage has earned
  a registered removal claim.
- **Named causal next-token-loss effect:** 10.923% is assigned to named causal pieces;
  89.077%, or 4.72714 nat of the registered loss stake, remains unnamed.
- **Terminal useful circuits:** 0/68 candidates has passed the full extraction,
  selective-removal, and OOD-transport standard.

The 68 items are candidate terminal behavior/circuit cells: a particular behavior
probe paired with a particular causal component or path.  They are not 68 discovered
features, and none currently counts as a finished circuit.

## Largest gaps

1. **No autonomous whole-model replacement is close to native behavior.**  The best
   context-free map needs native one-token streams; when fed its own recursively
   generated streams its CE deficit rises to about 1.1–1.27 nat.
2. **Local and downstream objectives conflict.**  Family-F decoder refitting lowers
   physical write error but worsens suffix KL.  We do not yet know whether native-Down
   success transfers to fresh documents or survives two-sided and finite edits.
3. **Shared compression is mathematically ready but empirically unmeasured.**  The
   common-output reduced-rank regression may save storage, but only whole-program CE
   at exactly matched storage can decide whether that shared basis is useful.
4. **Finite compositional state tests lack valid rows and lifecycle.**  The current
   transport triangle cannot provide independent-document uncertainty.
5. **No behavior-anchored circuit has closed the loop.**  We still lack one example
   that predicts a behavior, can be extracted, and can be selectively removed with
   limited collateral CE on natural-text/OOD data.

## Candidate actions considered and pruned

The criteria are: expected information gain, direct causal relevance, ability to
compose into a whole model, clear falsification, GPU cost, and duplication of finished
work.

- More MLP3 local decoder fitting is pruned.  The registered refit already improves
  local NRMSE while harming downstream KL; repeating that objective cannot resolve the
  anomaly.
- More recursive stream-map iterations are pruned.  Two direct tests failed by a wide
  margin and the iterative refit was much worse.
- Sparse rotation of a shared basis is deferred.  A rotation cannot create predictive
  value if the underlying shared subspace fails the matched-storage CE test.
- Launching the current transport triangle is pruned until unique-document rows and a
  source-closed lifecycle exist.
- Broad semantic labeling of rank-64/512 coordinates is deferred.  A dense shared
  subspace is identifiable only as a projector; naming arbitrary rotated columns would
  be gauge-dependent.

## Top five current priorities

1. **Run the real 36-site shared-output reduced-rank experiment.**  It is the highest
   information-per-GPU-minute action because it directly tests whether one stored
   output dictionary can replace 36 independent ones at equal storage and whole-model
   CE.  It is falsifiable on three already frozen discovery roles and uses no native
   component calls in the compiled arms.
2. **Run the fresh-document native-Down behavioral-port test.**  This attacks the most
   informative Family-F anomaly and separates ordinary transfer, two-sided downstream
   nullity, and finite edit transport.  It can turn a fit-only compression into a
   restricted editable program or decisively classify it as compensation.
3. **Repair and run one finite transport composition.**  Freeze genuinely unique
   documents, then test whether two learned maps predict a sealed composed response.
   This is the cheapest remaining route to a predictive state that is defined by what
   downstream computation preserves rather than by local activation MSE.
4. **Close one terminal behavior circuit.**  Use a sharply specified behavior with
   held-out templates and natural-text replication, then require extraction plus
   selective removal.  This supplies an external validity test for competing notions
   of simplicity.
5. **Try sparse/hierarchical coordinates only after priority 1 passes.**  If the shared
   projector preserves CE, rotate it using a frozen dictionary/sparsity objective and
   test cross-role support stability and selective interventions.  If it fails, this
   branch is eliminated before semantic labeling work begins.

## Highest-priority action executed

The shared-RRR runner now freezes 24 arms: global, attention/MLP-typed, same-rank
independent, exactly equal-storage independent, a direct global-rank-494 versus
typed-rank-481 pair, and two legacy wiring controls.  Its exact common-ridge theorem is
covered by known-answer tests; the autonomous dispatcher is tested to use token
embeddings/table lookup only and a zero inert attention-value sentinel; source,
checkpoint, row, price, call, resource, and receipt lifecycles are replayable.  The
combined focused CPU suite passes 43 tests.

The GPU remains occupied by the independent matched-map frontier, which has published
only partial logs as of this review.  No second GPU job is started concurrently.  Once
the runner completes independent outcome-blind audit, it will be committed/pushed and
source-closure replayed; its experiment is next in the GPU queue.  Until a real result
and receipt exist, E2.1–E2.3 remain unchecked.

## Update at 05:53 UTC — v1 implementation failure, no E2 result

The independently audited v1 authority opened and the runner completed the fit-only
native table capture and factorization.  At the first evaluation-role metric lookup it
failed because CUDA token IDs indexed a Boolean coverage mask that remained on CPU.
V1 therefore published authority plus terminal failure, and no arm metric, result, or
receipt.  This is not evidence for or against a shared basis and E2 remains unchecked.

The recovery is narrow and testable: move the immutable coverage mask to the token
device before indexing it.  A fresh v2 wrapper hash-binds and semantically replays the
spent v1 authority/failure and changes no rows, arms, objective, ranks, prices, gates,
or call schedule.  A non-CPU-device contract test now reaches this boundary without
loading the model or data.  V2 may launch only after its amendment, corrected source,
wrapper, and tests are committed/pushed, source closure replays, and an independent
outcome-blind audit returns GO.
