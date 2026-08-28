# Hourly strategic review — 2026-08-28 18:15 UTC

## Outcome of this review

The highest-priority missing interface is now implemented on CPU: every one of the
64 program-bearing actions has exactly one source-closed path from its semantic name
and full scored row to the runtime transaction that executes it. The four remaining
actions are the already implemented native/deployed baselines.

This does **not** change a scientific score or authorize the final run. It removes a
way that an invalid score could be produced by running RR in place of a hybrid, using
the wrong cross map, changing only the target token, or constructing a runtime trace
independently of the named action.

## How much of the model is explained?

There is no honest single percentage because the existing numbers measure different
things. The current ledger is:

| Meaning of “explained” | Current result |
|---|---:|
| Module has some tested structural surrogate | 36/36 modules |
| Whole-program storage certified removable for its registered consequence | 5.3481% |
| Older behavior assigned human-readable labels | 32.1% ± 6.4% |
| Strict named causal CE headroom recovered | 0.57968/5.30682 = 10.923% |
| Strict named causal CE still unexplained | 4.72714 nats |
| Newly admitted recovery of the current compiler's +0.8976-nat CE gap | 0% |
| Final coupled early-MLP actions evaluated | 0/68 |

The 36/36 line is coverage, not reverse engineering: it says every module has been
touched by a structural test, not that every module's function or composition is
understood.

## Largest remaining gaps

1. **Composition:** early MLP effects interact strongly. Prior held-out Shapley
   decompositions put interaction mass around 43--64%, and MLP2 can change from
   harmful to useful after MLP0/1 are repaired.
2. **Residual CE:** no newly admitted package recovers any of the current
   +0.8976-nat executable gap. Local reconstructions and analytic ceilings are not in
   this currency.
3. **Consumer compatibility:** the newest attention-scale arc shows that a locally
   plausible write can send a later live layer into a pathological regime.
4. **Edits and removal:** no complete replacement currently predicts the registered
   finite code/logit interventions and selective collateral together.
5. **Real OOD:** the existing alternate skips are same-corpus held-out slices, not a
   broad distribution shift. Frequency bins are useful stress tests but not a full
   OOD claim.
6. **Semantics:** MLP0's approximately rank-64 continuous lexical code is executable,
   but most axes or shared atoms still lack stable human-readable meanings.

## New evidence considered this hour

S1823 tested whether attention gains measured inside each partial composition are a
principled repair. It helps the shallow B0 boundary, raising gap recovery from
37--40% to 53--57%. It fails at the harder B3 and B5 boundaries: recovery remains
about -23% and -14%, while the earlier global gains happen to obtain about +10--12%.
The corrected curve is not monotone, and all three preregistered predictions fail.

The consequence is narrow but important: local norm matching is not a compositional
certificate. Exact foldable gauge changes remain genuinely free; a fitted scalar is
only a cheap function-changing program component and must survive complete held-out
composition, CE/KL, OOD, and edit tests.

Two concurrent diagnostics completed while this CPU slice was being verified. S1824
iterated gain calibration upward to its fixed point. It improves B0 recovery to
61--65%, but reaches only about 12% at B3/B5 and fails its registered five-point
improvement bar. This closes magnitude as a sufficient account of the deep-prefix
failure. S1825 then rejects a global direction story:
mean live-versus-corrected cosine at B3 is 0.770, not below the registered 0.50 bar,
and it does not decay monotonically with height. It nevertheless finds localized
sign reversals at attention L9: cosine -0.134 at B3 and -0.628 at B5. The remaining
failure is therefore localized nonlinear/content interaction, not one global gain or
one globally rotated stream. Its cross-run anchor reproduces the earlier
single-interface direction at +0.9979 versus +0.9990.

## Candidate actions considered and pruned

| Candidate | Information/causal value | Composition and falsifiability | Cost/redundancy | Decision |
|---|---|---|---|---|
| Bind all named actions to physical runtime traces | Very high; every final comparison depends on it | Fails on action, row, program, route, or receipt substitution | CPU-only; not previously closed | **1** |
| Execute paired finite edits | Very high for extraction/removal and causal abstraction after scale/direction accounts fail | Tests whether compressed state preserves intervention response through nonlinear consumers | Moderate forwards; not replaceable by CE | **2** |
| Measure all 18 later consumer norms | Integrity value after S1821--S1824; no longer a sufficient explanation | Locates scale failures but cannot certify compatibility | One instrumented run; already registered | **3** |
| Add frequency bins plus agreement/KL aggregation | High for rare tokens and functional faithfulness | Common rows make differences falsifiable | Cheap reductions on existing forwards | **4** |
| Assemble and independently audit the one-shot bundle | Required for a valid result | End-to-end closure and one-use lifecycle | CPU-heavy tests, low GPU cost | **5** |
| Another scalar/norm sweep | Low after S1823 | Local success would still not imply composition | Redundant | Pruned |
| Standalone MLP0 HOSVD/CP/SVD | Moderate structural interest | Cannot yet show downstream equivalence or edit transport | Duplicates rank/factorization curves | Deferred |
| Standalone SAE semantic labeling | Useful only if atoms compose downstream | Easy to overinterpret reconstruction features | Does not close current causal gate | Deferred |
| Another rank-only accuracy curve | Low | Does not distinguish extraction, KL faithfulness, or editing | Existing rank-1/16/64 evidence suffices | Pruned |

## Ranked top five

1. **Source-closed action-to-runtime binding.** Highest dependency value, causal
   relevance, and falsifiability; CPU-only. Executed in this review.
2. **Paired edits.** After global magnitude and direction stories fail, finite
   response transport is the highest-value test of whether the small code preserves
   the nonlinear distinctions used by the suffix.
3. **All-consumer norms.** Retain these as preregistered integrity/localization
   diagnostics, not as an explanatory or selection criterion.
4. **Frequency/agreement/KL aggregation.** Necessary to separate top-1 extraction
   from functional distributional faithfulness and expose rare-token failures.
5. **Bundle assembly and independent audit.** Converts the pieces into one legal
   one-shot experiment; it becomes informative only after items 2--4 exist.

## Highest-priority action executed

Implemented a final-only binding with these properties:

- all 64 program-bearing actions derive their runtime route, control, teacher kind,
  program hash, MLP2 background, and batch schedule from the sealed materialization;
- QQ uses physical L topology while retaining the semantic `inherited_q` identity;
- the four hybrids retain their exact mixed site sources instead of silently becoming
  RR, S0, or S1;
- true transport, zero cross, and all 20 false-pair cross maps remain distinct;
- shuffled and fit-mean programs retain their semantic control identities;
- the complete 513-token scored row remains bound before trace construction;
- the broker must be sealed to the same final run context;
- the lower runtime receipt is hashed back into a tensor-free action receipt.

An exhaustive test over all 64 program actions found a genuine omission:
`zero_A/T` was materialized but not licensed by the final runtime identity. It is now
licensed **only on final T traces**; it remains illegal during fit and validation.
Seven final-only semantic controls have explicit tests proving they cannot leak into
validation.

Focused runtime/action/adapter tests pass **74/74**. The complete suffix/observed
regression suite passes **269/269 in 102.66 seconds**. No fit, validation, final role,
model outcome, or GPU forward was opened for this implementation.

## Exact next boundary

The next safe action is the registered paired edited/unedited response path and its
physical call ledger. Then come the 18 consumer-norm integrity reductions,
nine-bin/agreement aggregation, complete action aggregation, and independent audit.
There is no external data, cache, checkpoint, `rspd`, or GPU blocker.
