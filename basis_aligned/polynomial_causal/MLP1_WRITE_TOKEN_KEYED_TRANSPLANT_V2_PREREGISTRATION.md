# Preregistration — token-keyed transplant v2: overlap-maximized donors, four-action format (parallel lane)

Date: 2026-09-02 18:35 UTC
Owner: Claude (parallel probe lane)
Status: frozen before any v2 outcome exists

## Lineage and what changed

V1 (§2620) landed strong null via pred_a: its XOR-1 donor pairing supplied only
474/482 matched positions per half (~1.9/doc) against a ≥500 floor, with at least
one zero-match batch. This is the registered redesign, with the supply problem
fixed by measurement (board 18:02, CPU-only on token rows, no outcomes): greedy
best-overlap donor selection within each half supplies 1,776/1,764 matched
positions per half, per-doc minimum 3/5, zero degenerate documents. The scientific
question is unchanged and still OPEN: is the MLP1 write-adjustment TOKEN-KEYED —
portable exactly where donor and recipient share the same token at the same
position (motivated by T being the token-only branch)? §2618 (whole-write
transplants actively harmful) and §2620's descriptive pro-hypothesis sign pattern
stand behind it; neither is evidence here.

Design changes from v1, all frozen now: (1) donor(d) = the document in d's own
half (0:250 or 250:500) maximizing same-position token matches at positions ≥ 1,
ties to the lowest index, donor ≠ d — computed in-run from token rows only,
deterministic; (2) two-pass collection — pass 1 caches every document's per-token
CE and float32 MLP1 adjustment for T and I, pass 2 runs the edited forwards with
cached donor adjustments (identical forward count; adjustments are read, never
refit); (3) a COMPOSE arm per the four-action receipt format (§2621's grading of
my v1 bundle, accepted 18:02); (4) floors set from the measured supply.

## Arms (all named; branches b ∈ {T, I}; all edits via rung493 `_merge_forward`
## M_ONLY with layers 2–17 recomputing)

Per document d with donor m(d), MATCH mask = same token, same position, pos ≥ 1;
MISMATCH_SAMPLE = seeded (20260903) same-count sample of non-matched positions:

- NATIVE, ABSENT_b (remove — rung493 exact branch removal);
- OWN_MATCH_b (restore): write = M_b, with M_N on the match mask;
- DONOR_MATCH_b (substitute, keyed): M_b + adj_b[m(d)] on the match mask;
- DONOR_MISMATCH_b (substitute, off-key control): same donor adjustment on the
  mismatch-sample mask;
- COMPOSE_b (compose): in the BOTH-branches-absent trajectory (T and I removed),
  apply DONOR_MATCH_b's edit; its recovery is scored against the same x_b as
  DONOR_MATCH_b to ask whether keyed restoration survives removing the other
  branch's service.

10 forwards per batch of 4 (1 native + 2 absent + 1 both-absent + 2×3 edited);
125 batches; the both-absent trajectory uses rung493's exact branch arithmetic
(removing T+I jointly), replayed against its registered identity bounds.

## Scoring

x_b = CE(ABSENT_b) − CE(NATIVE) on the scored mask; recovery_e = CE(ABSENT_b) −
CE(arm) on that mask; aligned recovered fraction f = <recovery,x>/||x||² (higher
= more restored; negative = worse than nothing) and cosine. Per branch × half.
COMPOSE uses the same x_b and mask as DONOR_MATCH.

## Frozen predictions

### pred_a — exact, lawful, live instrument (floors from measured supply)
Hashes match (rung493 source/result, v1 result showing its pred_a false +
strong null, this prereg). Rung493 identity suite at its registered bounds; the
both-absent trajectory passes the same replay bounds. Calls exact: 125 native +
250 absent + 125 both-absent + 750 edited = 1,250 forwards. Matched positions
≥ 1,500 per half (measured 1,776/1,764); every document ≥ 2 matches (measured
min 3); every edited write differs from its base (edit RMS > 0 in every batch);
OWN_MATCH f > 0 in all four branch×half cells.

### pred_b — restoration is token-keyed
For both branches, both halves: f_DONOR_MATCH ≥ .25 × f_OWN_MATCH AND
f_DONOR_MATCH ≥ f_DONOR_MISMATCH + .25 AND f_DONOR_MATCH ≥ 0.

### pred_c — off-key toxicity reproduced
For both branches, both halves: f_DONOR_MISMATCH ≤ 0.

### pred_d — keyed restoration composes across branch removal
Scored only if pred_b holds (otherwise reported descriptively): for both
branches, both halves, f_COMPOSE ≥ .5 × f_DONOR_MATCH with the same sign.

Descriptive regardless: T-vs-I keyed-restoration comparison (hypothesis predicts
T ≥ I, stated not scored), per-doc match-count distribution, donor-overlap
distribution, full arm tables, v1-vs-v2 supply comparison.

## Strong null and interpretation

Strong null fires if pred_a, b, or c fails (d is conditional). Null: token
identity does not unlock portability even at 3.7× supply — the write is
context-bound below token grain; §2618 stands; the mechanism routes to content
decomposition of the write, not to bar changes or a third supply redesign.
Pass: first conditional-portability statement (keyed positions), licensing a
registered cross-corpus token-matched validation only. No compression/rank claim
on any route.

## Literal price

1,250 full-model forwards (~110s), single phase, no validation or sealed roles.
Bundle: per-token CE (float32) for named arms + masks + donor map; sufficient
statistics for shifts none (no shift arms in v2 — the off-key sample is the
position control, same edit budget). Zero deployed parameters.

## Instrument addendum — v2b, 2026-09-02 19:02 UTC (price arithmetic only)

The v2 receipt is preserved as instrument-invalid: the arms section above
registers FOUR edited arms per branch (OWN_MATCH, DONOR_MATCH, DONOR_MISMATCH,
COMPOSE) but the price paragraph counted three, registering 1,250 forwards and
750 edited where the registered design costs 12 per batch: 125 native + 250
absent + 125 both-absent + 1,000 edited = 1,500 full-model forwards. The
implementation executed the registered arms faithfully; every other pred_a
clause passed (supply 1,776/1,764, min 3/doc, identity suite clean). v2b
corrects ONLY this arithmetic: expected merge forwards 1,000, expected total
1,500. No bar, arm, floor, mask, seed, donor rule, or scoring clause changes.
v2b writes a distinct receipt/bundle namespace; the v2 descriptive tables
cannot pass retroactively. Honest expectation, stated before v2b runs: those
descriptive tables make the scientific null (pred_b failing) the likely
outcome — v2b purchases a lawful verdict. v2b also repairs a template defect:
next_step now reports an instrument-repair route when pred_a is false, rather
than the scientific routing an A-false receipt does not license.
