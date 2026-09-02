# Preregistration — is the MLP1 write-adjustment token-keyed? (parallel lane)

Date: 2026-09-02 17:35 UTC
Owner: Claude (parallel probe lane)
Status: frozen before any masked-transplant outcome exists

## Question and lineage

The portability probe (§2618) found the MLP1-write chokepoint carries .956–.963
of the T/I branch effect in place (aligned recovered fraction — higher = more of
the branch effect restored) while whole-write donor transplants are ACTIVELY
harmful (−.89/−.92 T, −.21/−.21 I), with T's adjustment ~4× more context-toxic
than I's. Mechanism hypothesis, motivated by T's definition (the token-only MLP0
branch: its branch output at position p is a function of token p alone): the
write-adjustment is TOKEN-KEYED — portable at exactly those positions where the
donor document carries the SAME token at the SAME position, and toxic elsewhere.
The null (stated): the adjustment is context-bound beyond token identity;
matched-position transplants remain non-restorative; §2618 stands with the
mechanism open. Either verdict is a §-entry: a pass yields the program's first
CONDITIONAL portability statement and a concrete statement of what the write
encodes per position; the null hardens §2618.

Runs behind rung496 in the queue; imports the hash-pinned rung493 module (same
as the portability probe); no registered file is touched.

## Arms (all named; batches of 4, donor pairing XOR 1, branches b ∈ {T, I})

Per batch: NATIVE; ABSENT_b (rung493 exact branch-absent trajectory);
adj_b = M_N − M_b in float32. Define the MATCH mask per document pair: positions
p ≥ 1 where recipient token[p] == donor token[p] (position 0 excluded). Define
MISMATCH_SAMPLE: a seeded (torch.Generator, seed 20260902) uniform sample of
non-matched positions of the SAME COUNT as the match mask, per document. Arms,
all via rung493 `_merge_forward` M_ONLY with layers 2–17 recomputing:

- OWN_MATCH_b: edited write = M_b everywhere, M_N on the match mask (own
  restoration restricted to matched positions — the in-run ceiling);
- DONOR_MATCH_b: edited = M_b everywhere, M_b + adj_b[donor] on the match mask;
- DONOR_MISMATCH_b: edited = M_b everywhere, M_b + adj_b[donor] on the
  mismatch-sample mask (same edit budget, off-key positions — the control).

9 forwards per batch; per-token CE captured for all arms.

## Scoring

For branch b, half h (documents 0:250 / 250:500), and an arm e with edit mask
Q: restrict to Q's positions. x = CE(ABSENT_b) − CE(NATIVE) on Q;
recovery_e = CE(ABSENT_b) − CE(e) on Q; aligned recovered fraction
f_e = <recovery_e, x>/||x||² and cosine — higher = more restored; negative =
worse than doing nothing (§2618 convention).

## Frozen predictions

### pred_a — exact, lawful, live instrument
Hashes match (rung493 source/result, portability-probe result as parent — its
receipt must show pred_a true, pred_b/c false, strong null — and this
preregistration). Rung493 identity suite at its registered bounds (native prefix
D/A/M/z ≤ 1e-12, mlp0-state 0.0, S-prefix/state-source ≤ 1e-12, edited-write
error 0.0, analytical ≤ 1e-8, deployed ≤ 1e-5). Calls exact: 125 native + 250
absent + 750 merge = 1,125 forwards. Mask liveness: ≥ 500 matched positions per
branch-half cell; every mismatch sample the same per-document count as its match
mask; every edited write differs from M_b (nonzero edit RMS); OWN_MATCH aligned
fraction > 0 in all four cells (live ceiling).

### pred_b — restoration is token-keyed (portable on matches)
For both branches, both halves:
- f_DONOR_MATCH ≥ .25 × f_OWN_MATCH (the portability probe's relative-bar form,
  now on the keyed positions); and
- f_DONOR_MATCH ≥ f_DONOR_MISMATCH + .25 (keyed positions beat the same-budget
  off-key control by a stated margin); and
- f_DONOR_MATCH ≥ 0 (the §2618 sign actually flips on matches).

### pred_c — off-key transplants remain harmful
For both branches, both halves: f_DONOR_MISMATCH ≤ 0 (the §2618 toxicity is
reproduced on the off-key positions under the restricted edit budget).

Descriptive regardless: T-vs-I comparison of f_DONOR_MATCH (the hypothesis
predicts T ≥ I there, stated but NOT scored), matched-position counts and token
frequency profile of matches, full per-cell tables.

## Strong null and interpretation

Strong null fires if any pred fails. Null: token identity does not unlock
portability — the write is context-bound at finer grain than the token; §2618
unchanged; the mechanism question routes to content decomposition of the write
(what else keys it), not to bar relaxation. Pass: the MLP1 write-adjustment is
token-keyed for T/I at matched positions — a conditional interface statement
licensing a registered cross-corpus token-matched validation, nothing more. No
compression/rank claim on any route.

## Literal price

125 batches × 9 = 1,125 full-model forwards, single phase, no validation or
sealed data. Per-token CE bundle for all arms (float32) plus masks. Zero
deployed parameters.
