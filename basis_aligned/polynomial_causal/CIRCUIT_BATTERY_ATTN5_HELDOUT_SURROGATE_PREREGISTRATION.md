# CIRCUIT BATTERY — ATTENTION 5 HELD-OUT SURROGATE (preregistration)

Registered 2026-09-04 05:29Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_attn5_heldout_surrogate`. Script: `ops/circuit_battery_attn5_heldout_surrogate.py`.
IMMUTABLE: any change gets a new document, not an edit.

## Why this exists

§2832 measured attention 5's write as 98.1% one direction and found every low-dimensional surrogate cheap: against a 2.1996-nat
ablation cost, `RANK_32` cost .0352 nats, `MEAN` .1192, and the parameter-free `HEADS_57` .0883. But **the rank bases and the mean were
fitted on the very 32 documents they were scored on**. My §2832 preregistration argued that in-sample fitting was conservative because
it favoured the arm I expected to lose; the in-sample arm WON, so that argument is void and §2832 explicitly declines to claim the write
is cheap to approximate. This rung supplies the missing control: fit on one document set, score on a DISJOINT one, and then on a
different corpus entirely.

Fixed before the run: layer 5, kept heads {5, 7} (from §2831), ranks {8, 32, 128}, 48 natural documents to fit, 48 DISJOINT natural
documents to score, 48 code documents as the cross-corpus set, chunk 8. Sign convention: d_ce = CE_arm − CE_NATIVE in nats, POSITIVE =
the arm HURTS, each corpus scored against its own native CE. **Not the §312 frontier's L2 (CE added above the real model by an installed
approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735); nothing installs; energy bases remain negative controls and
metric-constructed spans stay CLOSED (§2118 lineage).**

## Predictions

```
BARS  = {heldout_rank32: .10, cross_rank32: .25, heads_gap: .05, cos_within: .90, cos_cross: .70, ce_tol: .01}
NULLS = {heldout_rank32_ge: .50, cross_rank32_ge: 1.00, heads_gap_ge: .20, cos_within_le: .30}
```

**pred_a_heldout_rank_transports** — the rank-32 basis fitted on the fit documents costs ≤ .10 nats on the DISJOINT natural documents.
*Worked example:* §2832's in-sample rank-32 cost .0352; if the write's low-rank structure is a property of the component rather than of
the sample, the held-out cost lands .03–.10, and if §2832's cheapness was fitting, it lands .5–2.2 (approaching the 2.20 of deleting the
component). This is the clause that decides whether §2832's headline survives. Null: ≥ .50.

**pred_b_cross_corpus_rank_transports** — the same basis costs ≤ .25 nats on CODE documents. *Worked example:* code is a different
distribution, so some degradation is expected even for a real structure: .05–.25 if the direction is intrinsic to the component,
≥ 1.0 if it is a property of natural text. Null: ≥ 1.00.

**pred_c_parameter_free_heads_transport** — `|d_ce(HEADS_57) on held-out natural − .0883|` ≤ .05 nats, where .0883 is §2832's in-sample
value. *Worked example:* HEADS_57 has NO fitted parameters, so it cannot overfit and the held-out value should match to sampling noise,
~.00–.03. A large gap would mean the two document sets differ enough that no cross-set comparison in this rung is readable, which would
qualify pred_a and pred_b as well — that is why this clause is here. Absolute difference of two damages in the same units.
Null: ≥ .20.

**pred_d_the_top_direction_is_stable** — |cosine| between the top singular direction of attention 5's write fitted on the fit documents
and (i) on the held-out natural documents ≥ .90 AND (ii) on the code documents ≥ .70. *Worked example:* if 98.1% of the write's energy
is one intrinsic direction, two independent estimates of it agree at .95–1.00, and a cross-corpus estimate at .8–1.0; if the direction
is sample-specific, ~.0–.3 (two random directions in R^1152 have |cos| ≈ .03). Absolute cosines, bounded in [0, 1]. Null: within-corpus
≤ .30.

**pred_e_instrument_reproduces_native_ce_matched** — |this rung's manual forward CE on the first held-out chunk − the model module's own
CE on the SAME chunk| ≤ .01 nats. *Worked example:* the same computation on the same data, ~1e-4. §2832's version of this check compared
a 32-document average against an 8-document module call and scored FALSE on a mismatch of sample sets rather than of computations; this
one compares matched chunks, which is what I should have written there.

## Stated null

The low-rank cheapness of §2832 was fitting: held-out rank-32 costs ≥ .50 nats, the cross-corpus version ≥ 1.00, and the top direction
does not transport. §2832's parameter-free HEADS_57 number would still stand, and the write-is-rank-1 claim would be reduced to a
statement about a particular 32-document sample.

## Price

Two corpora × (native + zero + heads + mean + 3 ranks) at 48 documents each, plus three basis-fitting passes and one matched instrument
chunk. Literal budget: ≤ 900 GPU document-forwards, 0 backwards, **~198,784 declared fitted parameters**, < 4 GPU-minutes.

## What this does NOT claim

The held-out set is a different slice of the same frozen natural cache, so pred_a is a sample-transport test, not a
distribution-transport one; pred_b is the distribution test and is registered with a looser bar for that reason. No claim about
installing anything, and no L2 numbers. Does not satisfy Codex's four-phase integration contract; updates no circuit record.
