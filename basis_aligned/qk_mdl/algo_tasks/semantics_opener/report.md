# Semantics verification: the "pending-opener" content channel in bilin18

Model: `bilin18` (gpt2-bilinear-sqrd-attn, 18 layers x 9 heads, D = 1152, no
softmax, pattern = (q1.k1)(q2.k2)/128^2 causal unnormalized, bilinear MLPs).
Channel: the r-dimensional DAS subspace of the residual stream ENTERING layer 13
found in `algo_tasks/bracket/` (head L13H8's steering subspace, final-position,
query-side).

**Hypothesis under test (the NAME as code):** "the channel's activation is a
function of the number of currently-unclosed openers (bracket depth /
open-quote parity) at that position."

Data discipline: all fitting/calibration on `data_fineweb_cooc_tokens.npy`
rows 0-360; ALL final numbers on the held-back audit slice
`data_fineweb_tokens.npy` rows 448:600 (77,824 scored tokens) or the bracket
battery's held-out 20 pairs. Scripts `s0_channel.py` ... `s5_redteam.py`;
results in `channel.json, readout.json, calib.json, gate.json, dial.json,
extraction.json, redteam.json`.

**Verdict up front (honest version):** the name survives only in a weakened
form. What the channel verifiably carries is a *type-blind, recency-weighted
"an opener is currently pending" signal, with LOW activation = open*: it is
graded but saturates by depth 2 (it is not a counter), it barely responds to
`{`, it decays over ~64-128 tokens while a bracket is still open, and it does
not fully reset after the bracket closes. Substituting the coded state for the
channel passes its controls everywhere (coded beats zero/mean/shuffled placebos
on both natural text and the battery), but a static depth code recovers only
half of what the exact channel value does; adding recency to the code closes
most of that gap (0.28 -> 0.43 of a 0.56 exact reference, r = 1).

## 0. Channel re-derivation (s0, channel.json)

DAS at layer-13 entry, final position, exact recipe/seeds of bracket/s3:
held-out recovery **0.564** (r = 1; paren 0.606 / quote 0.522) and **0.872**
(r = 4) — reproduces `bracket/das.json` to 4 decimals. Random-subspace controls
0.001. The r = 1 direction lies 0.979 inside the r = 4 subspace and 0.965
inside the old r = 16 'both' subspace: everything below probes the same object
the bracket study found.

## 1. Readout: activation vs coded state (s1, readout.json)

Coded state (independent knowledge, byte-level tracker over raw tokens): depths
of `( ) [ ] { }`, ASCII `"` parity, curly-quote depth; state at position t =
after token t. Fit on cooc rows 0:240, R^2 evaluated on cooc rows 240:300
(expl) and the audit slice.

| form (a1 ~ .) | R^2 fit | R^2 expl | R^2 audit |
|---|---|---|---|
| paren depth | 0.026 | 0.041 | 0.044 |
| total bracket depth | 0.024 | 0.039 | 0.041 |
| any-open binary | 0.049 | 0.056 | 0.037 |
| quote-any | 0.021 | 0.016 | 0.000 |
| all 5 raw features | 0.044 | 0.046 | 0.039 |
| tot depth + quote-any | 0.046 | 0.061 | 0.044 |
| position only (control) | 0.000 | -0.004 | -0.003 |

Global R^2 is small (~0.04-0.06) — but that is a base-rate artifact, not
absence of signal: only 10-13% of positions are open, and the channel carries
plenty of non-opener variance. The conditional structure on the audit slice is
strong and the position control is null:

- a1 mean at paren-open positions **523 +- 19** vs closed **1566 +- 3**
  (Cohen's d = **-1.31**; LOW = open). Any-open d = -0.67; quote-any d = -0.31.
- Linear decode of paren-open from a1: **AUC 0.80** (audit); any-open from the
  4-dim channel AUC 0.69; quote-any needs the 4-dim channel (AUC 0.74 vs 0.56
  from a1 alone) — the quote part of the signal is mostly in the r = 4
  complement, consistent with the bracket study's quote gap between r = 1
  (0.52) and r = 4 (0.81).
- Natural text has essentially no depth >= 2 exposure (2,602 positions at
  depth 1, < 50 at depth 2 on audit), so depth-vs-binary cannot be decided
  here; see red-team A.

## 2. Meaning gate by substitution (s2, gate.json)

### A. Differential injection at the final position (corrupted battery run)

Metric: recovery of the clean-vs-corrupted closer LOGIT gap (same metric as
bracket/das.json). Held-out 20 pairs, mean +- SE. Coded value = calibrated
linear map of the coded state (cooc-fit, `calib.json`); codedX = the WRONG
(corrupted-stream, all-closed) state; shuf = coded value of a random cooc
position (10 draws/pair).

| injection (r=1) | held recovery | paren | quote |
|---|---|---|---|
| exact channel value (reference) | **0.564 +- 0.023** | 0.606 | 0.522 |
| coded state | **0.280 +- 0.030** | 0.367 | 0.193 |
| coded + recency (red-team F) | **0.434** | 0.471 | 0.398 |
| codedX wrong-state placebo | 0.114 +- 0.036 | 0.217 | 0.011 |
| zero (deletion control) | 0.545 +- 0.017 | 0.559 | 0.531 |
| mean (neutral deletion) | 0.132 +- 0.035 | 0.232 | 0.033 |
| shuffled placebo | 0.133 +- 0.036 | 0.236 | 0.030 |

r = 4: exact 0.872 +- 0.026, coded 0.482 +- 0.038, codedX 0.183, mean 0.207,
shuf 0.210, zero 0.465 +- 0.190 (zero at r = 4 is wildly off-distribution:
paren +1.28 / quote -0.35).

Reading, with the value diagnostics (`r1_value_diag`): on battery prompts the
actual channel value swings from +2568 (corrupted) to **-694** (clean), while
the cooc-calibrated "open" code is only +930 — natural-text calibration is
attenuated because most natural open positions are far from their opener
(see red-team F). The response along the direction is graded and monotone:
+2568 -> +930 -> 0 -> -694 gives recovery 0 -> 0.28 -> 0.55 -> 0.56. This is
also why the "zero" deletion control scores 0.55: **zero is not neutral for
this channel — a = 0 is below the natural open mean (+523), i.e. zeroing
WRITES "opener pending"**. The neutral deletion is mean-substitution (0.13,
indistinguishable from the shuffled placebo). All three deletion/placebo
controls are reported per the known failure mode; coded beats codedX/mean/shuf
by 4-5 SE but recovers only ~50% of the exact reference; the recency-aware code
(calibrated on cooc positions with opener distance <= 12, still pure
code+cooc) reaches 77% of it.

### B. Full-stream substitution on natural text (audit, held-back)

Channel activation REPLACED at every position by the coded value. Paired
per-token dCE vs baseline (SE token-level; sequence-clustered SE similar), plus
top-1 agreement. Baseline CE 3.428.

| condition | dCE (nats) | SE | top-1 agree |
|---|---|---|---|
| r1 coded | **+0.0033** | 0.0003 | 0.982 |
| r1 mean | +0.0046 | 0.0004 | 0.981 |
| r1 shuffled coded | +0.0048 | 0.0004 | 0.980 |
| r1 zero | +0.0076 | 0.0005 | 0.956 |
| r4 coded | **+0.0097** | 0.0005 | 0.962 |
| r4 mean | +0.0147 | 0.0009 | 0.960 |
| r4 shuffled coded | +0.0147 | 0.0009 | 0.960 |
| r4 zero | +0.1267 | 0.0017 | 0.827 |

The coded replacement is the least damaging non-identity intervention at both
ranks (r1: 0.0033 vs 0.0046/0.0048; gaps > 3 SE) — the code reconstructs the
functionally-used content of the channel better than keeping its mean or its
shuffled values. All damage is small (< 0.3% of CE) — no-breakage holds.

Battery boost under full-stream substitution (held-out): base 6.46 +- 0.27;
r1 coded 5.05, shuffled 4.97, mean 4.59, zero 2.91. Caveat stated plainly:
with the channel forced identically in clean and corrupted runs, ~5 nats of
boost survive through redundant paths, and coded vs shuffled is NOT separated
here (5.05 vs 4.97) — the discriminating test is the differential injection in
A, not this one. Zeroing both runs costs 3.5 nats because a = 0 saturates the
"open" reading in both runs.

## 3. Dial (s3, dial.json)

Scale the r = 1 channel by s at all positions. Held-out battery boost and
audit-slice numbers:

| s | battery boost +- SE | natural sep_paren | natural sep_quote | audit dCE +- SE |
|---|---|---|---|---|
| 0.0 | 2.91 +- 0.14 | 3.60 | 1.40 | +0.0076 +- 0.0005 |
| 0.5 | 4.57 +- 0.18 | 4.42 | 1.71 | +0.0017 +- 0.0002 |
| 1.0 | 6.46 +- 0.27 | 5.23 | 2.01 | 0 |
| 1.5 | 8.45 +- 0.38 | 6.02 | 2.30 | +0.0021 +- 0.0002 |
| 2.0 | 10.43 +- 0.48 | 6.78 | 2.58 | +0.0080 +- 0.0005 |

(sep = mean logprob of the closer at open minus closed positions on audit.)
Clean monotone dose-response of closure anticipation in both the synthetic
battery and natural text, with natural CE essentially unchanged (max +0.008
nats = 0.23%). The channel acts as a graded closure-anticipation dial.

## 4. Extraction: standalone predictor (s4, extraction.json)

`predict_closer_boost(token_ids)` — pure python: byte-tracker state ->
delta_hat = c0 + c1 * open-flag, constants calibrated once on cooc rows
300:360 against the model's channel-mediated boost (lp_base - lp_zero of the
closer, r1 channel). Audit results:

- paren: mean channel-mediated delta at open positions -0.91 vs closed -2.54
  (the channel's ')' suppression relaxes by 1.6 nats when open, as coded);
  Pearson r = 0.26 raw, **0.61 balanced** (equal open/closed sample).
- quote: delta -1.73 open vs -2.34 closed; r = 0.15 raw, 0.27 balanced,
  balanced AUC **0.76**.
- Behavioral (base model only): open-flag vs top-5% closer-probability
  positions, AUC 0.60 (paren) / 0.59 (quote).

Failure the demo traces expose plainly: the actual per-position channel effect
is not flat across the open span — it spikes at syntactically plausible
closure points (after the noun in "( which was near the cats|") and dips at
"and"/"the". The code predicts WHEN a closer is licensed (state), not when it
is imminent; the imminence structure lives outside the coded state. Note also
delta here is from a full-stream zeroing run, so per-position attribution is
approximate.

## 5. Red-team (s5, redteam.json)

**A. Depth (is it a counter?): PARTIAL FAILURE.** a1 at stacked openers:
d = 0: +2574, d = 1: -739, d = 2: -1632, d = 3: -1664. Graded beyond binary
(d = 2 is a real further step, 4 SE) but saturated by d = 3 — it is not a
depth counter. The model's own ')' boost does not grow with depth either
(6.69 / 6.14 / 5.73 nats at d = 1/2/3): behaviorally the model also treats
depth mostly as binary here.

**B. Opener types (one channel for all?): FAILURE for `{`, partial for `[`.**
a1 drop and r1-channel share of the closer boost, matched carriers:
`(` drop 3312, share 60%; `[` drop 2792, share 27%; `"` drop 1053, share 49%;
`{` drop **299**, share **2%** ({-closure barely touches this channel; its
boost of 3.7 nats flows elsewhere). The channel is not a universal bracket
tracker.

**C. Distance stress: decay.** With the bracket still open, a1 separation
(control minus open) holds at ~2700-3350 for opener distances 5-32 tokens,
halves by 64 (1468), and is nearly gone at 128 (219); the behavioral boost
holds ~6-7 nats through 64 and drops to 3.8 at 128. The channel is a
recency-weighted flag, and at long distances the surviving behavior is carried
by other components. (Also: the closed-baseline a1 itself drifts with context
length, 2574 -> 773 — the code's "closed" level is context-dependent, another
deviation from a clean absolute code.)

**D. Closed long ago (does it reset?): FAILURE.** After "( ... )" + k tokens,
a1 stays 550-920 BELOW the never-opened control (k = 2/8/24), and a residual
')' boost remains (+0.3 to +2.0 nats, growing with k in this stimulus set).
Channel-zeroing check: 60-70% of that residual boost at k = 8/24 is mediated
by the channel — it is genuinely incomplete reset of this channel, not just
induction-copying of the literal ")" in context (though that likely
contributes too).

**E. Cross-substitution (does it know WHICH opener?): type-blind, as a 1-dim
channel must be.** Injecting the quote-coded value into parenless prompts
recovers ')' as well as the paren-coded value does (0.295 vs 0.322), and vice
versa (0.177 vs 0.142); every injection raises BOTH closers nearly equally
(own vs other dlp within 0.1 nats, e.g. +2.34 ')' and +2.45 '"' in paren
contexts). The channel says "something is open", not what; closer selection is
computed elsewhere (candidates: the r4 complement, quote-specific head L13H3,
and the value path).

**F. Natural-text attenuation (explains the gate gap).** a1 | paren-open by
distance-since-opener on cooc: 498 (0-2 tokens), 244 (3-6), 570 (7-12),
1107 (13-24), 1461 (25-60), 1536 (61+) vs closed baseline 1605 — the open
signal decays toward baseline with distance. Recalibrating the injected value
on recent-opener positions only (distance <= 12; value 434) lifts gate-A
recovery from 0.280 to **0.434** (paren 0.471 / quote 0.398) with no use of
battery data — most of the code-vs-exact gap was the missing recency term,
not missing state.

## Summary

1. The bracket study's channel is real and re-derived exactly (0.564 / 0.872
   held-out recovery; same subspace).
2. Name as tested ("activation = f(number of currently-unclosed openers)"):
   **directionally verified, literally false.** Verified: LOW activation
   encodes pending-opener; substitution of the coded state passes all placebo
   and deletion controls on held-back natural text (coded dCE +0.0033 <
   mean/shuffled/zero) and on the battery (coded 0.28-0.43 vs placebo 0.11-0.13,
   exact 0.56); dose-response is clean and monotone with natural CE unchanged.
   False parts, found by red-team: not a counter (saturates at depth ~2), not
   all opener types (`{` nearly absent, `[` mostly other paths), not
   distance-invariant (decays by ~128 tokens; recency term is a first-class
   part of the code), not fully reset after closure, and type-blind.
3. Best supported name: **"recency-weighted, type-blind pending-opener flag
   (low = open), strongest for ( and ", graded but saturating in depth,
   leaky after closure."**
4. Methodological notes for the program: (a) for this channel, ZEROING is a
   content-bearing intervention (0 sits beyond the natural "open" value), so
   the zero control scored near the exact patch — mean/shuffled were the
   informative placebos, exactly the failure mode the spec flagged; (b) the
   full-stream battery test is redundancy-dominated (shuffled ~ coded) and
   cannot gate meaning — differential injection can; (c) the exact-value
   reference beats any static code partly because the channel carries
   imminence/recency structure beyond discrete state.
