# Circuit write-up: `aspectual_anchor.has_vs_had`

**Date:** 2026-09-06 · **Model:** bilin18 (18-layer bilinear-MLP transformer, n_embd 1152, 9 heads; GPT-2 tokenizer) — an earlier revision of this file said "GPT-2 small", which was wrong · **Lane:** Claude (breadth) with Codex follow-ups
**Screen receipt:** `circuits/fast_screens/aspectual_anchor_has_vs_had_v1_result.json`
**Ledger:** `circuits/fast_screen_ledger.jsonl` · **Verdict:** `selective_causal_site`

---

## 1. The behaviour

A temporal preposition fixes an aspectual anchor point, and the model must carry that anchor
across the subject to the auxiliary. `since` opens an interval reaching the present and takes the
present perfect; `by` fixes a past deadline and takes the past perfect.

The cue is a function word two phrases before the target. The target is an auxiliary — not a
determiner, complementizer or preposition, which is what most of the corpus predicts.

### Stimuli (one panel, 32 panels total, 128 rows)

| hypothesis | base | donor | answer |
|---|---|---|---|
| **A1** direct temporal | `Since last survey the leader` | `By last survey the leader` | ` has` / ` had` |
| **A2** report-embedded | `The record shows that since last survey the leader` | `The record shows that by last survey the leader` | ` has` / ` had` |
| **P** invariance | `Since last survey the leader` | `Since last match the leader` | ` has` (both) |
| **C** control | `Beside the meadow the leader finished the survey in the middle of the` | `Inside the garden the member completed the sketch in the middle of the` | ` night` (both) |

Base and donor differ **only** in the preposition and end on the same final token (` leader`), so
a site that transfers the has/had decision is carrying the aspectual anchor rather than any
surface property of the patched position. A2 re-expresses the same variable under a report frame,
so a single-construction cue cannot satisfy both. P varies the *period*, not the subject, because
the subject is the final input token here.

### Native capability

All four families clear the 0.85 per-cell bar. The behaviour is one the model genuinely performs
before any intervention is applied — this is a gate, not a result.

---

## 2. Counterfactual (interchange) results

**Method — stated precisely.** Every number below comes from *interchange intervention*: run the
base prompt, copy the site's activations from the donor run, and measure how far the answer moves
toward the donor's answer, normalised by the native base→donor separation. Recovery 1.000 means
the patch moved the prediction the whole way.

**No DAS enters sections 2–4.** Nothing below involves a learned rotation, a trained subspace, or
an alignment search; the direction results came later and are in section 5 (2026-09-06 14:48 UTC),
which supersedes the "not attempted" statements in section 3.

### Site sweep — 55 sites, whole-block granularity

| site family | best site | A1 recovery |
|---|---|---|
| residual stream | `resid:18` | **1.000** |
| attention block | `attn:09` | 0.382 |
| MLP block | `mlp:11` | 0.146 |

Selected site `resid:18`; **9 passing sites, `resid:10` through `resid:18`**.
At the selected site: **A1 1.000, A2 1.000, direction fraction 1.00, P 0.181, C 0.202**
(C under canonical control v2; bars are A ≥ 0.5, direction ≥ 0.8, P ≤ 0.2, C ≤ 0.35).

**The circuit path, as far as this screen resolves it:** the aspect variable is present in the
residual stream from **layer 10** onward and fully recoverable by layer 18. No single attention
block or MLP block carries it — the best attention block reaches 38% and the best MLP 15%. The
effect is distributed across blocks and only the residual stream carries it whole.

---

## 3. Finer than a block? Partly — and the honest number is small

Codex's lane ran ~12 follow-up screens on this behaviour and did get below block granularity.

**What was found (`aspectual_anchor_mlp4_induced_attention5_four_head_factorial_v1`,
`..._mlp4_to_attention5_four_head_source_identity_v1`, terminal `screen`):**

- an explicit **MLP-4 → attention-5 four-head** path, with attention-5 **head 7 dominant**
- released as a typed path program, `aspectual_anchor.has_vs_had.transparent_path_program_v1`,
  with a prospective lexical holdout that passed all six registered predictions and a
  bank-to-complete coverage fraction of 0.9999987

**What that path actually recovers, which is the part that matters:**

| path | mean target recovery |
|---|---|
| all sources → complete four-head path | **0.046** |
| all nine attention-5 heads | 0.055 |
| `resid:18`, whole residual stream (for comparison) | **1.000** |

So the head-level path is real, replicated, and prospectively validated — but it accounts for
roughly **5%** of the effect that the residual stream carries in full. Reporting "we localised
this circuit to four heads" without that ratio would badly overstate it.

**What failed.** `aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1` returned **null**
(`writer_reader_mediation_or_specificity_failed`) at recovery 0.130. The layer-9 head-1/head-4
mediation hypothesis did not hold.

**Sub-head grain (directions, subspaces, individual neurons): not attempted.** The finest
resolution reached is an individual attention head. Going below that is what DAS or a
weight-space decomposition would be for, and neither has been run here.

---

## 4. What this circuit does and does not establish

**Established.** bilin18 carries an aspectual anchor from a temporal preposition, across an
intervening subject, to the auxiliary; the variable is recoverable in full from the residual
stream from layer 10; the transfer survives a construction change (A2) and an answer-preserving
edit (P 0.181).

**Not established.** That any single head or MLP *implements* the behaviour — the block-level
maxima (0.38 / 0.15) say the opposite. That the four-head path is the mechanism — it recovers 5%.
Any claim about directions or subspaces within a head.

**Caveat on the control clause.** C = 0.202 against a 0.35 bar. Measured across this corpus, a
same-answer control's ceiling is ~0.07–0.23, so the C clause here passed but was never at serious
risk of failing. The selectivity claim rests mainly on target recovery plus P invariance; C should
be read as a reported statistic, not as a test that could have gone the other way.

---

## 5. Update, 2026-09-06 14:48 UTC: a three-head set carries 0.61, and one direction per head carries that

Sections 3–4 said the finest resolution reached was a single head at ~5% of the effect. That was
before the greedy head-set protocol (`ops/circuit_unit_greedy.py`; `unit_greedy_battery_v5`,
re-measured under corrected patch semantics in `unit_block_live_directions_v8`). All numbers are
held-out interchange effects on the task margin; the held-out rows are the reverse direction
(had→has) on fresh sentences.

**Head set.** 162-head sweep, greedy forward selection over the top 12 (gain floor 0.02, at most
6): T = {attn:08:head:01, attn:09:head:04, attn:09:head:01}, joint recovery **0.605** on the fit
rows, **0.614** on held-out A1, **0.633** on A2; through the exact set P = 0.096, C = 0.013. The
best whole block was attn:09 at 0.38, so the three heads together exceed any single block; the
remaining ~0.39 is spread over units that did not clear the gain floor.

**Direction.** Per head, the sign-aligned mean of (donor − base) over the head's 128-d output at
the read position (diff-in-means, no search), applied to the live value as
`x + qqᵀ(x_donor − x)`. Held-out fraction of the set's effect **0.98**, A2 **0.89**; complement
(swap everything in the three heads except the direction) 0.00 / 0.09; subspace + complement =
0.98 (linear); random rank-matched direction 0.00; P 0.063, C 0.001. A fitted DAS direction (rank
1 per head, exact-set objective) matches (1.02 / 0.97, complement −0.02) at cosine 0.75 to the
diff-in-means per head. The cached-vs-live full-rank control is 0.613 vs 0.614 (the layer-8 patch
barely perturbs the layer-9 heads' outputs), so this set was not affected by the cross-layer
inflation that hit the possessive sets.

**What this changes in section 4.** "Not established: that any single head or MLP implements the
behaviour" stands; "any claim about directions or subspaces within a head" no longer does — a
single direction per head carries 0.98 of what the three heads carry. What the residual stream
carries in full (1.000 at `resid:10`–`resid:18`) is still larger than T (0.61); `resid:18` itself
is tautological (the margin is a linear readout of it) and is no longer a comparison point.
The layer-9 head-1/head-4 mediation null in section 3 (0.130) was a path (MLP-4 → those heads)
test, not a test of the heads themselves; 09:01 and 09:04 both belong to T.
