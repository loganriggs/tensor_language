# What the error actually is: exploration of the 183-Mbit dictionary's residual (tick 164)

Target: the most-compressed frontier arm — n=256 atoms, k=4 per head-branch, 183.4 Mbit
(2.47% of raw), OV-context-trained. Mean held-out ΔCE here +0.0079 (MSE dictionary at the
same budget: +0.0171). All statistics over the 307,200 held-out FineWeb predictions.
Companion files: `qk_err_explore_examples.md` (top-100 decoded), `qk_err_explore.json`
(all numbers), `qk_err_explore.pt` (per-position arrays).

## 1. The error is a small net of two large opposing flows

- **45.7% of predictions get BETTER under compression.** The improved positions sum to
  −4.7× the net error; the harmed positions sum to +5.7× the net. The +0.0079 headline is
  the thin difference between two big flows — compression is a regularizer on half the
  distribution and a lesion on a small tail.
- **Concentration in the harmed tail is extreme**: the worst 0.1% of positions carry 21% of
  the net; the worst 1% carry ~93%; the worst 5% carry 2.4× the net (offset by improvements
  elsewhere). So the net cost is essentially ~3,000 identifiable bad predictions, not a
  uniform fog.
- ctx-trained and MSE-trained dictionaries at the same budget only moderately agree on
  *which* positions fail (Pearson 0.46, Spearman 0.39): the objective substantially steers
  who pays, so the tail is not intrinsic to the budget — it is optimizable.

## 2. Who fails: commonality of the top-1000 (vs a random-1000 control)

| statistic | top-1000 worst | random control |
|---|---|---|
| target frequency rank (median) | 1,713 | 188 |
| target seen earlier in context | 33.7% | 47.3% |
| repeated bigram | 5.2% | 10.2% |
| preceding token = "\n" | 7.2% | 3.4% |
| position in sequence (median) | 298 | 259 |
| distinct documents | 441 | 478 |

ΔCE by target-frequency decile climbs from ~+0.005 (frequent) to **+0.014 on the two rarest
deciles**; by position it drifts mildly upward late in the window. Reading the decoded
top-100 confirms three recurring situations:

1. **Compound-name completion**: "Search Engine → Watch/Land/Journal" (≈10 of the top 100,
   across several SEO-newsletter list documents), "Box Office → Mojo", "Radio→head",
   "Peregrine Cuttle→fish", "polesitter", "JNCO", "dOCUMENTA". Second tokens of multiword
   names — local composition / copy structure that layer-0 QK apparently scaffolds.
2. **Structured list/table documents**: tournament tables with "|" separators, AV-receiver
   spec sheets, footer boilerplate — sharp format predictions ("\n", ",", "-", repeated
   headers) that the intact model gets at CE ≈ 0.1–0.4 and the compressed model misses by
   +2 to +5 nats. Newline-as-previous-token is over-represented 2× in the tail.
3. **Context-driven content retrieval**: "cold war → rivalry", "credit union → industry",
   "Free Agent, → restricted" — mid-to-rare content words needing the context assembled.

## 3. Where the pattern error lives in weight space (eq.-† attribution, full vocabulary)

- Per-query-token error contribution is **extremely token-concentrated: the top 50 tokens
  carry 52% of the total**, and the top frequency decile carries 81%. The top query tokens
  are "\n" (8.9% of ALL weighted error by itself), " to", " the", " a", "-", ",", " of",
  ".", " in", " and". Key-side attribution: same set (",", " the", ".", "\n", " to"...).
- Resolution of the apparent paradox (pattern error on *frequent* function tokens, CE damage
  on *rare* targets): layer-0 heads use punctuation/newline/function tokens as structural
  anchors; corrupting how those anchors score against keys scrambles the scaffold, and the
  downstream cost surfaces exactly where the scaffold was load-bearing — compound-name
  continuations after "\n" or "-", list structure, rare content retrieval. The frequent
  tokens are where the error *is*; the rare targets are where it *bills*.

## 4. Who fails among heads

Auditing with only one head compressed (others exact): **head 3 alone costs +0.0032 — 40%
of the joint +0.0079**; every other head costs +0.0002–0.0007. (Sum of singles +0.0061 vs
joint +0.0079: mildly superadditive.) Recall from the collapse sweep: heads 2 and 5 are
content-free (+0.001–0.002 when collapsed entirely), yet they receive the same 256-atom
budget as head 3. The uniform per-head budget is clearly misallocated.

## 5. What the factor-space residual looks like

- Per-token relative row error is ~0.50–0.60 across all frequency deciles (mildly better on
  frequent tokens — the ctx objective's q-weighting at work, but weakly).
- The residual is **not low-rank**: top-8 directions hold ~8–11% of residual energy, top-32
  only ~25–33% (of 256 dims). A dense low-rank correction term would NOT capture it —
  rank-augmentation is ruled out as a cheap fix.
- Residual splits ~evenly between q-half and k-half (0.38–0.58 across head-branches).
- The worst-fit tokens in factor space are the GPT-2 anomalous/"glitch" tokens
  (" davidjl", "TPPStreamerBot", "oreAndOnline"...) at ~0.93 relative error — but they
  never occur in data, so they cost nothing. Harmless bit-waste, not a problem.

## 6. Hypotheses → solutions (ranked by expected value / cost)

- **S1 — Exact rows for the anchors** (from §3): 52% of weighted error sits in ≤50 tokens
  per the † attribution. Storing exact 256-float rows for the top-B tokens per head-branch
  costs B × 8 kbit per head-branch (B=128 across all 18 head-branches ≈ 19 Mbit, +10% of
  budget) and needs no retraining to test. Predicted: large fraction of the tail recovered.
- **S2 — Reallocate budget across heads** (from §4): fixed total bits, but head 3 gets a
  large dictionary, heads 2/5 get near-nothing (they're collapsible for +0.001–0.002),
  others stay small. The uniform allocation is provably wasteful.
- **S3 — Tail-aware objective** (from §1/§2): the † objective minimizes the *expected*
  error under unigram query weights, which is why 46% of positions improve while a rare-
  query tail (compound names, list anchors) carries 93% of the net. Reweight the query
  distribution (e.g. q^0.5, or upweight by per-token normalized error) to buy tail
  robustness with average-case slack — the low ctx-vs-MSE position correlation (§1) says
  the tail is steerable.
- Ruled out by the data: dense low-rank residual correction (§5); co-occurrence weighting
  (tick-162 null); reader co-adaptation (tick-161 null).
