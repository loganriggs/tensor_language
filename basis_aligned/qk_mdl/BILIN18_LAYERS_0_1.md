# Decomposing the first two layers of bilin18

What worked, what was understood, and what it means. Numbers are traced to `RESULTS_l0_mdl.md`
(cited as §), `ov_metric_explainer.md`, `qk_two_ledgers_explainer.md`, `qk_circuit_tensor_notes.md`,
`TIER2_RESULTS.md`, `qk_mech_bridge.json`, `LOG.md`, the two red-team files of 2026-07-30, and —
for the corrections in section 7 — `../tiny_full_interp/RESULTS.md` FINDING 13.

---

## 1. What the object is

**bilin18** = `gpt2-bilinear-sqrd-attn-18l-9h-1152embd`, 546M parameters, 18 blocks, 9 heads, head
dimension 128, vocabulary 50,304 (`TIER2_RESULTS.md`). Three properties make block 0 fold exactly:

- **No softmax.** The pattern is `(q₁·k₁)(q₂·k₂)/d_head²` times a multiplicative causal mask — a
  product of two independent bilinear branch scores, with no normalising nonlinearity between the
  token tables and the pattern.
- **Ungated bilinear feed-forward.** `out = Down(Left(h) ⊙ Right(h))`: a written-in-weights CP
  tensor of rank 4608 (§7a), exactly trilinear.
- **Affine-free pre-RMSNorm.** A positive per-position scalar with no learned gain — a gauge that
  commutes with the bilinear form and carries outside it (§7f).

So layer 0's query/key input is exactly the RMS-normed embedding, and the circuit folds in closed
form: per head and branch, unit-RMS factor tables `q̂(t), k̂(t)` of shape (50304, 128). The
vocabulary-by-vocabulary score map per head-branch **is** their product through the rotary
expansion, so the map is rank ≤ 128 by construction — "rank-128 SVD" is the exact object
(7,417.6 Mbit), not a baseline. The coding row is `cat([q̂[:,h], k̂[:,h]])`, (V, 256) per
head-branch, 18 head-branches (§1).

**Identity gates.** Folded versus reference forward: max error 1.3e-15; the uncompressed-factors
arm audits at ΔCE +0.0000 everywhere (§1). Baseline CE 3.0763 on the standard 307k-prediction
FineWeb audit. One architectural constraint: unnormalised score-product attention degrades past
T≈512 (CE 3.23 at 512, 5.50 at 1024), so all audits are frozen at T=512 (`TIER2_RESULTS.md`). A
red-team note travels with these gates: they are **architecture tautologies**, holding for any
weights of this shape — method licences, not findings (`redteam_findings_2026-07-30.md` item 6).

## 2. Layer 0 — what worked

**The audit distribution came first, because it had to.** The first three audits were Pile and gave
sign-unstable, sometimes *negative* ΔCE (SVD rank 64: −0.022 on Pile, +0.006 on FineWeb).
Coarsening layer-0 QK genuinely helps off-distribution. All headlines are FineWeb, the training
distribution; this confound caused two false starts (§2).

**The dictionary result.** 1024 atoms per head-branch, 8 active per token, 455 Mbit (6.1% of raw):
**+0.006 nats** with OMP/least-squares, +0.005 trained against the context-expected objective.
Matched-bits low rank (SVD rank 16, 466 Mbit) costs +0.035 — **six times worse** — and the
dictionary equals SVD rank 64 at a quarter of its bits; at 12.4% of raw it gives +0.003 against SVD
rank 32's +0.017. Vocabulary merging is real but modest (+0.042 at 0.6% of bits), and
per-head-branch partitions beat one global partition at matched bits (+0.020 versus +0.035), so
"tokens that attend alike" is a per-head-branch notion (§3).

**Objective progression**, each step buying something specific:

| objective | what it adds | gain |
|---|---|---|
| reconstruction error on factor rows | baseline | +0.0076 at 455 Mbit (§5c) |
| context-expected OV objective (eq. † of `ov_metric_explainer.md`) | splits pattern error into a scatter part accumulating as T and a systematic part as T², charged under a unigram context model | halves cost at 2.5–3% of raw (+0.0073 vs +0.0149 at 183 Mbit); matches MSE-at-455-Mbit with 2.5× fewer bits (§3b) |
| exact-moment form of the same objective | the T² term is a contraction of the head's third-moment tensor, computable exactly at 128³ per head, replacing a sampled estimator (`qk_circuit_tensor_notes.md` §3) | +0.0027 at 455.4 Mbit vs sampled +0.0048 (M=1024) and +0.0033 (M=4096); the M=1024 estimator read 1.084 where the exact value was 0.338, i.e. ~3/4 noise (LOG tick 169a) |

The context objective is seed-robust where its gain is large (+0.0075/+0.0071/+0.0073 across three
seeds at 224 Mbit) and crosses over to plain reconstruction at ~12% of raw, where its
i.i.d.-unigram floor binds (§3b).

**Exact anchor rows and the hybrid frontier.** Error analysis of the 183-Mbit arm: the net cost is
a thin difference of large flows (46% of predictions *improve*; the worst 1% of positions carry
~93% of net), and the top 50 tokens — newline, punctuation, function words — carry 52% of delivered
pattern error **by exposure, not misfit**: their rows fit *better* than average. Head 3 alone
carries 40% (§3c). The fix is exact factor rows for the top-B tokens by attribution, bits charged,
with a dictionary on the tail. With the incoherent-rotary objective (the coherent, offset-averaged
form washes out 98.8% of the systematic signal and loses) this redrew the frontier:

| Mbit | 192 | 262 | 493 | 606 | 1074 | 1393 |
|---|---|---|---|---|---|---|
| hybrid ΔCE | .0044 | .0036 | **.0024** | .0019 | **.0011** | .0010 |
| best previous at same bits | ~.0072 | ~.0070 | .0054 | ~.0048 | ~.0031 | .0018 |

1.8–2.9× lower damage at matched bits everywhere; seed-robust at 493 Mbit (.0024/.0022/.0022) (§3c).

**Why batch-top-k failed.** Pre-registered to lose, and it did: +0.014 versus OMP's +0.006 at
identical bits (2.3×). It selects atoms by raw magnitude under a uniform view of the vocabulary and
never refits, so it spends capacity on rare high-norm rows and degrades when atoms correlate (§3).
This is the anchor result from the other side: the metric bills by **exposure**, and any criterion
allocating by magnitude allocates against it.

**Two negatives that shaped the arc.** Cross-entropy polish through the frozen model buys exactly
zero — held-out damage degrades from the first evaluation while train CE falls, overfitting 12M
dictionary parameters on 154k tokens (§5c). And the two-stage merge-then-dictionary point looked
free on an 8k-prediction audit (−0.0004) and was headlined; the 307k audit put it at +0.028, and it
was retracted (§3).

## 3. Layer 0 — what was understood

**Atom semantics.** The dictionary axes are recognisable token categories: topical classes (music,
film, food, television, religion, persuasion, disaster/place clusters) alongside morphology (plural
and past-tense suffixes, "-ical" adjectives, truncated stems, names, 3-digit numbers, hedging
adverbs) — `qk_dict_features.md`, §4. Topic-level semantics at layer 0 was the surprise; the
expectation was morphology only. **This is a qualitative dump, not a gated measurement** — no
purity-versus-null statistic exists for bilin18's atoms.

**The mechanism ledger.** A separate object: per head, the unigram-weighted third moment of the
source-side triple rows `[k₁ | k₂ | v]`, sparse-coded, then symmetric nonnegative CP. Components
are **archetypes** — a direction naming a key-token class on both branches plus what is written
when it matches (`qk_two_ledgers_explainer.md` §1). They came out **case-invariant closed-class
scaffold classes**: head 8 factors into {the}, {a/an}, {of}, {and}; heads 2 and 5 into punctuation
families, newline and document-boundary units (§5e).

Solver discipline makes this a result rather than numerology. Of five CP fitters tried, only
**tensor power iteration with deflation** passes the planted known-answer test (matched cosine
0.9998); every ALS and gradient variant fails on spiky-sparse cores. The sparse coder needed
nonnegativity plus k-annealing to reach planted recovery 1.0 (base 0.88). These controls caught
solver bugs twice (§5e).

**Capacity spans 128-fold across heads.** Minimum atoms to pass the moment gate: head 2 needs 32
(at one feature per token), heads 6/8 need 128–256, heads 1/3/5/7 need 256–1024, heads 0/4 need
4096. Two features per token is the sweet spot, and retraining at the right size beats pruning a
large dictionary by roughly an order of magnitude (head 8: 0.042 retrained versus 0.344 pruned);
per-head optima cost 53.5 Mbit against 131 Mbit uniform (§5f).

**Causal importance dissociates from weight mass.** Whole-head ablations, full audit: h3 +0.0780,
h7 +0.0090, h8 +0.0051, h6 +0.0041, h0 +0.0026, h4 +0.0016, h5 +0.0014, h1 +0.0006, h2 +0.0005.
Head 3 is ~8× the runner-up and ~60% of the layer's causal load, while heads 0 and 4 — largest
mechanism mass, 4096 atoms each — are nearly free. The ten-archetype span carries 73–88% of
whole-head load on heads 3/6/7/8, and channel ablations are sub-additive (singles sum +0.251 versus
+0.070 for the span on head 3), so the channels overlap on a shared direction (§5l). A cheap
weight-space correlate exists: expected output magnitude Σₜ pₜ‖W_o v_t‖ ranks heads at Spearman
+0.87 (§5m).

**The corrected permutation null.** Heads 0 and 4 were reported to *fail* the permutation null. A
head-5 control exposed the statistic as invalid: a mode-permuted core approaches a product of
independent marginals, which is intrinsically near-low-rank, so comparing fits across two
*different* target tensors can favour the permuted one. The corrected statistic scores everything
on the same real core — transplant the null's factors onto it, refit only the nonnegative weights.
Under it null factors explain essentially nothing (relative error 0.91–1.00 ≈ predicting zero)
while real fits explain 71–87% of core mass, and **all nine heads validate**; the earlier verdict
was an artifact (§5g). Cores are corpus-general: mean effective 9.7–10.4 of 12 document components
per archetype (§5h).

## 4. Layer 1 — what worked

Layer 1 reads the post-block-0 residual, so the exact fold no longer applies. The **port test**
asks whether the layer-0 machinery transfers with token-conditional mean-residual tables in place
of embeddings. **It does, and this is the landmark: layer 1's pattern is ~99% token identity.** Replacing the
entire layer-1 pattern with static token tables costs **+0.027 nats** against **+2.70** for zeroing
it — a 100× ratio (§6a). Two facts sit beside it: layer 1's pattern is ~27× more causally important
than all of layer-0 attention (+2.70 versus ~+0.10 summed), and layer-1 heads are massively
super-additive — single-head zeroing sums to +0.128, 21× below the joint effect. Layer-0 heads do
not behave this way.

**Shrinkage for rare-token noise.** Head 1's tables initially failed the moment gate. Restricting
the gate to tokens seen ≥4 times gave 0.031 (a pass) — the divergence came entirely from
poorly-estimated rare-token means, cubed by the third moment. A shrinkage estimator (τ=8 toward the
embedding prior) fixed it outright: gate 0.0000 at 1024 atoms (LOG tick 195).

**Per-head validation.** All nine heads pass the moment gate (seven at 512 features, heads 1 and 3
at 1024), beat the corrected null (real fits 0.10–0.52 versus null-factors-on-real 0.54–1.00), and
are restart-stable at 0.96–1.00 (§6b). Per-head port costs are uniformly small (h8 +0.0045, h4
+0.0036, h1 +0.0032, rest 0.0005–0.0013; sum 0.016 versus joint 0.027), and layer 1's static
component is *cheaper* than layer 0's: six of nine heads sit at a 32-atom floor, h0/h6 at 128, h3
at 1024 (LOG ticks 195, 198).

**A different vocabulary from layer 0.** Where layer 0 is function-word scaffold, layer 1's top
archetypes are sentence and discourse boundaries (heads 0, 4, 7: period/quote/exclamation
families), document boundaries (heads 2, 5, 6: newline, end-of-text), quote and bracket openers
(head 8), and — on head 1, the causally dominant head — **mid-word subword fragments**
('cknowled', 'theless', 'secut'), a subword-continuation role that explains both its causal weight
and its context-dependence (§6b).

**The equal-ablation negative.** Do archetype directions have causal privilege? Six arms on head 3,
scored by pattern energy removed. Damage per unit energy: archetype-10 **2.14**, PCA-10 2.41,
energy-matched uniform shrink 2.44 — the archetype directions do no *more* damage per unit than
generic or uniform removals, slightly less, and concentration is identical across arms. Head 3's
downstream consumption is approximately isotropic in pattern space: the big directions are the used
directions. **The archetypes are descriptive, compressive and predictive, not causally
privileged** (§5n). This also explains an earlier null random-subspace control: the random-10 arm
removed 77× less energy than the archetype span.

## 5. The block-0 feed-forward block

Layer 1 cannot be understood without it: the block-0 bilinear MLP **authors most of layer 1's
context sensitivity** — ridge regression gives its output 45–64% of layer-1's per-position factor
deviations against 21–35% for layer-0 attention (§6d) — and zeroing it costs +2.50 nats, as large
an object as layer 1's whole pattern (§7b).

**Exact tensor, neuron basis as gauge.** The block is a rank-4608 CP tensor written in the weights;
the neuron basis indexes that factorisation, not features (neuron permutation leaves the folded
tensor identical), and it is empirically useless — flat usage spectrum (top 128 neurons carry 6% of
write-weighted usage), keeping the top half costs +0.030, and layer-1's read maps touch essentially
all neurons (effective count 4361–4568 of 4608) (§7a). Weight-space rank is not sparse either
(effective rank 12–101, median 68 across the 36 read channels), and composed-tensor CP in pure
weight space is dense with the corrected null **tying** the real fit, 0.483 versus 0.485 (§7b,
§7g, §7h).

**The realised interface collapses.** On held-out text each MLP→layer-1-reader channel has
**median effective rank 10** (min 1, max 62) against 68 in weight space — stable across 16× more
data (10.4 → 10.5 → 10.7 at 32k/131k/524k positions) (§7c, §7f). Priced by full-model audit, with
layer 1's factors as token table plus a rank-r context adapter (§7d):

| context rank per channel | 0 (static tables) | 4 | 16 | 64 |
|---|---|---|---|---|
| held-out ΔCE | +0.0515 | +0.0208 | +0.0113 | +0.0009 |

Sixteen dimensions recover 78% of the static gap, sixty-four recover 98%, and the adapter bases
cost ~2.4 Mbit.

**Linear generation explains under a third.** Replacing the oracle projection with a real generator
— one shared 64-dimensional principal subspace of the MLP output plus a ridge map per channel,
nothing reading the true factors — audits at **+0.0365**, i.e. **29% of the context gap** (§7e).
Architecture is not the bottleneck: bilinear-gate, SwiGLU, single-encoder-MLP and hierarchical
generators all land at +0.0334–0.0336, and adding attention codes reaches +0.0319, ~49% of the gap
(§7i, §7j). What is missing is *information*, and it has a name: failures are 2.5× enriched at
mid-word positions and localise to layer-1 head 1, which alone carries 56% of remaining damage
(§7k, §7l). It is not a function of recent token identities — a previous-token correction table
explains 2% of the residual and gains nothing end-to-end; window codes give R² 0.02–0.04 (§7l,
§7m). What *does* reproduce it: **block 0 itself, run on the last sixteen tokens** (3% of the
context), audits at **+0.0099**, beating even the 16-dimensional oracle interface. W=8 gives
+0.028, W=4 +0.070, W=1 +0.363 (worse than no context at all); dropping the MLP from a W=4 window
gives +0.667 (§7n).

## 6. How much was understood — an honest ledger

**(a) Exactly represented.** The layer-0 per-head-branch query/key object (gate 1.3e-15,
uncompressed arm +0.0000) — 100% of layer-0 QK, as an identity. The block-0 feed-forward block as a
rank-4608 CP tensor, by construction. The weight-space split of the MLP→layer-1-QK channel over
writers (§7f). That is the whole of (a); everything else is approximation.

**(b) Causally substitutable at a stated cost.** Layer-0 QK: 493 Mbit at +0.0024, 1074 Mbit at
+0.0011 (§3c). Layer-1 pattern: static token tables at +0.027 (raw-mean) or +0.0515 (shrunk
tables), against a +2.70 destruction floor (§6a, §7d). Layer-1 interface: oracle rank-16 adapters
+0.0113, rank-64 +0.0009; best fully generated pipeline +0.0319; block-0-on-16-tokens +0.0099 (§7d,
§7j, §7n). Mechanism-ledger keys substituted for real ones: **+0.0067** at 168.1 Mbit of codes but
with the query side left raw at 3708.8 Mbit (`qk_mech_bridge.json`) — roughly 3× the function
frontier's damage at ~8× its total bits.

**(c) Named with a passing gate.** Layer-0 archetypes: 9/9 heads beat the corrected null (§5g); 7/9
pass the moment gate at 512 atoms, the other two at 4096 (0.0279/0.0293, LOG tick 180); restart
stability 0.94–1.00; corpus-general at effective 9.7–10.4 of 12 components (§5h). The ten-archetype
span is causally near-complete on the load-bearing heads (73–88% of whole-head damage, §5l).
Layer-1 archetypes: 9/9 validate, restart 0.96–1.00 (§6b). The two ledgers converge: 14–21% of top
archetype-loading tokens fall in the anchor-256 set against a 0.5% baseline, 28–42× enrichment
(§5e). One negative gate that counts as understanding: **no layer-0 head is a direct-path copy
head** (copy cosines −0.08 to +0.03 across every archetype of every head) (§5j).

**(d) Merely described.** Atom semantics — real and readable, but ungated at bilin18. The block-0
feed-forward interior: dense in neurons, dense in weight rank, and the one attempt at sparse
token-pair structure had its null tie the real fit. Layer-1 head 1's context keys: every table- and
code-based generator failed; the honest description is "block 0 on a 16-token window", whose
interior remains dense.

**The proportions are uncomfortable and should be stated.** Collapsing all nine layer-0 heads'
content costs +0.57 nats; summed whole-head ablations are ~+0.10. Layer 1's pattern is worth +2.70
and the block-0 feed-forward block +2.50. The object we understand best — layer-0 QK: exact, named,
gated, causally profiled — is **the smallest object in the first two blocks by more than an order
of magnitude**. The two large ones are substitutable (layer 1) or priced at their interface (the
MLP); neither is named. By causal weight, the fraction of block-0-plus-block-1 computation that is
both named with a passing gate and causally verified is on the order of a few percent.

## 7. What it means

**Correction 1 — the fold is an expansion, so "6.1% of raw bits" is not a description length.** The
7,417.6 Mbit object is generated by the embedding (50304×1152 = 1854.4 Mbit) plus four layer-0
projections (4×1152² = 169.9 Mbit): 2024.3 Mbit total. The fold is therefore **3.66× the weights it
is derived from**, exactly 4.0× the embedding alone — the same ratio measured in the ported
replication, where at that scale the fold is 3.1× the whole model (FINDING 13 §0). Against the
honest baseline of storing the generators, the 455 Mbit dictionary is a **4.4× compression, not a
16× one**, and the 493 Mbit hybrid is 4.1×. The frontier remains a true and useful statement
**about the circuit** — how much stored structure is load-bearing for prediction — but it is not a
minimum description length of layer-0 QK, and any citation of the 6.1% figure as an MDL claim must
carry this.

**Correction 2 — the context-expected objective's geometry contributes nothing.** In the
replication, a control discarding the metric's directions and keeping only its per-token, per-block
scalar mass (trace/hd × I) **beats the full metric** at both budgets: CE 4.8296 versus 4.8524 at
4.85 Mbit, 4.7715 versus 4.7854 at 19.5 Mbit, with both beating plain reconstruction (4.8868 /
4.8013) (FINDING 13 §4b). The transferable ingredient of eq. (†) is *how much of the circuit's work
passes through this token's row*; the T-versus-T² cancellation split, the OV directions and the
Gram/norm interpolation are decoration at that scale. **This has not been retested at bilin18** —
it is a prediction, and worth one run. If it holds, several ticks of derivation reduce to a single
per-token scalar weight, and the objective, the anchors and the batch-top-k failure become one
statement: *allocate bits by exposure*.

**The selection/content boundary.** Three naming hypotheses were run against a substitution meaning
gate on exact weight-derived spectra: broad classes on PCA coordinates gave 3/576 codable (all
newline), median class-R² 0.022; the same classes on the 144 archetype value-coordinates gave
2/144, median R² 0.103; token-spike codes gave zero coordinates at concentration ≥0.8. The few
names that exist pass exactly (§34). The dichotomy: **who is selected is nameable; what is written
is spectral.** Scaffold classes live in the third-moment factors — the selection side — while the
value-write spectra are distributed and graded in every basis tried. This is the same shape as the
layer-0 equal-ablation negative (selection directions carry no causal privilege per unit energy)
and the same shape as the MLP result (a 16-dimensional interface, priced, whose content is not
nameable). The implication for practice: target selection predicates, and stop expecting content to
factor into human categories — for content, the exact weight-derived spectrum *is* the description.

## 8. What we would do differently

1. **Charge description length against the generator, not the object.** The expansion arithmetic
   was available before any run. The same omission recurs later, where composed forms referencing
   full weight tensors were compared against data programs 27× smaller
   (`redteam_findings_2026-07-30.md` item 4).
2. **Match controls on energy, not dimension.** The matched-dimension random subspace passed as a
   control for two ticks while removing 77× less pattern energy than the arm it controlled; the
   energy-matched shrink turned a "directions matter" claim into a clean negative (§5l, §5n).
3. **Never compare fits across two different target tensors.** The permutation null cost a wrong
   verdict on two of nine heads; transplant-onto-the-real-core should be the default structural
   null (§5g).
4. **Size the audit before headlining.** The two-stage retraction (−0.0004 on 8k predictions,
   +0.028 on 307k) and the negative-ΔCE saga were small-audit and off-distribution artifacts (§2,
   §3), and 0.005-nat contrasts ran most of the arc without paired standard errors
   (`redteam_findings_2026-07-30.md` item 8).
5. **Treat gauge identities as licences, not milestones** — exact-fold gates hold for any weights
   of this architecture (`redteam_findings_2026-07-30.md` item 6).
6. **Restrict mechanistic nouns to the calibrated set, and label shared priors.** The semantics
   red-team found a "token pointer law" that was a per-calibrated-element table (held-out elements
   follow at 0.00–0.25), an agreement figure inflated by a shared fallback (0.94 → follow rate
   0.65–0.71), and three "independent confirmations" of the selection/content dichotomy that were
   one coordinated probe with shared priors (`redteam_semantics_2026-07-30.md`).
7. **Prefer smooth bit allocation to exact rows.** The replication's stratified-sparsity code
   matches the exact-anchor hybrid at matched bits (4.8231 at 5.14 Mbit versus 4.8244 at 5.37), so
   exactness buys nothing a longer code for the same tokens does not (FINDING 13 §4a). If that
   retests here, the anchor machinery simplifies to a per-token bit budget.
