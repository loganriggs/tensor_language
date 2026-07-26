# Layer-0 query/key MDL decomposition — results (ticks 150–155, 2026-07-21/22)

**Program:** two-stage minimum-description-length decomposition of the embedding as read by the
first query/key circuit of bilin18 (546M-parameter bilinear-attention model, no softmax).
Stage one = vocabulary merge ("tokens that attend the same are the same token"); stage two =
sparse dictionary ("each token is a sparse combination of sub-patterns"). Everything is
**weight-only**: the object is the exact layer-0 fold (verified to ~1e-15 against the reference
forward); data enters only the held-out evaluation.

**TLDR:** The sparse-dictionary hypothesis wins. A per-head-branch dictionary of 1024 atoms with
8 active per token reproduces the circuit at **+0.006 held-out cross-entropy on the training
distribution using 6.1% of the raw bits** — six times better than matched-bits SVD, and equal to
an SVD spending four times the bits. The dictionary atoms are interpretable and surprisingly
semantic. Two headline-shaping methodology findings along the way: audit on the **training
distribution** (off-distribution Pile audits have a real coarsening-helps confound), and plain
factor-level FVU is the best cheap proxy for behavioral cost (energy-weighted / OV-composed
metrics do worse).

---

## 1. The object

At layer 0 the query/key input is exactly the RMS-normed embedding, so the circuit folds in
closed form: per branch and head, unit-RMS factor tables `q̂(t), k̂(t)` of shape (V=50304, 128).
The vocab-by-vocab score map per head-branch **is** the product of these factor tables (through
the rotary cosine/sine expansion), so decomposing the factors decomposes the map losslessly —
and the map is rank ≤ 128 *by construction* (it factors through the head), so "rank-128 SVD" is
the exact object (7,417.6 megabits), not a baseline. The baseline is the **rank-r bits frontier**.

Rows for merging/coding: `cat([q̂[:,h], k̂[:,h]])` — (V, 256) per head-branch, 18 head-branches
(9 heads × 2 bilinear branches). Gates: fold vs reference forward max error 1.3e-15; the
uncompressed-factors arm audits at ΔCE +0.0000 on every audit set used.

## 2. Methodology finding: audit on the training distribution

The first three audits (16 seqs → 8k preds; 128 seqs → 65k preds; 512 seqs → 262k preds) were all
**Pile**, and produced sign-unstable, sometimes *negative* ΔCE for compressed arms. The 600-seq
**FineWeb** audit (307k preds — the model's training distribution) resolved it:

| arm | Pile-big (262k) | FineWeb (307k) |
|---|---|---|
| svd rank 16 | +0.014 | +0.035 |
| svd rank 64 | **−0.022** | +0.006 |
| dict n=1024 k=8 OMP/LS | −0.011 | +0.006 |
| merge K=2048 per-head-branch | −0.003 | +0.020 |

Coarsening the layer-0 QK circuit genuinely *helps* on off-distribution text (a regularization
effect), while on the training distribution every compression has an honest positive cost that is
nearly monotone in bits. **All headline numbers below are FineWeb.** (This also retro-explains the
whole negative-ΔCE saga in LOG ticks 151–153 — part noise, part distribution.)

## 3. The frontier

![Layer-0 MDL frontier on the training distribution](fig_qk_mdl_frontier_fw.png)

*Panel A — held-out ΔCE (FineWeb) vs description length, log scale. Blue = SVD rank frontier,
orange = stage-one merges, teal = stage-two dictionaries, star = the (retracted) two-stage
composition, black dot = exact raw factors. Panel B — structural error (fraction of variance
unexplained) vs bits. Panel C — the two error measures against each other: on-distribution they
mostly re-couple.*

Full FineWeb table (baseline CE 3.0763; raw object 7,417.6 Mbit):

| arm | Mbit | % raw | ΔCE (FineWeb) | factor FVU |
|---|---|---|---|---|
| svd rank 8 | 233 | 3.1% | +0.045 | 0.69 |
| svd rank 16 | 466 | 6.3% | +0.035 | 0.62 |
| svd rank 32 | 932 | 12.6% | +0.017 | 0.51 |
| svd rank 64 | 1864 | 25.1% | +0.006 | 0.35 |
| svd rank 128 | 3728 | 50.3% | +0.002 | 0.15 |
| merge K=256 per-head-branch | 45 | 0.6% | +0.042 | 0.69 |
| merge K=2048 per-head-branch | 312 | 4.2% | +0.020 | — |
| merge K=8192 per-head-branch | 1220 | 16.4% | +0.008 | 0.47 |
| merge K=2048 **global** partition | 303 | 4.1% | +0.035 | 0.66 |
| **dict n=1024 k=8, OV-context-TRAINED** (tick 159; linear encoder, trained on eq. † of `ov_metric_explainer.md`) | **455** | **6.1%** | **+0.005** | — |
| **dict n=1024 k=8, OMP/least-squares** | **455** | **6.1%** | **+0.006** | **0.40** |
| dict n=1024 k=8, linear encoder | 455 | 6.1% | +0.008 | 0.46 |
| dict n=1024 k=8, matryoshka | 455 | 6.1% | +0.008 | 0.46 |
| dict n=1024 k=8, batch-top-k | 455 | 6.1% | +0.014 | 0.48 |
| **dict n=4096 k=8, OMP/least-squares** | **923** | **12.4%** | **+0.003** | **0.30** |
| two-stage merge2048 → dict 512/8 | 98 | 1.3% | +0.028 | 0.66 |

Commentary:

- **Dictionaries Pareto-dominate every family.** At 6.1% of raw bits the OMP dictionary matches
  svd r64's quality at a quarter of its bits; at 12.4% it beats svd r32 five-fold. The token rows
  really are better modeled as sparse combinations of sub-patterns than as a low-rank subspace.
- **Stage one (merge) is real but modest**: per-head-branch clustering beats the SVD curve at low
  bits (+0.042 at 0.6% vs svd r8's +0.045 at 3.1%), but dictionaries beat both.
- **Per-head-branch structure matters**: one global vocabulary partition shared by all 18
  head-branches costs +0.035 where 18 independent partitions cost ~+0.020 at the same bits —
  "tokens that attend the same" is a per-head-branch notion, consistent with 7 of 9 heads having
  marginal effective alphabet 1.
- **Encoder ordering (pre-registered in Phase 0 and confirmed here)**: OMP with least-squares
  refit is the strong arm; batch-top-k is the weakest (2.3× OMP's cost) — raw-magnitude atom
  selection without a refit degrades when atoms correlate, exactly as the planted control
  predicted. Matryoshka ≈ linear ≈ mid.
- **Retraction**: the two-stage merge-then-dictionary point briefly looked free (−0.0004 on the
  8k-pred audit) and was headlined at tick 152; the 65k- and 307k-pred audits put it at
  +0.017…+0.028. Small-audit overfitting — it is *not* a good point.

## 3b. The in-depth Pareto sweep (tick 160): objective × budget × seeds

![Dictionary Pareto sweep](fig_qk_pareto.png)

Overnight sweep: 8 budgets (2.5%–20.7% of raw) × {MSE, OV-context-trained (eq. † of
`ov_metric_explainer.md`)} at identical bits, OMP audits at seed 0, 3 seeds at three anchors.
Findings:

1. **OV-context training dominates the low-bit frontier.** At 2.5–3% of raw bits it *halves* the
   cost of the best MSE arm (+0.0073 vs +0.0149 at 183 Mbit) and matches MSE-linear-at-455-Mbit
   quality with 2.5× fewer bits. Its curve is nearly flat (+0.005–0.007) across the whole range —
   the objective extracts the behaviorally relevant structure almost independently of budget.
2. **Seed-robust where it matters**: the paired linear-vs-context gap at 224 Mbit is
   +0.0075/+0.0071/+0.0073 across seeds (seed spread ±0.0004); at 455 Mbit +0.0010–0.0022;
   at 923 Mbit ≈ 0 (sign still consistent).
3. **Crossover ≈ 12% of raw**: richer budgets favor plain MSE with the OMP encoder (down to
   +0.0018 at 16.7%), while the context arm plateaus at ~+0.005 — the metric's i.i.d.-unigram /
   pre-rotary approximation floor binds once near-exact reconstruction is affordable. Candidate
   refinements: co-occurrence-corrected q, rotary inside the training objective, blended MSE+ctx loss.
4. **Honest flag**: at n=8192 the *linear encoder* training degenerates (FVU 1.18; the atoms are
   fine — OMP on the same dictionary reaches +0.0020). Encoder instability at extreme
   overcompleteness, not an atoms failure.

Current overall Pareto frontier: **OV-context dictionaries from 183–614 Mbit, MSE+OMP dictionaries
from 923 Mbit up**; SVD, merges, the global partition, and the two-stage arm are dominated everywhere.

## 3c. The error-analysis arc (ticks 161–166): exploration → diagnosis → solutions

![redrawn frontier](fig_qk_hybrid.png)

Logan's redirect ("look at the residual itself, then consider solutions") replaced blind
objective-tweaking with a diagnose-then-fix loop. The chain, each step feeding the next:

1. **Reader co-adaptation is a null** (tick 161): jointly training a LoRA on the OV reader
   with the dictionary — against a faithful match-the-original-delivery objective — buys
   nothing at any budget; migration meters all quiet. The original OV is already the right
   reader; the QK-side pattern is what carries the loss.
2. **Error exploration** (tick 164, on the 183-Mbit arm): the net cost is a thin difference
   of large flows (46% of predictions improve; the worst 1% of positions carry ~93% of net).
   Weight-space attribution is extremely token-concentrated — the top 50 tokens (newline,
   punctuation, function words) carry 52% of delivered pattern error, by exposure, not
   misfit (their rows are fit *better* than average). CE damage bills on rare continuations
   (compound names, list structure) that depended on those structural anchors. Head 3 alone
   carries 40%.
3. **Rotary diagnosis** (tick 163): including rotation in the objective *coherently* (offset-
   averaged mean) washes out 98.8% of the systematic signal — that formulation trains on a
   DC remnant and loses. The **incoherent** form (T²·E_Δ‖μ_Δ‖², all bands preserved) wins:
   +0.0047 vs +0.0055 at 455 Mbit, and +0.0028 vs +0.0052 at the old 1242-Mbit plateau.
4. **Solutions** (ticks 165–166): exact factor rows for the top-B anchor tokens (bits
   charged) recover the tail — causal confirmation of (2). Nulls that sharpen the story:
   per-head budget reallocation, tail-weighted query distribution, co-occurrence context
   weights, and the MSE-blend once incoherent rotary is in.

**The composed frontier** (incoherent-rotary dictionaries + exact anchors, FineWeb 307k):

| Mbit | 192 | 220 | 262 | 334 | 493 | 606 | 1074 | 1393 |
|---|---|---|---|---|---|---|---|---|
| hybrid ΔCE | .0044 | .0037 | .0036 | .0029 | **.0024** | .0019 | **.0011** | .0010 |
| old frontier at same bits | ~.0072 | .0069 | ~.0070 | ~.0067 | .0054 | ~.0048 | ~.0031 | .0018 |

Seed-robust (+0.0024/+0.0022/+0.0022 at 493 Mbit). The hybrid dominates every previously
measured arm at every budget — 1.8–2.9× lower ΔCE at matched bits — and its 1074-Mbit point
(+0.0011) is below the old frontier's best result at any size. Files: qk_err_explore*.{py,md,json},
qk_rot_diag.{py,json}, qk_solutions.{py,json}, qk_hybrid_frontier.{py,json}, fig_qk_hybrid.py.

## 4. Are the atoms meaningful? Yes — and semantic, not just morphological

Full dump: [qk_dict_features.md](qk_dict_features.md) (6 head-branches, most-used + random atoms,
top tokens by coefficient). Expectation from earlier qualitative work was morphology at layer 0;
the reality is **topic-level semantics alongside morphology**. Examples from head 0, branch 1:

- **music**: musician, music, song, songs, tunes, concerts, band, album, guitarist
- **film**: films, movie, director, cinema, filmmakers
- **food**: restaurant, cuisine, meal, culinary, menu, chefs
- **television**: TV, NBC, CBS, ITV, aired, episode
- **religion**: church, pastor, Christians, theological, sermon
- **persuasion**: persuade, convince, influence, swayed, deceive
- **disasters/places**: Orleans, Katrina, Louisiana, FEMA, hurricanes, Tripoli, Gaddafi
- morphology in the same dictionary: plural suffixes (ups/ins/ures/nesses — and a separate
  *negative-signed* plural atom in branch 2), past-tense suffixes (ered/ised/ized/ated),
  "-ical" adjectives, truncated stems (Ġinst/Ġresear/Ġreconc), first names, surnames,
  3-digit numbers, hedging adverbs (basically/actually/just), quantity words (Two/Three/triple).

So the first attention layer reads the embedding in a basis whose axes are recognizable token
categories — the compression is interpretable, not just compact.

## 5. Why did FVU and ΔCE decouple? (metric ladder, weight-only)

Question raised when dictionaries beat SVD behaviorally while (off-distribution / small-audit)
numbers looked contradictory. Ladder of six structural metrics per arm, Spearman-correlated with
FineWeb ΔCE across 8 arms — all computed from weights alone:

| metric | Spearman vs FineWeb ΔCE |
|---|---|
| **plain factor FVU** | **0.952** |
| **context-expected OV metric** (`pat_ctx`; T-scatter + T²-mean split, see `ov_metric_explainer.md`) | **0.905** |
| **frequency-weighted pattern FVU** (unigram rows × columns) | **0.905** |
| score-level FVU (q̂k̂ᵀ) | 0.881 |
| pattern FVU + rotary offsets (pair-count weighted) | 0.786 |
| pattern FVU (s₁·s₂ product) | 0.714 |
| pattern + rotary + OV-weighted | 0.714 |
| OV-weighted pattern (columns × ‖W_o W_v ê_j‖) | 0.571 |
| OV-**Gram** pattern (error through the full OV map, exact) | 0.571 |
| OV-Gram + rotary | 0.571 |

Findings: (a) on-distribution, the decoupling **mostly dissolves** — plain FVU is a good proxy
(panel C of the figure); (b) the OV-weighting hypothesis (weight score errors by what the
output-value circuit reads) is **not supported**, and not because of the crude norm
approximation — the exact OV-Gram version (cancellation and null space handled properly) predicts
identically badly; (c) rotary position helps the pattern metric (0.71 → 0.79) but doesn't close
the gap.

**Why the composed metrics fail — two mechanisms, quantified by the diagnostics:**

1. *Uniform-vocabulary sampling.* Score/pattern energy concentrates on high-norm factor rows,
   over-representing rare tokens relative to real usage. Weighting rows and columns by empirical
   unigram frequency rescues the pattern metric from 0.714 to **0.905** and makes it correctly
   rank the dictionary above matched-bits SVD.
2. *Differential cancellation through OV.* The cancellation index (‖ΔP·U‖² over the no-cross-term
   sum; the true pattern's own value is 31.6) shows SVD residuals self-cancel through the OV map
   more than dictionary residuals (≈10–11 vs ≈13–14; merges ≈16). Any post-OV energy metric
   therefore awards SVD a discount that held-out cross-entropy does not honor. The alignment
   coefficient (+0.20…+0.30 for every arm) acquits the "dumps error where OV cares" hypothesis.

Practical rule: trust factor FVU or frequency-weighted pattern FVU in search loops; report the
cancellation index beside any post-OV metric — a large cancel-index gap between arms flags a
distorted comparison. Held-out ΔCE (FineWeb) stays binding.

## 5b. Per-head free merges (is any head's query/key content-free?)

Collapsing one head's factor rows to the vocabulary mean (pattern becomes position-only through
rotary), others exact, FineWeb ΔCE per head:

| head | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| ΔCE | +.103 | +.004 | **+.002** | +.013 | +.007 | **+.001** | +.004 | +.019 | +.005 |

**Heads 2 and 5 are individually content-free and — unusually for this program — compose**
(joint collapse +0.0028 ≈ additive). Head 0 alone carries +0.103; collapsing all nine costs
+0.57. The earlier "7 of 9 heads have marginal alphabet 1" claim was a Pile-audit artifact and
does not survive on the training distribution.

## 5c. CE-training upper bound (is the MSE objective leaving CE on the table?)

Frozen-support CE polish through the frozen model (atoms + coefficients + biases trainable,
supports fixed; FineWeb 300/300 train/audit split; not weight-only — diagnostic): **zero gain.**
Held-out ΔCE degraded monotonically from the very first eval (+0.012 at step 150 → +0.061 at
step 1200) while train CE fell to ~2.3 — pure overfitting of ~12M dictionary parameters on 154k
train tokens. Best held-out remains the weight-space MSE fit (+0.0076). Replicates the earlier
stream-tables finding that CE polish buys nothing once structure is right. Bounded by the 154k
training tokens available, but the direction was clear from the first evaluation; combined with
factor FVU's 0.95 rank correlation, the weight-faithful objective is not measurably suboptimal.

## 5d. The tensor-network picture

Per head-branch the exact vocab-by-vocab score map is a chain
`token ──[Q: V×128]──[R_δ]──[Kᵀ: 128×V]── token` (rotary node on a 128 bond = the head
dimension; the full pattern is the Hadamard product of the two branch chains, and the value path
hangs U = W_o W_v ê off the same token leg — the token index is a copy node feeding the q, k, v
roles). The dictionary is surgery on the token→factor edge: factor the (V, 256) table through a
new **atom bond**, `token ──[S: V×1024, 8-sparse]──[D: 1024×256]──`. The bond is *wider* than
what it replaces (overcomplete); the bits live in the sparsity of S, not the bond dimension.
All three compression families are this same surgery with different structure on S — SVD = dense
narrow bond (n = r), clustering = one-hot S (the degenerate sparse code, k=1), dictionary = wide
sparse S — so the matched-bits frontier compares bond structures under one accounting.
Caveats: this is node insertion, not a gauge move (lossy, dimension-changing); and sparsity is
not gauge-invariant — the description-length objective is what pins the atom basis (MDL is the
gauge-fixing). For tensor-similarity training: the network-vs-network objective reduces to
weighted Frobenius distance between factor tables with a metric node per leg; the ladder says use
identity (or unigram-frequency) on the token leg and stop contracting before OV (differential
cancellation, section 5).

## 5e. The mechanism ledger (spec Stages 1-3): archetypes = scaffold-token classes

Separate ledger from the compression frontier (section 3c): here the object is the
p-weighted third-moment core of each head's source-side triple rows [k1 | k2 | v], its
sparse code (512-atom head-space SAE, nonnegative, unigram-weighted — the hardened trainer
that passes the planted recovery gate at 1.0), and its symmetric nonnegative CP factors
(tensor power method + deflation — the only fitter of five tried that passes the planted
CP known-answer test, at 0.9998 matched cosine).

Findings, all gated by known-answer controls and nulls:
- Seven of nine heads pass the sketched moment-residual gate; their CP fits are rank-
  monotone (down to 2-5% residual at rank 64 for heads 2/5/6), restart-stable (0.94-1.00),
  and beat column-permutation nulls by 2-10x. Heads 0 and 4 (content-heavy) stay over-gate
  even at doubled dictionary capacity — their third moments have heavier tails; excluded.
- The dominant archetypes are case/form-invariant closed-class categories: head 8 factors
  into {the}, {a/an}, {of}, {and} classes; heads 2/5 into punctuation families (comma,
  period, colon, dash), newline and document-boundary units.
- Quantified convergence with the compression arc: 14-21% of top archetype-loading tokens
  fall in the anchor-256 set (random baseline 0.5%) — 28-42x enrichment. The two ledgers
  independently identify the same scaffold-token population as layer-0 QK's organizing
  structure: exact rows for it buy the frontier; its category interactions ARE the
  third-moment mechanism.
Files: qk_stage1_triple.*, qk_stage23.*, qk_cp_planted.py, qk_h04_refit.*,
qk_arch_anchor_overlap.json, qk_solver_harden.*, qk_planted_synth.py.

## 6. Robustness notes

- Dictionary result is stable across 3 training seeds × 2 encoders (spread ≤ 0.003 nats).
- k-means merges have real seed spread (+0.009…+0.018 wide-audit at K=2048) — less stable than
  the dictionaries.
- Phase-0 planted-structure control (selectivity 2/2, atom recovery cosine 0.986) stands behind
  the solver family; its two pre-registered predictions both held on the real circuit.

## 7. Open next steps (awaiting steer)

(a) dictionary (n, k) sweep for the FineWeb knee; (b) shared atoms across head-branches;
(c) joint product-of-branches decomposition; (d) tensor-similarity weight-space training with
the factor-level metric (now justified by the ladder); (e) the layer-1 object — deferred by
design until this arc settled.

## File map

| file | contents |
|---|---|
| `qk_merge_stage1_l0.py/.json` | stage-one merge frontier (Phase 1) |
| `qk_sae_dict.py/.json` | stage-two dictionary arms + SVD frontier (Phases 2–3) |
| `qk_sae_robust.py/.json` | wide-audit + seed robustness (Phase 4) |
| `qk_audit_big.py/.json` | 262k-Pile + 307k-FineWeb audits; saves seed-0 dictionary |
| `qk_fw_fill.py/.json` | remaining arms on FineWeb (completes the frontier) |
| `qk_dict_features.py/.md` | atom → top-token dumps |
| `qk_ovweight.py/.json` | six-rung metric ladder + correlations |
| `qk_sae_lib.py` | consolidated solver recipes |
| `fig_qk_mdl_frontier_fw.py/.png` | the frontier figure (training distribution) |
| `fig_qk_mdl_frontier.py/.png` | v1 figure (original Pile audit — superseded, kept for the record) |

### 5f. Per-head capacity frontier of the mechanism ledger (tick 181)

![capacity frontier](fig_qk_capacity.png)

Logan asked whether the seven gated heads really all need 512 atoms, and what the
trade-off is between total atoms $m$ and features-per-token $k$. Sweep: per head, per
$k \in \{1,2,4,8\}$, ascending $m$ until the sketched-moment gate ($<0.05$) passes;
plus usage-ranked pruning curves from one oversized dictionary per head.

**Minimal atoms passing the gate** (bits-optimal configuration bolded):

| head | $k=1$ | $k=2$ | $k=4$ | $k=8$ | bits-optimal |
|---|---|---|---|---|---|
| 0 | – | – | – | 4096 | $k{=}8$, $m{=}4096$: 68 Mbit |
| 1 | 1024 | **512** | 512 | 512 | $k{=}2$: 10.4 Mbit |
| 2 | **32** | 32 | 32 | 32 | $k{=}1$: 2.3 Mbit |
| 3 | 1024 | **512** | 512 | 512 | $k{=}2$: 10.4 Mbit |
| 4 | – | – | – | (4096)* | $k{=}8$, $m{=}4096$: 68 Mbit |
| 5 | 512 | **256** | 256 | 256 | $k{=}2$: 7.2 Mbit |
| 6 | **256** | 128 | 128 | 128 | $k{=}1$: 5.2 Mbit |
| 7 | 2048 | 1024 | **512** | 512 | $k{=}4$: 14.5 Mbit |
| 8 | **128** | 128 | 128 | 128 | $k{=}1$: 3.5 Mbit |

\* head 4's $k{=}8$ ladder was falsely abandoned by the decay-projection heuristic at
$m{=}256$ (9000-step trainings decay slower early); tick 180's direct 12000-step
measurement at $m{=}4096$ passes at 0.0293, so its entry is taken from there. Head 0's
$k{=}2$ ladder reaches 0.060 at $m{=}4096$ — just misses.

**Findings.** (1) Required capacity spans a **128-fold range across heads** (32 to
4096 atoms): head 2 is trivially compressible (passes at 32 atoms even with one
feature per token, residual 0.002), heads 8 and 6 need 128–256, heads 1/3/5/7 need
256–1024, heads 0/4 need 4096. Uniform 512 was over-provisioned for four heads and
under-provisioned for two. (2) **Two features per token is the sweet spot**: going
from $k{=}1$ to $k{=}2$ halves the required $m$ on heads 1, 3, 5, 6, 7; $k{=}4$ helps
only head 7; $k{=}8$ never helps. The rows behave like one dominant class plus one
modifier. (3) **Retraining at the right size beats pruning by roughly an order of
magnitude in residual** (for example head 8: 128 retrained atoms give 0.042, the top
128 of a trained 2048-atom dictionary give 0.344) — the small dictionaries find
genuinely different, coarser atoms, so "train big and prune" is not a substitute.
(4) Bits-optimal mechanism ledger: seven gated heads at per-head optima cost 53.5 Mbit
versus 131 Mbit at uniform $(512, k{=}6)$ — 2.4× cheaper for the same gates.

### 5g. Corrected permutation-null statistic; heads 0/4 verdict revised (ticks 182–183)

Tick 180 reported that heads 0 and 4's cores "fail the permutation null" (real core
fits CP worse than the permuted core). The head-5 control in the asymmetric setting
exposed that statistic as broken: a mode-permuted core approaches the product of
independent marginals, which is intrinsically near-low-rank, so it can fit *better*
than a structured real core. Comparing fit quality across two different target tensors
was invalid.

**Corrected statistic** — everything scored on the *same* real core: factors fit on the
permuted core are transplanted onto the real core (only the nonnegative component
weights refit by Gram solve), versus the real fit, versus a rank-one
product-of-marginals baseline:

| head (form) | real fit | null factors on real core | marginals rank-1 |
|---|---|---|---|
| 0 (asymmetric, $m{=}2048$/mode) | **0.281** | 1.000 | 0.997 |
| 4 (asymmetric, $m{=}1024$/mode) | **0.291** | 1.000 | 0.995 |
| 5 (asymmetric control, $m{=}128$/mode) | **0.132** | 0.911 | 0.996 |
| 0 (symmetric, $m{=}4096$) | **0.389** | 1.000 | 0.996 |
| 4 (symmetric, $m{=}4096$) | **0.530** | 1.000 | 0.999 |

Null-derived directions explain essentially nothing of the real cores (relative error
0.91–1.00 ≈ predicting zero), while the real fits explain 71–87% of core mass in the
asymmetric form. **All nine heads therefore have genuine, null-beating interaction
structure**; heads 0 and 4 are not "structureless" — they need a larger feature
inventory (tick 181) and prefer the mode-separated asymmetric form (rank-32 error 0.28
versus 0.39, and 0.29 versus 0.53, at half or quarter the atoms per mode). The
token-space branch asymmetry of their components is real but partial (mean cosine
between branch-1 and branch-2 token loadings 0.44–0.61): archetypes are neither
mirror-symmetric classes nor unrelated pairs.

### 5h. Corpus-component decomposition of the mechanism cores (tick 185)

Twelve document components (k-means over token-cluster histograms of the 6000 held-out
co-occurrence documents; named by over-represented tokens — e.g. commerce/product
reviews, health/legal, one small Cyrillic outlier, one game/list outlier). Per head,
the third-moment core is rebuilt under each component's token distribution (codes
fixed) and every archetype profiled across components.

**The archetype structure is corpus-general, as the scaffold interpretation predicts:**
mean effective number of components per archetype is 9.7–10.4 out of 12 for every head
(near-uniform spread). The most concentrated single archetypes reach effective 3.6–4.5
components — a modest topical minority. Component-core cosines generalize the two-slice
stability result: the seven gated heads sit at mean 0.84–0.99 across all 66 component
pairs, while **heads 0 and 4 vary far more across data components** (mean 0.77–0.80,
minima 0.18–0.24, driven by the outlier components) — consistent with their long
archetype tail being partly component-specific structure on top of a shared scaffold.

### 5i. Joint training vindicated by the true warm start (tick 186)

Tick 177's joint-training negative (CP structure degrading 4–15×) carried a warm-start
caveat: its archetype matrix was initialized from random rows of the code matrix. Rerun
with a single variable changed — the archetype matrix initialized from the true
deflation solution, scaled so the CP model equals the stagewise core exactly at step
zero — **joint training now improves every head**:

| head | stagewise CP rel-err | joint (tick 177, random-ish init) | joint (true warm start) |
|---|---|---|---|
| 2 | 0.031 | 0.452 | **0.030** |
| 8 | 0.121 | 0.540 | **0.101** |
| 1 | 0.188 | 0.648 | **0.146** |

Moment residuals stay gated (0.007–0.039), reconstruction is unchanged, archetype drift
is minimal (matched cosine 0.97–1.00 to the stagewise archetypes), and the trained
archetype matrix agrees with a fresh refit on the joint core. Verdict revised: the
gamma-ramped joint objective is a working **final polish stage** (10–22% better CP fit
at held gates) — the tick-177 collapse was entirely a warm-start artifact.

**Extension to heads 0/4 (tick 187).** The same warm-started polish applied to the
asymmetric mode-separated form: head 4 improves 0.238 → 0.194 at rank 64 with the gate
held (0.025 → 0.031); head 0 improves 0.228 → 0.179 at a gentler gamma of 0.025
(0.16 was reachable at gamma 0.05 but breached the gate by 0.001; gamma must respect
each head's gate margin). Factor drift 0.94–0.97 throughout — the polish sharpens the
existing archetypes rather than replacing them. The joint objective has now improved
the core fit on all five heads it has been applied to.

### 5j. Minimal inventories validated; no layer-0 copy heads on the direct path (tick 188)

The seven scaffold heads retrained at their bits-optimal minimal configurations
(32–512 features, 1–4 per token), warm-started-joint-polished (gamma 0.025), and
re-validated: all moment gates hold (0.003–0.048) and all corrected nulls fail
decisively (null factors 0.94–1.00 on the real cores). Trade-off measured: at minimal
capacity some cores are less CP-compressible than at the 512-feature standard (head 1:
0.33 vs 0.19; head 8: 0.27 vs 0.12; head 7 improves to 0.03) — density of features
trades against archetype crispness. Minimal inventories are also *more* interpretable
in places (head 3 gains a make/made/making verb-lemma class; head 7 a 4/four number
class; head 1 clean we/you/she person classes).

**Copy-head question (Logan): answered negative at layer 0.** Direct-path logit-lens
test — each archetype's write vector decoded through W_o into the unembedding, boosted
profile compared against the attended class — gives copy cosines of −0.08 to +0.03
across every archetype of every head (per-head means ≈ 0). No layer-0 head is a
direct-path copy head; the archetypes deposit class-occurrence signals consumed by
layers 1–17 (analysis deferred). Weak exception worth noting: head 2's abstract-noun
classes reach ≈ 0.10. Per-archetype branch agreement for heads 0/4 (post-polish):
scaffold components ≈ 0.9–1.0, fringe components near 0 (means 0.43 / 0.60) —
quantifying which parts of the hard heads genuinely use the bilinear asymmetry.

### 5k. Causal per-archetype ablation on held-out text (tick 190)

Ablation = project the archetype's key channel out of the head's exact key tables on
both branches (structured zero); score = per-position cross-entropy delta over 64
held-out documents (33k predictions); first ten archetypes per head (90 ablations).

**Weight mass and causal usefulness decouple** (the program's third measured
decoupling): heads 3, 6, 7, 8 carry real per-archetype causal load — head 3's channels
cost 0.009–0.059 nats on average and up to +7.6 nats on single predictions ("…in its
cold war → rivalry"); head 8's cost ~0.0015 mean with worst cases like "…Graph Search
is in search → of" (+2.5) — while heads 1, 2, 5, 0, 4's channels are individually
near-zero on ordinary text (mean |ΔCE| < 0.0004): redundant at the margin despite
carrying large core mass. Second structural finding: within a head, the hardest-hit
position is often identical across archetypes (all ten of head 8's channels break the
same "in search of"; all of head 6's break "…Exhibitions Ltd. → of"), i.e. the
archetype channels overlap on a shared key direction, and specific predictions lean on
that single layer-0 signal. Role confirmations are direct: the scaffold heads' worst
failures are exactly preposition/continuation completions after function-word context.
Artifact updated with per-archetype ablation panels (mean ΔCE + hardest-hit passages).

### 5l. Group ablations: overlap, head hierarchy, and where the tail lives (tick 191)

Four numbers per head on the tick-190 evaluation set: sum of ten single-channel
damages, damage from removing the ten-channel SPAN at once, a matched-dimension
random-subspace control, and whole-head silencing.

| head | singles sum | group (10-dim) | random | whole head |
|---|---|---|---|---|
| 3 | +0.251 | +0.070 | +0.002 | **+0.083** |
| 7 | +0.016 | +0.005 | 0 | +0.0066 |
| 6 | +0.016 | +0.003 | 0 | +0.0035 |
| 8 | +0.013 | +0.003 | 0 | +0.0034 |
| 0 | +0.0004 | +0.0003 | 0 | +0.0028 |
| 4 | +0.0006 | +0.0011 | 0 | +0.0018 |
| 1 / 2 / 5 | ≈0 | ≈0 | 0 | +0.0001–0.0003 |

Findings: (1) **sub-additivity** — the group is smaller than the sum of singles on
every load-bearing head, confirming the channels overlap (singles double-count a shared
direction); (2) the ten-archetype span carries **73–88% of whole-head causal load** on
heads 3/6/7/8 — the archetype description is causally near-complete there; (3) random
subspaces do nothing: the directions matter, not the dimension; (4) heads 1, 2, 5 barely
matter even fully silenced (+0.0001–0.0003) — their near-zero singles meant "small
head", not redundancy; (5) heads 0/4's small causal budget mostly lives BEYOND the
top-10 archetypes (span carries 11–60% of whole-head) — the long tail is where their
function is; (6) the hierarchy inverts capacity: head 3 (512 features) carries roughly
20× the causal load of everything else combined, while the 4096-feature heads are minor
and the trivial 32-feature head 2 is negligible. Note: evaluated on the 64-document
subset; whole-head numbers are one-head-at-a-time (interactions unmeasured).

### 5m. Full-audit head importance + a weight-space correlate found (tick 192)

Whole-head ablations re-scored on the full 307k-prediction audit (10× tick 191's text):
h3 +0.0780, h7 +0.0090, h8 +0.0051, h6 +0.0041, h0 +0.0026, h4 +0.0016, h5 +0.0014,
h1 +0.0006, h2 +0.0005. Logan's suspicion confirmed in degree: the quiet heads rose
3–5× with more text (h5 0.0003→0.0014) — every head matters somewhat, none are zero —
but the hierarchy stands: head 3 remains ~8× the runner-up and ~60% of the layer's
total causal load.

**Correlate found: expected output magnitude.** Weight-only candidates vs causal
importance (Spearman): ov_norm = Σ_t p_t ‖W_o^h v_t‖ → **+0.87** (best); pattern×write
composite +0.80; expected squared pattern +0.75; third-moment core scale +0.65; key
table norms uninformative (unit-RMS makes them constant). The causally dominant head is
visible in the weights: h3's expected output magnitude (1430) is 3–6× every other head
(224–545). So "how hard does this head write, frequency-weighted" is a good cheap
proxy, with the known caveat that heads 0/4's mechanism mass still overstates them.

Layer-1 architecture (recon): each block = bilinear attention + a Bilinear MLP
(Left/Right 4608×1152 gating, Down projection) — NOT attention-only; the layer-1
program must account for block-0's MLP in the residual.

## 6. Layer 1 (opened by Logan, tick 192; ticks 193–)

### 6a. The token-identity port test (tick 193)

Layer 1's attention input is the post-block-0 residual (attention + Bilinear MLP +
lambda-mixed skip), so the exact embedding fold no longer applies. Test: token-
conditional MEAN-RESIDUAL tables (average block-1 input per vocabulary token, estimated
on 1024 disjoint co-occurrence documents, 62% of types / 94% of audit token mass seen;
unseen tokens fall back to embedding rows), pushed through block 1's own projections
with the model's per-head normalization, patched into layer 1's pattern, audited on the
standard 307k set.

**Layer 1's pattern is ~99% token-identity-driven:** replacing the entire layer-1
pattern with static token tables costs only **+0.027 nats**, against **+2.70 nats** for
zeroing the layer-1 pattern outright — a 100× ratio. The whole validated pipeline
(fold → dictionaries → third-moment cores → archetypes, with all gates) therefore
transfers to layer 1 with mean-residual tables in place of embeddings. Also notable:
layer 1's pattern is ~27× more causally important than layer 0's entire attention
(+2.70 vs ~+0.10 summed) — and massively super-additive across heads: single-head
zeroing sums to only +0.128 (h1 +0.065, h4 +0.020, h8 +0.017, h3 +0.011, rest ≤0.006),
21× below the joint effect, so layer-1 heads back each other up in a way layer-0 heads
do not.

### 6b. The layer-1 mechanism ledger: all nine heads validate (ticks 194–196)

Stage 1 (token tables from shrinkage-estimated mean residuals): all nine heads pass the
moment gate — seven at 512 features, head 3 at 1024, head 1 at 1024 on shrunk tables
(its raw-mean divergence was rare-token estimation noise cubed by the moment; tick 195).
Stages 2–3 (cores, rank-32 symmetric CP): every head beats the corrected null (real
fits 0.10–0.52 vs null-factors-on-real 0.54–1.00), restart stability 0.96–1.00.

**Layer 1 speaks a different archetype vocabulary than layer 0.** Layer 0 is
function-word scaffold ({the}, {a/an}, {of}); layer 1's top archetypes are
sentence/discourse boundaries (heads 0, 4, 7: period/quote/exclamation families),
document boundaries (heads 2, 5, 6: newline/end-of-text), quote/bracket openers
(head 8), and — on head 1, the causally dominant layer-1 head — **mid-word subword
fragments** ('cknowled', 'theless', 'secut'), consistent with a subword-continuation
role, which explains both its causal weight and its context-dependence. Head 3 fits
worst (0.52, long-tail-ish, layer-1's analog of the layer-0 hard heads).

### 6c. Who reads layer-0 head 3? Everyone, a little — and layer 1 partially self-repairs (tick 197)

Sensitivity: silencing l0-h3 during mean-residual estimation moves EVERY layer-1 head's
tables by 10–30% (value modes most, 0.19–0.30) — no dedicated reader; the determiner
signal is broadcast infrastructure. The archetype loadings that shift most are
function-word classes across all nine heads (the/a/that/comma positions), i.e. h3's
writes specifically shape how layer 1 keys function-word positions. The subword-
continuation head (l1-h1) reads h3 least (0.11–0.15), as its role predicts.

Path accounting of h3's +0.078: rebuilding layer 1's pattern tables from h3-less
residuals (model otherwise normal) costs +0.028 beyond the port baseline — roughly
**one-third of h3's effect flows through layer-1 pattern formation**, two-thirds
through values/MLP/deeper layers. Striking interaction: zeroing h3 while SHIELDING
layer 1's pattern with normal static tables costs +0.171 — more than double h3's live
effect (+0.078) — i.e. the live, context-computed layer-1 pattern partially
COMPENSATES for the missing h3 signal, and freezing it removes that self-repair
(plus a large positive interaction between the two corruptions; both readings logged).

### 5n. Equal-ablation control: archetype directions are NOT causally privileged (tick 199)

Logan's question: do head 3's archetype ablations differ from an equal ablation? Six
arms on head 3, all scored by pattern energy removed (E[(ΔP)²] over 16M sampled token
pairs) and per-position damage on 128 documents: top-1 archetype, top-10 archetype
span, top-10 PCA of the p-weighted key tables, random 10-dim subspace, and two uniform
score shrinks calibrated to match the archetype arms' removed energy exactly.

**Damage is proportional to pattern energy removed, regardless of direction.** Damage
per unit energy: archetype-10 2.14, PCA-10 2.41, energy-matched shrink 2.44, shrink
matched to archetype-1 1.58 vs archetype-1 1.05 — the archetype directions do no
*more* damage per unit than generic or uniform removals (slightly less). Concentration
is identical across all arms (top-1% of positions carry ~16–17% of damage everywhere;
hit fractions 0.79–0.88): damage concentration is a property of the head and the data,
not of which directions are removed. (Random-10 removes 77× less energy than the
archetype span — which retroactively explains tick 191's null random control.)

Reading: head 3's downstream consumption is approximately isotropic in pattern space —
the big directions ARE the used directions (consistent with §5m's expected-output-
magnitude correlate). The archetypes' value is descriptive, compressive, and predictive
(naming classes, minimal inventories, where-it-matters text), not a causal privilege
per unit of pattern. The sparse-interaction hypothesis is not supported at the level of
cross-entropy damage for this head. Clean negative, honestly logged.

### 6d. The context-dependent remainder: mid-sized, half low-rank, written by the MLP (tick 200)

Per-position deviations of layer 1's actual query/key factors from their token-table
values, across all nine heads (65k positions): (1) SIZE — the context part is 21–41%
of factor norm (yet only +0.027 nats of function: the pattern product and downstream
layers tolerate most of it). (2) RANK — the top 16 of 128 directions carry 44–64% of
deviation variance: a 16-dimensional context summary halves the remainder, but there
is no sharp low-rank cliff. (3) SOURCE — ridge regression onto upstream signals is
decisive and uniform: the block-0 **Bilinear MLP output explains 45–64%** of the
deviation, layer-0 attention outputs only 21–35%, both together 51–68%. Layer 1's
context-sensitivity is chiefly authored by the block-0 MLP, not by the attention heads
whose ledger we built — and a third to a half of it is not linearly explained by
either, the first object in the program that resists all current machinery.

## 7. The block-0 Bilinear MLP (opened by Logan; ticks 201–)

### 7a. Reconnaissance: dense in the neuron basis (tick 201)

The MLP is already a written-in-weights CP tensor of rank 4608 (out =
Down(Left(h)·Right(h))). Recon verdict: **no neuron-level sparsity anywhere.**
(1) Usage spectrum is flat: the top 128 neurons carry 6% of write-weighted usage, the
top half only 52%. (2) The neuron-count frontier is harsh: keeping the top half costs
+0.030 nats, top quarter +0.115, top 256 +0.783 — every neuron tier is doing work
(contrast: all of layer-0 attention is +0.10 total). (3) Logan's next-circuit reader
test, weight-space: the layer-1 query/key read maps touch essentially ALL neurons
(effective count 4361–4568 of 4608; top-256 neuron sets overlap only 0.17 between
heads) — the MLP→layer-1-QK channel is NOT neuron-sparse. (4) Most neurons are
context-driven: split-half token-identity R² has median 0.34 with only 16% of neurons
above 0.5 — consistent with the MLP being the author of layer 1's context part.

Conclusion: the interesting structure, if any, is basis-free — low RANK rather than
few neurons. Next: eigendecomposition of the reader-composed bilinear forms.

### 7b. Weight-space rank analysis: channels are only mildly compressible (tick 202)

Singular spectra of all 36 composed MLP→layer-1-QK read channels (each ≤ rank 128):
effective rank 12–101, median 68; median rank-for-90%-energy 92. Eigenfeature spectra
of representative read directions are flat (top-12 of 1152 eigenvalues carry only
17–26% of energy), and most lead eigenvectors align with junk/rare-token directions —
one interpretable hit: the lexical reader's lead feature is a change-of-state verb
class (get/came/became/create). Calibration: zeroing the whole block-0 MLP costs
+2.50 nats — as large an object as layer-1's entire pattern (+2.70). Verdict so far:
the MLP resists neuron-sparsity AND weight-space rank-sparsity; remaining hypothesis
is that compression lives on the DATA MANIFOLD (weight-space ranks count directions
the data never visits) — tick 203 measures data-weighted channel ranks.

### 7c. The channels collapse on the data manifold (tick 203)

Realized covariance of each MLP→layer-1-reader channel output on held-out text:
**effective rank median 10** (min 1, max 62) versus 68 in weight space — the manifold
compression the weight ball hides. Channel outputs are 45–95% token-identity-explained
(median 0.56), so what each layer-1 reader actually receives from the MLP is mostly a
vocabulary signal plus a ~10-effective-dimensional context signal. Converges with
§6d (16 dimensions halve layer-1's factor deviations): the context computation flowing
from the block-0 MLP into layer 1's pattern is low-dimensional in practice, even
though the MLP computing it is dense and high-rank in weight space. The compact
two-circuit object is now concrete: per reader, a token table plus a rank-≈10 context
adapter — tick 204 fits and audits exactly that.

### 7d. The compact two-circuit object, priced (tick 204)

Full-model audits (manual 18-layer forward, verified to reproduce the reference
baseline to five decimals) with layer 1's pre-rotary factors replaced by
token table + rank-r context adapter (top-r PCA of realized deviations, estimated on
disjoint data), all four maps, all nine heads:

| context rank r per channel | held-out ΔCE |
|---|---|
| 0 (pure static tables) | +0.0515 |
| 4 | +0.0208 |
| 16 | +0.0113 |
| 64 | +0.0009 |

Sixteen context dimensions per channel recover 78% of the static gap; sixty-four
recover 98%. The adapter bases cost only ~2.4 Mbit total. Honest framing: this prices
the INTERFACE — the context information layer 1's pattern actually consumes is
~16-dimensional per channel — but the adapter projects the true factors at runtime,
so it compresses the channel, not the MLP computation itself. The natural completion
(open): a linear generator for those dimensions from a low-rank projection of the MLP
output, which the tick-203 manifold collapse (median effective rank 10) says should
exist. (Note: rank-0 here is +0.052 vs the +0.027 raw-mean port of §6a — shrinkage
tables trade a little function for moment robustness; both numbers stand, different
table estimators.)

### 7e. The generated adapter: linear generation recovers a third; the rest is nonlinear (tick 205)

Replacing tick 204's oracle projection with a fully generative pipeline — one shared
64-dimensional principal subspace of the block-0 MLP output, plus a ridge-fit linear
map per channel (64→16), nothing reading the true factors — audits at **+0.0365**,
versus +0.0515 static and +0.0113 for the oracle rank-16 projection. So a purely
linear read of the MLP output recovers **29% of the context gap** (oracle interface:
78%). The shortfall matches the tick-200 linear-explainability ceiling (R² 0.51–0.68):
the 16 dimensions layer 1 consumes are only partly a linear functional of the MLP
output — the remainder involves the interleaved normalizations and sources beyond a
single linear view. Arc summary: the block-0 MLP is a dense, high-rank computation
(no neuron or weight-rank sparsity) whose FUNCTIONAL role for the next QK circuit is
a ~16-dimensional, mostly-token-identity signal; that interface is now named, priced
(~2.4 Mbit), and two-thirds of its context content awaits a nonlinear generator.

### 7f. Rank claims data-verified; weight-space block fold (tick 206)

**(A) The manifold-rank claim survives 16× more data.** Streamed channel covariances
at 32k / 131k / 524k positions: median effective rank 10.4 → 10.5 → 10.7 (max ~62,
r90 57–58, all stable). "Effective rank ≈ 10" is not a small-sample artifact.

**(B) Weight-space block decomposition** (weights + frozen unigram only; the bilinear
form splits exactly over writers, the per-position normalization being a shared
positive gauge): across the 18 q1/k1 reader channels, the embedding×embedding block —
a weight-exact token table — carries **69–93% (median 84%)** of channel second moment;
attention×attention 5–26%; embedding×attention cross terms only ~3% — and **layer-0
head 3 tops the cross block in all 18 of 18 channels**, the weight-space confirmation
of both the output-magnitude correlate and the broadcast finding. Coherent with the
data-side token-identity share (0.56 median): both views agree the MLP→layer-1-QK
channel is mostly token identity with an attention-driven remainder led by head 3.

### 7g. Composed-tensor CP in pure weight space: dense — the measure is the message (tick 207)

Joint CP (learned left token class × right token class × output direction, no inherited
basis, ZERO data — not even unigram) on the composed tensor G[o,s,t] = Σ_j A_oj
(L_j·ê_s)(R_j·ê_t) for four reader channels. Verdict: **no sparse embedding-pair
structure in unweighted weight space.** Rank-32 relative error 0.74–0.89 (the object
barely compresses); fitted token classes are essentially uniform (top-16 token mass
0.2–0.3% of 50k); and the corrected null TIES the real fit (0.81 vs 0.81) — what
little is captured is generic spectral bulk, not token-interaction structure. The
dense-subset consistency check validates the fitter (factored = dense to 1%). One
genuine signal: the subword-giant channel's top output direction aligns at cosine 0.86
with that head's own fitted archetype detector — the output axes are right even when
the token modes are not. This is the program's third demonstration that raw
weight-space objects are diffuse and the MEASURE concentrates them (folding; manifold
rank); tick 208 refits the same object under the frozen unigram weighting to quantify
exactly how much sparsity the measure buys.

### 7h. The weighted refit: sparsity appears, but the null still ties — composed-CP line closed (tick 208)

Under the frozen unigram weighting the fit improves (rank-32 relative error 0.48–0.67
vs 0.74–0.89 unweighted) and the token classes become concentrated and readable
(top-16 mass 37–77%; the classes are comma/the/period — the frequency scaffold). But
the decisive statistic is unchanged: **the corrected null ties the real fit in all
four channels** (e.g. 0.483 null vs 0.485 real). The concentration is frequency
structure, not token-interaction structure: components built on a token-misaligned
tensor explain the real one equally well once weights are refit. Double negative,
cleanly established: the embedding→MLP→layer-1-QK composition carries no
null-beating sparse token-pair structure at this rank, in either measure. This
coheres with the whole MLP arc: the layer is a dense mixer whose functional product
is a low-dimensional manifold signal (Section 7c–7e) — the sparse, nameable structure
of this model lives in the QK factor tables and their archetypes, not inside the MLP
composition. (Caveats logged: rank 32, greedy deflation; the output-mode alignment
with layer-1 archetypes — cosine up to 0.89 — remains the one real positive.)

## 8. Measure calibration: which geometry predicts causal damage (tick 209)

![measure calibration](fig_measure_calibration.png)

Spearman rank correlation between each candidate measure's residual and measured
causal damage, across the stored intervention library.

**Family A — nine whole-head ablations (full audit):** expected output magnitude
ov_norm 0.87 > pattern×write 0.80 > expected squared pattern 0.75 > core scale 0.65 >
uniform table norm 0.50 (degenerate — the fold makes all head tables equal-norm).

**Family B — ninety per-archetype channel ablations:** uniform weight fraction
**0.83 pooled / 0.61 within-head** > unigram-weighted fraction 0.73/0.51 ≈ pattern
energy 0.71/0.55 ≫ **mechanism core mass λ: 0.11 pooled, 0.02 within-head —
essentially uncalibrated.**

Two lessons. First, the moment-versus-function decoupling is now fully quantitative:
the mechanism ledger's own importance weights carry almost no information about causal
damage — the two ledgers must never be conflated, measured at ρ ≈ 0. Second, a
surprise against the simple thesis: at the *channel* level the plain uniform weight
fraction calibrates best (at the *head* level it is degenerate and worst) — which
measure is right depends on the object and intervention family, strengthening the
practical conclusion: calibrate the measure against causal probes for each claim
class, rather than assuming any single geometry (weight, unigram, pattern, or moment)
is universally the right one.

### 7i. The generator zoo: architecture-insensitive — the bottleneck is information, not expressivity (tick 210)

Round 1 failed training hygiene (gated arms diverged without whitening/skip; logged as
training failure, not evidence). Round 2, parameter-matched (~140–150k) with whitened
codes and linear-skip wrappers:

| arm | val R² | full-audit ΔCE |
|---|---|---|
| linear (64-code, 37k) | 0.465 | +0.0363 |
| linear + norm scalars | 0.475 | +0.0358 |
| linear (256-code) | 0.606 | +0.0356 |
| bilinear gate | 0.629 | +0.0334 |
| swiglu gate | 0.643 | +0.0335 |
| single-encoder MLP | 0.638 | +0.0335 |
| hierarchical | 0.624 | +0.0336 |

Findings: (1) **every nonlinear family lands in the same place** (R² 0.62–0.64, audit
+0.0334–0.0336) — the model's own bilinear primitive holds no advantage at matched
capacity, and weight-sparsity statistics don't separate the arms either (top-1% mass
0.07–0.10 everywhere), so the "sparsest fit reveals the prior" test is inconclusive:
no architectural prior is distinguished by this task. (2) Nonlinearity is worth only
~+0.003 over linear on the audit; the generated pipeline plateaus at **45% of the
oracle gap** (+0.0334 vs oracle +0.0113 / static +0.0515) for every family. The
remaining signal is not a function-class problem: it is INFORMATION the 64-dimensional
MLP-output code does not carry (the wider linear code matching nonlinear-arm R²
corroborates). What layer 1's pattern reads beyond this code draws on the attention
outputs, the skip path, or fine MLP structure beyond 64 principal directions — the
interface remains priced at 16 dimensions (oracle), with generation from the natural
code capped near half of it regardless of architecture.

### 7j. Code-composition sweep: attention helps a little; the remainder is entangled (tick 211)

| generator | val R² | full-audit ΔCE |
|---|---|---|
| linear, 128-dim MLP code | 0.544 | +0.0353 |
| linear, 512-dim MLP code | 0.656 | +0.0359 |
| linear, mixed code (MLP-64 + 9×attention-PCA-8 + scalars) | 0.526 | +0.0345 |
| swiglu, mixed code | 0.667 | **+0.0319** |

Two results. (1) The fine MLP spectrum is a dead end for FUNCTION: the 512-dimensional
code fits deviations best among linear arms (R² 0.66) yet audits WORSE than the
128-dimensional code — yet another fit-versus-function decoupling; the extra variance
predicted is variance layer 1 does not consume. (2) The attention-output code is the
real (if modest) missing source: the mixed swiglu pushes past the 64-code plateau to
+0.0319 — the best generated result, 49% of the oracle gap recovered. Marginal returns
per additional named code are now ~0.001–0.002 nats against a ~0.021 remainder: the
ungenerated half of the interface is ENTANGLED across the residual state rather than
concentrated in any small named code. Generator arc closed at: interface 16-dim
(oracle), ~half generatable from named codes (MLP-64 + attention summaries + scalars),
the other half the honest price of not running the dense layer.

### 7k. What the generator misses: mid-word positions, key-side, common structure, coupled across maps (tick 212)

Error analysis of the best generated interface (route-only patching throughout):

**(1) The failures have a name: subword continuation.** At the worst-200 positions the
current token is a mid-word fragment 34% of the time versus 13.7% baseline — 2.5×
enrichment (line boundaries: no enrichment). The generator fails hardest exactly in
layer-1 head 1's subword-continuation territory.

**(2) The missing signal is common structure, not idiosyncrasy.** Worst-position
residuals are 1.8× typical in norm (the generator genuinely fits worse there, beyond
mere sensitivity), and their PCA is strongly low-rank: 16 of 576 dimensions carry 67%.
What is missing is learnable in principle — the inputs, not the targets, are wrong.
Missing mass concentrates on the KEY side of the lexical and content heads (k1 of
layer-1 heads 3, 7, 5; k2 of head 1) — consistent with the hypothesis that the
ungenerated half is fine-grained lexical context (which specific word/fragment
precedes), which lives in high-dimensional token space and cannot fit through a
64+72-dimensional code.

**(3) Partial repairs backfire — the map errors are COUPLED.** Restoring oracle
corrections for single maps: q2 helps (+0.0298 vs +0.0320), but k1 HURTS (+0.0337) and
q1 slightly hurts. Because the pattern is the product of two branch scores, generation
errors across maps partially cancel; fixing one map alone breaks the cancellation.
Any improved generator must be trained jointly against the pattern (or the loss), not
per-map regression — Logan's gradient framing is the right one, and per-map MSE is
provably the wrong loss at this stage.

**(4) The interface is not cheaply nameable.** Distance-to-newline, subword flags, and
position explain only 1–2% of the oracle interface coordinates — the ten-dimensional
signal is low-rank but not a human-obvious position feature, even though its failures
cluster at nameable (subword) positions.

### 7l. Head-level localization and the residual stage (ticks 213–214)

**Cluster classification (tick 213):** the missing signal's dominant class (231 of the
worst 512 positions; k1 channels of the lexical heads) is LEXICAL CONTINUATION —
proper nouns and title completions ("gave Lindsay → L", "Beneath a Granite → Sky").
Missed-link tracing: layer-1 head 1's corrupted attention links sit at offsets 0–2 in
95% of cases — the connection to the fragment immediately behind the current position.

**Per-head joint repairs (tick 214, mode bug fixed with asserts):** restoring oracle
context for all four maps of one head at a time is positive for every head (the
per-map backfires were indeed cross-map coupling within heads), and the ranking is
decisive: **head 1 alone carries 56% of the remaining damage** (repair takes +0.0319
to +0.0203; next best: head 8 −0.0069, head 4 −0.0037; head 2 nil). The generator
problem is now substantially a single-head problem: reproduce layer-1 head 1's
context-dependent keys.

**Residual stage on window token-identity codes (Logan's arm): near-null.** Second-
stage models trained on the stage-1 residual with embedding-PCA-32 codes of tokens at
offsets 0..−3: residual R² only 0.02–0.04 (gated variant overfits); end-to-end
+0.0304 versus +0.0319. Recent-token identity at 32-dim code resolution is NOT the
missing information — consistent with the lexical-continuation story needing
high-resolution token identity (which specific fragment), not a coarse embedding
code. Natural next arm: a per-token CORRECTION TABLE keyed on the previous token
(bigram-style, weight-space-flavored) for head 1's key channels specifically.

### 7m. The bigram null: the missing signal is composed context, not token lookups (tick 215)

The previous-token correction table for head 1's channels explains only 2% of the
stage-1 residual (adding a two-back table makes it worse — overfit), and end-to-end
the gain is zero (+0.03203 versus +0.03206 stage-1; head-1 oracle target +0.0203).
Together with the window-code null (§7l): the missing context for head 1's keys is
not ANY function of the last several token identities — not coarse codes, not
full-resolution single-token lookups. It is genuinely COMPOSED context state (the
word-assembly state block 0's attention and MLP build across positions), which is
why it lives in the dense machinery and resists every table- and code-based
generator. The generator search closes honestly at: static +0.0515 → code-based
generators +0.032 → oracle interface +0.0113, with the remaining half being the
price of composition itself. Hypothesis space fully walked: architecture (null),
code width (null), attention codes (small), window identity (null), bigram identity
(null), head-localization (strong: h1 = 56%).

### 7n. The pruned sliver: layer-1's pattern context is a 16-token local computation (tick 217)

Layer-1 factors computed from a REDUCED block-0 state (attention causally windowed to
W positions; real forward elsewhere; route-only patching), full audits:

| sliver | ΔCE | vs anchors |
|---|---|---|
| W=1 + MLP | +0.363 | worse than static tables (+0.0515) — truncated composition is worse than none |
| W=2 + MLP | +0.153 | |
| W=4 + MLP | +0.070 | |
| W=8 + MLP | +0.028 | **beats the best code generator (+0.032)** |
| W=16 + MLP | **+0.0099** | **beats the 16-dim oracle interface (+0.0113)** |
| W=4, no MLP | +0.667 | the block-0 MLP is essential to the composition |

**Level-3 understanding closes:** the generator that no lookup or code could be is
simply *block 0 itself, run on the last sixteen tokens* — 3% of the attention context
— which reproduces layer-1's entire pattern context beyond even the oracle interface.
The composed multi-token state is LOCAL (16 tokens suffice; 4 do not; 1 is worse than
nothing), and the dense MLP is a mandatory ingredient of the composition, not a
bypassable mixer. Final form of the two-layer pattern circuit: token tables +
archetypes everywhere, plus one named subroutine call — "block-0 on a 16-token
window" — whose internal algorithm remains dense (level 4) but whose input scope,
compute cost, and output law are now all measured.

## 9. Double dissociation tests: how "code-like" are the sub-circuits? (tick 219)

Logan's necessity-and-sufficiency test on a named subsection (the determiner channel),
four conditions (full / channel-removed / channel-only / head-zeroed), damage split by
determiner-context positions vs others, 128 held-out documents.

**Layer 1 (head 7): no dissociation at all.** Channel removal costs +0.0003 (nothing,
even at determiner positions); channel-only equals head-deleted (+0.0577 vs +0.0580).
The ensemble recomputes determiner keying elsewhere — layer 1 is written like
redundant code with no single owner per variable.

**Layer 0 (head 3): partial dissociation.** Removal (T1, +0.0235 of the head's
+0.0795): determiner positions hurt 1.5× more than others (+0.0343 vs +0.0229) —
real but mild mean-level selectivity. Retention (T2): keeping only the determiner
channel preserves 24% of head function — but equally at determiner and other
positions, i.e. NOT selectively sufficient. The clean class-selectivity that exists
lives in the TAIL, not the mean: tick 190's per-channel ablations break specific
completions catastrophically (+4 to +7.6 nats on single predictions) while mean
damage is diffuse.

**Refined answer to the coding analogy:** layer-0 archetype channels are functions
with a few critical call-sites plus a large body of shared incidental work — remove
one and its critical callers fail loudly while everything else degrades slightly;
layer-1 channels have no owned call-sites at all. Neither layer is modular in the
textbook sense; layer 0 is "code with hot paths," layer 1 is "an ensemble." The
variable-level diagrams (§ sub-circuit stories) remain valid as contribution maps;
necessity semantics attach only to layer-0 tail cases and whole heads.

## 10. The generality spectrum: silence is the computation (tick 220)

Logan's hypothesis: uniformly-useful components might reduce to scalars (offset
kernels); broadly-topical ones should show class structure. Both tested on all 18
heads (zero vs offset-kernel replacement; per-position damage split over six context
classes).

**(1) No head is a positional scalar — and the kernel test failed in the most
informative direction possible.** Replacing a head's pattern with its content-free
average offset kernel is almost always far WORSE than deleting the head (content
ratios 4–238; layer-1 head 8: kernel +3.74 versus zero +0.016). In this no-softmax
bilinear regime, a head's pattern is near-zero at most (query, key) pairs and fires
selectively on class matches — so the head's SILENCE is a computed output. Forcing
the average kernel makes it attend indiscriminately, which poisons the residual far
more than absence. The generality of the ensemble is not positional-prior-ness; it is
content-gated sparse firing. One genuine exception: layer-0 head 6 (delimiters),
content ratio 0.65 — the kernel is BETTER than zeroing, so it is part positional
prior; layer-0 head 0 sits near parity (1.66).

**(2) No topic hierarchy at head grain.** Class-enrichment rows are nearly flat
everywhere (top enrichments 1.3–1.6×, almost always capitalized-target or
after-determiner, for every head); no induction specialists in the first two layers
(repeat contexts are 48% of positions and never a head's peak class). The hierarchy
Logan proposed exists WITHIN heads (the archetype classes) but not ACROSS heads by
context class: heads are general engines with mild tilts, differentiated by which
token classes they key, not by which text domains they serve.

Combined with §9: the first two layers' organization is archetype-classes inside
content-gated heads inside a redundant ensemble — with head silence doing much of the
computational work.

### 10b. The diffuse floor identified by elimination: distributed pattern precision (tick 221)

What is the "degrades everything slightly" component of an archetype channel (test
case: l0-h3's determiner channel, removal +0.0235)? Three hypotheses, two rejected:

**H1 (duty cycle) — rejected.** Damage across det-channel-firing deciles is flat
(0.017–0.029, rank correlation 0.19): positions where the channel fires hard suffer
no more from its removal than positions where it idles.

**H2 (bias supply) — rejected.** The channel's mean pattern-weighted write is a large
constant vector (norm 27.4), but restoring it as a bias recovers only ~2% of the
damage (+0.0235 → +0.0231; low-firing half likewise).

**H3 stands, and now has a name: distributed pattern precision.** With firing-locality
and bias both excluded, the remaining account is the one §5n's energy law already
implied: archetype directions overlap the head's working span (§ tick 191
sub-additivity), every pattern value draws on shared dimensions, and downstream
consumption is isotropic — so removing any channel slightly degrades the precision of
essentially all pattern values, with damage proportional to pattern energy removed
and fungible across directions. The general part is not a signal; it is provisioning
of precision. This is why it resists factorization into nameable variables: it is the
circuit's error budget, not its message content. (In MDL terms: the archetype
dictionaries spend most of their bits on precision shared across all uses, and a
minority on the class-specific hot paths where necessity semantics live.)

## 11. The interaction-order ladder (ticks 223–226): the explicit frontier of the window

Target: layer-1's context signal (the 576 adapter coordinates), explained by fully
explicit objects — no block-0 weight references. Ladder of interaction orders over
embedding codes at window offsets:

| rung | object | bits | val R² | audit ΔCE | share of context gap |
|---|---|---|---|---|---|
| order 1 | token lookups, any resolution (window codes, bigram tables) | up to 52 Mb | ~0.02 | no gain | 0% |
| order 2 | 45 offset-pair bilinear maps, rank 8, controlled training | ~14 Mb | 0.283 | +0.0398 | **29%** |
| order 2+3 | + 10 close-offset trilinear terms | ~8–15 Mb | 0.292 | +0.0397 | 29% |
| (reference) weight-referencing nonlinear generators | ~2,700 Mb effective | 0.63 | +0.032 | 45% |
| (reference) oracle 16-dim interface | — | — | +0.0113 | 100% |

**Verdict: the explicit polynomial ladder saturates at ~29% of the context gap.**
Second-order token interactions are real and cheap (the program's first explicit
context object, U ≈ 0.33); third-order adds ~1% — the remaining computation inside
the 16-token window is not low-order polynomial structure over coarse embedding codes.
Candidate explanations for the un-captured 70%: interactions requiring full-resolution
token identity INSIDE the product terms (codes blur exactly what the subword head
needs); the two nested bilinear forms creating effective order ≥ 4 with strong
cancellation structure (§10's "silence is computed" at the feature level); or
normalization coupling across the window. Ladder paused here for Logan's input per
the standing arrangement — the saturation curve, not another rung, is the deliverable.

### 11b. The named-basis reframe: archetype activations carry the missing blocks (tick 229)

Features = ledger objects only: the current token's 96-dim embedding code + 16
archetype activations per layer-0 head (attention outputs projected onto validated
archetype value directions — "head 3's {the}-weighted content", "head 1's
fragment-weighted content"). No MLP codes, no weight references; new bits = the
fitted map only.

| arm | map bits | val R² | audit ΔCE | share of context gap |
|---|---|---|---|---|
| named linear | ~4.4 Mb | 0.334 | +0.0362 | 38% |
| **named gated bilinear** | **~9.2 Mb** | **0.521** | **+0.0309** | **51%** |

The named gated arm surpasses every weight-referencing generator (best +0.0319 at
~2,700 Mb effective) at three hundred times fewer bits — **U ≈ 0.42, the program's
best row** — and vindicates the reframe's premise: the un-captured computation was
the attention-mediated blocks (emb×attn0, attn0×attn0), inexpressible over raw tokens
but linear-plus-gate expressible over the layer-0 archetype activations the ledger
already named. Ladder position: raw-token polynomial 29% → named basis 51% → oracle
100%. (The entity-restricted fidelity check is still owed; next rungs available:
g×g structured terms, deeper archetype inventories per head, joint named+pairwise.)

### 11c. The fresh-eyes protocol: consensus elicited, falsified, and the real mechanism cornered (tick 230)

Logan's protocol executed: 8 failure clusters under the named-basis model, 8 parallel
cold-reader agents (20 raw samples each, no priors shared). **Seven of eight converged
on one hypothesis** — in-document induction/copy ("the target follows an earlier
verbatim occurrence of the current suffix") — each with concrete thresholds.

**Falsified by ground truth, decisively.** With a positive-controlled locator
(19/19): the induction candidate (longest-suffix-match continuation from earlier in
the document) equals the target in **1% of worst-case positions versus 12% at random
positions** — anti-enriched; and the target token occurs anywhere earlier in the
document at 43% versus 51% baseline — no enrichment. The hard cases are
predominantly FIRST MENTIONS: "Matvich → uk", "Lindsay → L(ohan)", "Bradley →
Manning" LOOK like copies to any knowledgeable reader (which is precisely why seven
independent LLM readers converged on it — shared plausibility priors make convergence
≠ correctness), but the sources are not in the documents.

**The corrected characterization of the missing 49%: parametric entity memory.** The
continuation is retrievable only from pretraining knowledge, keyed by the exact
multi-token local prefix. The block-0 MLP's role in the interface is an ASSOCIATIVE
MEMORY: exact window-suffix → entity-continuation key content. This single account
explains every prior null at once: single-token tables fail (memory keys are
multi-token), low-rank pair codes fail (keys need exact joint identity), the MLP is
mandatory in the sliver (it IS the memory), and it is dense in every basis
(distributed storage is how associative memories look). Natural next rung, now
well-posed: replace the parametric memory with an EXPLICIT one — a mined
prefix→continuation datastore feeding the named-basis map (kNN-style), MDL-priced
like everything else.
