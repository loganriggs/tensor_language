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

### 11d. Explicit memory: exact n-gram form refuted; the parametric form may be the MDL frontier (ticks 231–232)

v1 (3M-token datastore): +0.0004 nats gain. v2 (30M tokens, entity-filtered, 805k
trigram keys): **no gain at all** (+0.03096 vs named-only +0.03087) despite 10× data.
The exact-suffix datastore is the wrong replacement form at any feasible scale we can
mine: either the model's memory keys are fuzzier than exact n-grams, or the entities
driving the hard cases ("Matvichuk") are so rare that only pretraining-scale exposure
contains them — in which case the honest conclusion is notable: **the MLP's parametric
encoding (~680 Mbit) may already be near the MDL frontier for its memory content**,
i.e., no small explicit object exists not because we haven't found it but because the
content is irreducibly a compressed long-tail entity table. Level-4 standing: the
dense mixer's function is NAMED (fuzzy prefix-keyed entity-continuation memory), its
scope MEASURED (16-token keys, 10-dim output law, entity-continuation domain), and
its storage format characterized as plausibly already-efficient. The two remaining
discriminators are heavyweight and Logan-gated: a pretraining-scale datastore
(billions of tokens) or soft-key embedding-space retrieval.

### 11e. Memory-extraction rungs completed: the memory is real but not short-key readable (ticks 231–235)

Four constructive extraction attempts, all against the named-basis anchor (+0.03087):
corpus datastore 3M tokens (+0.0004); corpus datastore 30M entity-filtered (null);
self-distilled table, corpus keys (+0.0003); **self-distilled table with PERFECT key
coverage from the audit texts themselves (diagnostic-only, labeled leakage): +0.03131
— null-to-negative even as an upper bound.** The model's confident three-token
readouts are not its memory: with only short keys the memory does not activate (its
true keys are up-to-16-token fuzzy contexts), and/or its interface contribution is
key-content for matching rather than next-token identity.

Standing account after the full arc: the missing ~49% is parametric entity memory —
supported by the failure taxonomy (first-mention entities), the toy ladder
(exposure-matched capacity ~1.1M pairs; explicit tail-tables cost MLP-scale bits at
feasible mining), and the elimination of induction, exact n-grams (two scales), and
short-key self-distillation. Every cheap constructive extraction is now falsified;
the remaining extraction ideas (16-token fuzzy-key caching, soft retrieval) are
architecturally heavy. The honest level-4 closing statement: the memory's CONTENT is
characterized and its EXTRACTION is measured to require either pretraining-scale
mining or the model's own key-matching machinery — which is the quantified sense in
which the parametric form is the practical description-length frontier.

### 11f. Adversarial patching round: the entity memory is KEY-SIDE mid-stack MLP enrichment, late-attention transported (tick 238)

Four agents (two hypothesis pairs, advocate/falsifier each) authored 15 patch specs
with pre-registered predictions and abandon conditions; all executed in one batch.

**Both advocates abandoned by their own criteria.** H2-advocate (deep attention
re-reads the key and assembles the answer at the target): attention-output
restorations at the target recovered 0.03–0.08 versus its ≥0.70 prediction —
abandoned. H1-advocate (mid-stack attention carries entity retrieval): same specs,
same failure — abandoned. The H2-falsifier's MLP-memory account survived its
concession test (MLP-output restorations beat attention 8–10×: 0.31–0.65 versus
0.03–0.08) but its "distributed additive MLP at the span" version was incomplete.

**The verified account is a third structure the key-side probes exposed:** restoring
the KEY position's residual at layer 8 alone recovers 0.65 (cluster 5); corrupting
the key position's residual only at layers 13–17 destroys 0.99 of the prediction —
and this holds even for the "early-resolved" surface case (cluster 2: 0.99). So:
**entity retrieval is computed at the KEY TOKEN's own position by mid-stack MLPs —
"Lindsay"'s residual is progressively enriched into an entity representation — and
late attention (layers 13+) transports the enriched state to the target.** Target-side
assembly is minor (attention-out restores ≈ 0); target-side MLPs contribute
moderately (0.63 in the handle case). The cross-regime dissociation held in one
direction (corrupting mid-stack attention on the surface-key example: damage −0.02).

Both fresh-eyes rounds now ended the same way: agent consensus or advocacy corrected
by pre-registered ground truth — and each correction bought a sharper mechanism than
any agent proposed. The memory story, final form: fuzzy compositional keys (68%),
bound early; entity enrichment at the key position via mid-stack MLPs; late attention
transport; block-0's slice of this is the layer-1-pattern route we fully decomposed.

### 11g. The enrichment mechanism at population scale (tick 239, n=64 strong-key failures)

The tick-238 case-study mechanism generalizes: (1) **enrichment-completion depth** —
restoring the key position's clean residual at a single layer recovers a median 0.64
at layer 2, 0.88 at layer 8, 0.98 at layer 11, 1.00 at layer 14: the key token's
entity representation is substantially built by layer 8 and complete by 11–14 (means
lag medians: a subpopulation with other routes remains). (2) **Late key-side
necessity** — corrupting the key position's residual only at layers 13–17 destroys a
median 0.99 of the prediction: late attention transport from the key position is the
delivery path for essentially all median cases. (3) **The enrichment is MLP-driven**:
restoring mid-stack (layers 5–11) MLP outputs at the key position recovers a median
0.81, versus 0.24 for attention outputs — a 3.3× ratio at the population level.

Final mechanism statement, causally verified at scale: **fuzzy compositional keys
(68% of failures) are bound early; the key token's own residual is then enriched into
an entity representation by mid-stack MLPs (layers ~5–11); late attention (13+)
transports the enriched state to the prediction site.** Block-0's role — the part
this program decomposed exhaustively — is the first stage of that pipeline plus the
layer-1 pattern route. The two fresh-eyes/adversarial rounds, the patching program,
and this scale-up together turn the "parametric memory" claim from an inference into
a depth-resolved, component-resolved, population-verified circuit account.

### 11h. Inside the enrichment pipeline (tick 240, n=48): front-loaded and context-BOUND

**Which MLPs.** Single-layer MLP-output restoration at the key position: block 1's
MLP is the largest individual contributor (median recovery 0.50), followed by
block 0 (0.30) and blocks 2-3 (0.24, 0.21); every mid-stack layer 4-13 individually
gives only 0.07-0.12 while the 5-11 band jointly gave 0.81 (tick 239). The
enrichment is front-loaded in the first four blocks and then distributed/redundant
across the mid-stack — no single mid-stack MLP is the memory organ; the band works
superadditively (the same redundancy signature layer-1 attention showed).

**What is encoded — the transplant refutation.** If the enriched key-side state
were a context-independent entity vector (dictionary lookup keyed by the token),
transplanting the key position's transport-band residual (layers 13-17) from
another context containing the same key token should restore the prediction. It
does not: synthetic-template donor median 0.04 (indistinguishable from the
neutral-token donor control at -0.01); same-token-in-a-different-real-document
donor -0.05 median, mean -0.33 (actively harmful). The positive control (the
position's own clean residual, same machinery) restores median 1.00. Caveat: donor
positions differ from target positions, but values and residuals carry no rotary
encoding, and real donors at many positions fail identically.

**Conclusion:** the enriched key-position state is CONTEXT-BOUND — mid-stack
enrichment integrates surrounding context into the key token's residual rather
than retrieving a token-keyed dictionary entry. This is the mechanism-level match
to the key taxonomy (68% of failures need a median of 3 heavy tokens): the
"memory key" is a compound assembled at the key position, and the parametric
memory is addressed by that compound, not by the token identity alone.

### 11i. Where the compound key assembles: at the query, by layer 8 (tick 241, n=39)

For failures with two strong context keys (both >0.5 nat by neutral substitution,
offsets >= 1), corrupt only the secondary key and restore clean residuals at single
depths. Two curves: (a) restoring the PRIMARY key position's residual rescues
nothing at any depth (medians 0.00-0.05 from layer 2 through the full 13-17 band) —
the secondary key's content never routes through the primary key's position;
(b) restoring the QUERY position's residual rescues 0.03 at layer 2, 0.56 at
layer 5, 0.90 at layer 8, 0.99 at layer 14 — the compound key is assembled AT THE
QUERY POSITION, substantially complete by layer 8.

Reconciliation of the full picture: each strong key token's residual is enriched
in place, conditioned on its local context (tick 240 transplant failure) but not on
the other keys (curve a); the query position aggregates the separate key
contributions through the mid-stack, finishing by ~layer 8; the primary entity
token's content additionally stays key-side and is fetched late (tick 239's
0.99 late-band necessity at the primary key). The "compound key" that addresses
parametric memory is a query-side aggregate of independently-enriched per-token
routes — not a bound structure at any key position.

### 11j. Query-side aggregation is distributed transport (tick 242, n=39)

Site resolution of the query-side integration (same secondary-key-corruption
regime as 11i): restoring attention outputs at the query position over layers 0-8
recovers a median 0.90; MLP outputs over the same band recover 0.80 — both high
because they sit serially on one path (attention imports the key routes; query MLPs
transform the aggregate). The per-layer attention profile is broad and shallow —
individual layers give 0.02-0.26 (peak at layers 4-5) against 0.90 for the band —
so the aggregation is distributed across layers roughly 3-8 with no single
collector hop, the third appearance of the redundancy/superadditivity signature
(after layer-1 attention and the mid-stack enrichment band).

**The verified memory pipeline, complete (ticks 236-242):** (1) each strong key
token's residual is enriched IN PLACE by front-loaded MLPs (block 1 largest single
share), conditioned on local context but independent of the other keys; (2) the
query position aggregates the per-token routes through distributed attention over
layers ~3-8, interleaved with query-side MLP processing, complete by layer 8;
(3) the primary entity token's content additionally remains key-side and is
fetched by late attention (layers 13+, necessity 0.99); (4) capped readout at the
target. Every stage is causally verified at population scale with positive and
negative controls; two adversarial single-mechanism advocates were eliminated by
pre-registered criteria on the way.

### 11k. The pipeline is the general context mechanism (tick 243, random positions, n=64)

The identical tick-239 battery on strong-key positions sampled uniformly at random
(any position where a context token in the last 16 costs >1 nat when substituted —
which is 95% of random positions, 152 of 160: strong-key context dependence is
ubiquitous). Results match and sharpen the failure-set numbers: single-layer
key-residual restore median 0.71 at layer 2, 0.98 at layer 8, 1.00 at layer 14;
late key-side corruption damage median 0.98; MLP-versus-attention at the key
position 0.94 versus 0.39. Means run higher than on the failure set (0.86-0.89
versus 0.69-0.71) because worst failures over-sample hard multi-route cases.

**Conclusion of the arc:** the four-stage pipeline is not a tail pathology — it is
the model's general mechanism of context use. The worst-failure set differs from
typical positions only in being where the named basis cannot reproduce the
pipeline's output, not in mechanism.

### 11l. The enrichment signal has NO shared low-dimensional subspace (tick 244, n=48 held-out)

Causal dimensionality of the enrichment delta (clean-minus-corrupt key-position
residual at layer-8 entry, the vector that alone restores median 0.98): principal
components fit on 80 deltas, causal recovery of rank-r projections on 48 held-out
positions. Result: recovery climbs roughly linearly with rank and never knees —
0.00 at rank 1, 0.13 at rank 16, 0.29 at the FULL 80-dimensional fit-set span —
versus 0.98 for each position's own unprojected delta and 0.00 for a random
16-dimensional control. The shared variance is causally inert: rank 16 carries 54%
of fit-set variance but only 0.13 of recovery, and even the complete span of 80
other instances' deltas recovers less than a third.

**Reading:** enrichment writes are position- and entity-specific — each instance
occupies its own nearly-orthogonal direction in residual space, the signature of a
hash-like associative store rather than a structured feature code. This closes the
level-4 sparsification question for pipeline stage 1 with a controlled negative:
the MLP's interface to layer-1 readers is narrow (10 effective dimensions per
reader), but its memory-content channel is irreducibly high-dimensional — which is
precisely what "parametric memory near the description-length frontier" predicts:
content that cannot be compressed into shared directions is content no small
explicit object can substitute for.

### 11m. The single-fact eraser (tick 245, n=48): erasure is surgical

Application of the pipeline + orthogonality findings: subtract a fact's own
enrichment delta at its key position (layer-8 entry) in the clean run. Results:
the targeted prediction drops a median 2.95 nats (mean 3.36), while the mean
absolute disturbance to every OTHER prediction in the same document is 0.006
nats — a selectivity ratio near five hundred. A norm-matched random vector at the
same site drops the target only 0.04 median (direction, not perturbation size,
carries the effect). Cross-fact interference: where a document contains a second
measured fact, erasing the first moves the second by a median 0.000 nats (61
pairs). Facts are individually addressable and individually removable at
inference time, with no weight modification — the causal cash-out of the
per-instance orthogonality of enrichment writes (11l).

### 11n. Retrieval is position-addressed (tick 246, n=37): relocation fails completely

Erase the fact at its key position and inject the same delta elsewhere: recovery is
zero everywhere — median 0.008 one token away, -0.002 four tokens away, -0.001 at
sixteen tokens before the query. Injecting a duplicate of the delta at a second
position without erasing the original changes the prediction by -0.000: content
placed anywhere but the key slot is simply never read. Combined with the 0.98
recovery when the same delta is re-injected at the key slot itself (ticks 239/243),
the late fetch is POSITION-ADDRESSED: the query-side compound key computes WHERE to
look from the surface token arrangement, and the enriched vector is a payload
readable only at that slot. The division of labor in the pipeline is therefore:
query aggregate = addressing; key-side enrichment = payload at a fixed address.
Practical consequence: inference-time fact injection must target the slot the
query will address (where it works at 0.98); the eraser (11m) is unaffected.

## 12. LAYER 2 (opened by Logan 2026-07-26)

### 12a. Layer-2 port (tick 247): token identity still carries 93%

Token-conditional mean-residual tables at block-2's input (62% of types seen,
94% of audit token mass), through block 2's own projections: patching layer-2's
pattern with table scores costs +0.0278 nats versus +0.3904 for zeroing the layer
— a 92.9% token-identity share. The depth-decay curve so far: layer 0 exact
(100%), layer 1 99.0%, layer 2 92.9%. Two structural notes: (1) layer-2 attention
carries 7x less total causal load than layer-1 (+0.39 vs +2.70 zeroed) — the
pattern machinery is front-loaded; (2) the ABSOLUTE context residual is nearly
constant per layer (+0.027 at layer 1, +0.028 at layer 2) even as the token share
falls — consistent with a fixed-bandwidth context channel per layer rather than
exponential growth. Per-head zeroing: head 5 dominant (+0.034), heads 2/3/6
moderate (+0.010-0.012), heads 1/4 negligible; per-head damages sum to 0.083
versus 0.390 jointly — the superadditive redundancy signature, fourth appearance.

### 12b. Block-1 MLP output: flat spectrum, steep causal frontier (tick 248 part A)

PCA of block-1's MLP output (disjoint corpus): a notably flat spectrum — 64
dimensions hold 51% of variance, 256 hold 76%, 512 hold 90%. The causal
rank-truncation frontier (project the output onto its top-r subspace inside the
full forward pass, 307k audit): rank 256 costs +0.025; rank 64 +0.319; rank 16
+1.266; rank 4 +1.944 — WORSE than removing the MLP entirely (+1.553): a
hard-truncated write is more damaging than silence, so the tail directions are
load-bearing, not noise. Contrast with block-0's per-reader ten-dimensional
windows: block-1's output needs ~256 dimensions to carry its function. Part B
(strong-key versus weak-key selectivity of truncation) rerunning after an
out-of-memory in the first attempt.

### 12c. Part B verdict: no selectivity at the strong-key grain (design-limited)

Rank-64 truncation hurts strong-key positions (mean 0.31 nats) the same as the
corpus average (0.28); rank-256 likewise (0.06 versus 0.02, medians near zero).
Honest caveat: since 95% of positions carry a strong key, the "weak-key" contrast
group had n=5 — the comparison cannot discriminate at this grain. The sharper
test (next tick): truncate block-1's MLP output AT THE KEY POSITION ONLY, on the
failure-packet positions where payload demonstrably carries the prediction,
against truncation at a random other position as the control.

### 12d. Key-local truncation (tick 249, n=64): single writes need the tail

At failure-packet key positions, truncating block-1's MLP output AT THAT POSITION
ONLY: rank 64 of the shared PCA basis loses most of the write's value (median
drop 0.135 versus 0.19 for zeroing the write entirely — only ~30% retained);
rank 256 retains ~90% (drop 0.022). Identical truncation at random non-key
positions in the same documents costs nothing (medians 0.002-0.004) — the effect
is position-specific. Verdict: the flat global spectrum is NOT mere aggregate
diversity across positions — each individual memory write already occupies the
64-256 band of the shared basis. The payload lives in the tail at per-write
grain, completing the component-level account: block-1's MLP is a high-rank
writer whose content resists both shared-subspace compression (11l) and local
truncation (here). Note the modest absolute necessity of the single write
(median 0.19): enrichment is redundant across blocks 0-3, consistent with 11h.

### 12e. THE SYMBOL RECURSION WORKS (tick 250): 384 named numbers beat the token table

The compositional fold test: 384 named codes per position (96 embedding
coordinates + 144 layer-0 archetype activations + 144 layer-1 head activations),
ridge linear decoder to block-2's attention input, layer-2 pattern computed from
the SYNTHETIC residual, full 307k audit. Result: **+0.0176 nats — beating the
50304-row token tables (+0.0278)** and reaching 95.5% of layer-2's pattern
function (zero gate +0.390). The layer-1 symbols carry contextual pattern
information no token-identity object can.

The deeper structural fact: the decoder captures only 41% of the residual's
variance, yet the pattern is nearly perfect. The missing 59% — which contains the
high-rank payload traffic (12b-12d) — is PATTERN-IRRELEVANT: layer-2's QK reads
the symbol channel and ignores the payload channel. The symbol/payload type split
is thus not just a bookkeeping convention; it is realized in the model's own
wiring, with attention-pattern formation subscribing to symbols and late-layer
retrieval subscribing to payloads. The dictionary-growth scaling plan is live:
the fold recurses over named symbols with S-squared cost, not exponential context.

### 12f. Layer-3 port (tick 251): token share falls to 76% — symbols now load-bearing

Layer-3 tables +0.0390 versus zeroed +0.1646: token-identity share 76.3%. The
depth-decay curve: 100 / 99.0 / 92.9 / 76.3 — decay accelerating, so this is the
depth where compositional symbols stop being a refinement and become the story.
Total layer load keeps shrinking (2.70 -> 0.39 -> 0.16 zeroed), and per-head
damage is diffuse (max +0.013). Next: the symbol-generated pattern at layer 3 —
first with the CURRENT 384-symbol dictionary (does it already carry layer 3?),
then with layer-2 symbols appended (does the dictionary need to grow per layer?).

### 12g. Dictionary sufficiency at layer 3 (tick 252): symbols are REUSED across depth

The unchanged 384-symbol dictionary (nothing from layer 2) generates layer-3's
pattern at +0.0265 — again beating the token tables (+0.0390; zero +0.1646;
83.9% of function versus the tables' 76.3%). Decoder held-out R-squared falls to
0.35 (from 0.41 at layer 2) yet the pattern stays strong — the pattern-relevant
channel remains inside the symbol span one layer deeper. Two consequences: the
dictionary is not per-layer vocabulary but a shared, reused code (additive-growth
hypothesis, strongest form); and the +0.0265 residual is the budget layer-2
symbols can compete for (tick 253: append 144 layer-2 head activations, re-test).

### 12h. Dictionary growth rate (tick 253): layer-2 symbols close 29% of the residual

Appending 144 layer-2 per-head activations (528 symbols total) improves the
layer-3 fold from +0.0265 to +0.0188 (tables +0.0390, zero +0.1646) — 88.6% of
the layer's pattern function, with decoder R-squared rising 0.35 to 0.43. The
growth datum: a 37.5% larger dictionary buys back 29% of the remaining residual.
Reading: the shared code carries most of the load at every depth (reuse
dominates), while each layer contributes a real but diminishing vocabulary
increment — additive growth with shrinking increments, the friendliest possible
scaling shape: the dictionary converges rather than compounds.

### 12i. Layer-4 port (tick 254): the decay curve is NOT monotone

Layer-4 tables +0.0474 versus zeroed +0.3477: token-identity share 86.4% — UP
from layer 3's 76.3%, and the layer's total load (+0.348) is twice layer 3's
(+0.165). The five-point curve: 100 / 99.0 / 92.9 / 76.3 / 86.4. Reading: layers
are functionally specialized rather than uniformly deepening — layer 3 is the
most context-hungry pattern machine of the first five, layer 4 swings back
toward token-driven patterns with more overall weight. Depth is not a gradient;
it alternates. Heads 0/1/5 lead (+0.013-0.015), moderately diffuse.

### 12j. Layer-4 fold (tick 255): the dictionary covers both regimes

The 528-symbol dictionary generates layer-4's pattern at +0.0263 versus tables
+0.0474 and zero +0.3477 — 92.4% of function at the token-heaviest, highest-load
layer since 1. Three layers, one dictionary, tables beaten everywhere: 95.5%
(layer 2), 88.6% (layer 3), 92.4% (layer 4). The symbol account covers both the
contextual and the token-driven pattern regimes; the recursion is stable.

### 12k. Layer-5 port (tick 256): a hub layer with a single giant head

Layer-5 zeroed costs +2.303 — nearly layer-1 scale (+2.70), an order above
layers 2-4 — and HEAD 7 ALONE carries +0.956, with every other head at or below
+0.011. Token tables still capture 91.9% (+0.187 table damage). The load curve
by layer (zeroed): 2.70 / 0.39 / 0.16 / 0.35 / 2.30 — bimodal, with layer 5 a
second hub. Note the causal tie-in: the query-side aggregation band (11j) peaked
at layers 4-5; layer-5 head 7 is the prime candidate for the aggregation
workhorse, now visible as a single component. Sum of per-head damages 1.01
versus 2.30 joint — still superadditive, but far more concentrated than any
layer measured so far.

### 12l. Layer-5 fold (tick 257): 97.2% of the hub layer from the same dictionary

The 528-symbol dictionary generates layer-5's pattern at +0.0649 versus tables
+0.1869 and zero +2.3030 — 97.2% of the model's second-largest pattern function,
the widest symbols-over-tables margin yet (2.9x), from a decoder holding only
34% of residual variance. Four consecutive layers, one dictionary, tables beaten
at every depth: 95.5 / 88.6 / 92.4 / 97.2 percent of function at layers 2-5.
The compositional recursion holds at hub scale.

### 12m. Reviewer-2 control (tick 258): the random-projection null FAILS — structure is real

Same fold pipeline with every basis replaced by random orthonormal projections of
identical dimensions (96 + 144 + 144): +0.0318 — worse than the token tables
(+0.0278) and 1.8x the structured dictionary's residual (+0.0176). In residual
terms the archetype/principal structure explains half again what any same-width
random summary explains (leaves 4.5% of function versus 8.1% random, 7.1%
tables). The headline survives its strongest deflation: the fold result is a
named-structure claim, not a bandwidth claim. (Decoder R-squared falls only 0.41
to 0.32 while the pattern degrades sharply — further evidence the structured
bases specifically span the pattern-relevant channel.)

### 12n. Bootstrap intervals (tick 259): all layer-2 fold claims survive at 95%

Per-document cross-entropy, four conditions in one pass, 10,000-resample
percentile bootstrap over 150 audit documents: structured fold +0.01764
[+0.01671, +0.01859]; tables +0.02778 [+0.02651, +0.02906]; random null
+0.03178 [+0.03048, +0.03312]. Pairwise: tables minus structured +0.01014
[+0.00904, +0.01125]; random minus structured +0.01414 [+0.01326, +0.01502] —
both decisively positive. Point estimates reproduce the original runs to the
fifth decimal (independent basis draws for the null). Reviewer-2 items 1-2 done.

### 12o. Head-7 identity test (tick 260): NOT compound-selective — a general workhorse

Ablating layer-5 head 7 on failure packets: compound (two-key) positions drop a
mean 2.24 nats, single-key positions 1.82 — a 1.2x ratio, far from the
selectivity the aggregator hypothesis predicted (whole-layer ablation shows the
same mild tilt, 4.43 versus 3.89). Verdict: head 7 is a general heavy-lifter on
context-dependent predictions, not specifically the compound-key assembler; the
aggregation stage remains distributed even though the layer's importance is
concentrated. Secondary observation: on the failure set, layer-5 pattern
ablation costs ~4.3 nats — nearly twice its corpus-average load (2.30) — the
memory-heavy positions lean disproportionately on layer 5.

### 12p. Neutral-token robustness (tick 261): existence robust, identity partially not

Rerunning key identification with three neutral tokens (" one", " thing", " and")
on the 96 failure-packet positions: strong-key EXISTENCE is robust — 91/96
positions have a >1-nat key under all three neutrals (93-94 under each). But the
TOP key's identity agrees across all three in only 47% of positions: compound
positions carry several near-equal keys, and different substitutions reorder
them. Implication, honestly bounded: every patching experiment validated its own
key with an in-run >1-nat denominator, so the recovery/damage fractions stand;
but "the primary key" should read "a key among several near-equals" for roughly
half the positions, and single-key orderings (as in the tick-241 pair selection)
inherit that fuzziness. The paper should use consensus keys or report
per-neutral stability alongside any key-identity claim.

### 12q. MAJOR CORRECTION (tick 262): zero-ablation inflated every layer load 10-60x

Mean-ablation (replace each layer's pattern with its batch-mean at the same
positions — full positional structure kept, all content dependence removed):
layer 2 +0.0265, layer 3 +0.0225, layer 4 +0.0245, layer 5 +0.0363 — versus
zero-ablation's +0.390 / +0.165 / +0.348 / +2.303. Ninety to ninety-eight
percent of the "layer load" was the off-distribution artifact of deleting the
average pattern; the CONTENT-dependent pattern function of layers 2-5 is only
0.02-0.04 nats each. Consequences, stated bluntly:

- The "hub layer 5" story (12k) deflates: its 2.30-nat load is positional
  structure, not content computation; head 7's +0.956 is likewise zero-ablation.
- Every "share of function" percentage in 12e-12l was computed against inflated
  zero gates. Honest re-baseline against the positional mean: at layer 2 the
  symbol fold (+0.0176) beats the positional mean (+0.0265) — it captures ~34%
  of true content function, and the tables capture ~none beyond position; at
  layer 3 symbols also win (+0.0188 vs +0.0225); at layer 4 they roughly tie
  (+0.0263 vs +0.0245); at layer 5 they LOSE (+0.065 vs +0.036) — the "97.2% of
  the hub" claim (12l) is retracted: at layer 5 a position-only mean pattern
  outperforms the symbol fold.
- What survives untouched: the symbols-versus-tables-versus-random comparisons
  (same positional machinery inside all three, bootstrap-certified), the
  layer-2/3 content wins, and the entire memory-pipeline battery (which never
  used zero-ablation). What is retracted: hub rhetoric and all percentages
  quoted against zero gates.

The program's own history repeats: this is the "measure is the message" lesson,
fourth occurrence, caught — as always — by a control, not by inspection.

## 13. MEDICAL MODEL EXTRACTION (Logan redirect 2026-07-27)

### 13a. Foldable bilinear ViT on PathMNIST + first probe (med phase 1-2)

Trained a foldable no-softmax bilinear-attention ViT (0.348M params, D=96, 6 heads,
3 layers, 7x7 patches -> 16 tokens) on PathMNIST (9-class colorectal histology):
val 96.7%, test 85.7% (institutional shift caps test; above AutoKeras 83.4%).

Probe (mean-ablation, per tick-262 discipline):
- Layer importance rises with depth: attention pattern content worth -0.018,
  -0.021, -0.092 test-accuracy at layers 0/1/2. Late attention does the
  classification work; early attention is nearly content-free (mostly positional).
- Heads are sparse: of 18, only layer-2 heads 1 and 2 carry real load (-0.045,
  -0.031); layer-0 head 0 the only early contributor (-0.012). ~3 heads run the
  model. This is a far sparser head inventory than bilin18 and an easy extraction
  target.
- The exact layer-0 patch-code fold reproduces the model's scores to 0.0e0 (exact,
  as designed): layer-0 q/k are a closed form in raw patch pixels, no vocabulary.

Reading: the classifier is a shallow-attention + late-attention machine with ~3
load-bearing heads. Extraction plan: fold layer 0 exactly (done), CP-decompose the
layer-2 head-1/head-2 score tensors into visual archetypes, and test an explicit
pipeline against the 85.7% target.

### 13b. Minimal circuit (med phase 3): attention prunes hard, MLPs carry the model

Mean-ablation pruning on the 85.7% model:
- Attention heads prune drastically. Six of eighteen heads reproduce the full
  accuracy exactly (0.857); the top-4 reach 0.844; top-3 (l0h0, l2h1, l2h2) hold
  0.830 (3 points under full); a single head (l2h1) already gives 0.769. Twelve
  heads are dead weight — a 3x head reduction at near-parity.
- The bilinear MLPs, not attention, are the model. Mean-ablating MLP-0 costs
  0.742 accuracy (collapse to near-chance for 9 classes = 0.11 floor, so this is
  most of the model); MLP-1 costs 0.254; MLP-2 costs 0.000 (entirely prunable).
- Minimal circuit so far: 3 heads + MLP-0 + MLP-1 (drop MLP-2 and 12 heads),
  0.830 test — a substantial compression at -2.7 points.

Reading (mirrors bilin18): attention is a light, sparse router; the bilinear MLPs
do the classification work, front-loaded in the first block. The extraction target
sharpens accordingly: the algorithm to render explicit is MLP-0's bilinear map on
the patch embeddings, gated by ~3 attention heads. Next: fold layer-0 head 0 and
the layer-2 heads to visual archetypes, and probe MLP-0's structure (its input is
the exact patch-pixel embedding, so it is directly analyzable).

### 13c. MLP-0 dissection (med phase 4): partly patch-local, distributed feature bank

- Patch-local test: removing layer-0 attention entirely holds 0.781 (from 0.857) —
  MLP-0 is largely a per-patch pixel-embedding feature extractor, with ~7.6 points
  coming from attention's cross-patch mixing. The dominant computation is local
  texture, not spatial arrangement.
- MLP-0 inner units are a DISTRIBUTED bank, not a sparse code: top-128 of 192 units
  reach parity (0.857), top-64 give 0.613, top-32 only 0.441. Unlike the heads
  (18->6), the bilinear feature bank barely compresses — colorectal tissue texture
  needs many filters. Honest efficiency verdict: heads prune 3x, MLP-0 units ~1.5x.
- Each unit's preferred pixel pattern is an exact 7x7x3 visual filter (top
  eigenvector of its symmetric pixel-space form); top-32 saved for the artifact.

Extraction status: the algorithm is localized (front-block bilinear texture bank on
near-patch-local pixel embeddings + light attention pooling + 3 gating heads) and
made visual (renderable filters), but is a genuine ~128-feature bank, not a handful
of rules — the honest analogue of bilin18's "dense engine, and here a moderately
wide one."

### 13d. The standalone explicit pipeline (med phase 5): 71.6%, accuracy decomposed

A fully explicit, auditable classifier — patch -> frozen extracted quadratic texture
filters (a_j.p)(b_j.p) -> mean-pool -> linear head, no attention/residual/rms_norm,
only the head fit — reaches 0.716 test with all 192 filters (0.711 at 128, 0.701 at
64, 0.660 at 32). This decomposes the model's accuracy into interpretable rungs:

| stage | test acc | adds |
|---|---|---|
| pooled raw pixels + linear (color) | 0.612 | floor |
| + extracted patch-local quadratic texture filters | 0.716 | +0.104 |
| + deep bilinear composition (2-3 MLP blocks, rms_norm) | 0.781 | +0.065 |
| + attention cross-patch pooling | 0.857 | +0.076 |

Honest verdict on the extraction: the explicit patch-local quadratic surrogate
recovers 84% of the full model's accuracy and 42% of its above-color gap, as a
transparent model (192 renderable filters + pool + linear). The remaining ~14
points are genuinely non-patch-local / deep: bilinear composition across blocks and
attention pooling each carry a comparable slice. So colorectal tissue classification
is roughly half explicit local texture and half distributed composition — a
quantified, auditable answer to "what algorithm," not a full lossless reduction.

### 13e. Filter-to-class attribution (med phase 6): a labeled texture dictionary

Fitting the explicit head on just the top-32 rendered filters gives 0.688 test
(nine classes) and a readable filter->tissue map:
- Unit 146 is the dominant detector for normal mucosa (+5.19), mucus (+4.99), and
  adipose (+3.48) — a bright/low-texture "empty-or-pale-tissue" filter, sensibly
  shared across the three least-cellular classes.
- Unit 87 is the lymphocyte detector (+4.53), unit 33 its partner (+3.57) — a
  fine dark-speckle texture, matching lymphocytes' dense small nuclei.
- Unit 82 is the adenocarcinoma detector (+4.54) and also fires for normal mucosa
  (+3.09) — an epithelial-gland texture; the cancer-vs-normal distinction rests on
  the composition rungs (13d), not this single filter, explaining why local texture
  alone plateaus.
- Unit 54 is the background filter (+3.84), unit 128 a smooth-muscle/stroma
  filter recurring across the fibrous classes.

The extracted algorithm is now a labeled dictionary: renderable pixel filters, each
tied to the tissue types it signals, with the clinically hard call (carcinoma vs
normal epithelium) explicitly attributed to composition rather than texture — an
auditable, honest account of what the classifier keys on.
