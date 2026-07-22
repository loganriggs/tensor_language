# HANDOFF — query/key minimum-description-length program (written 2026-07-21)

For whoever picks this up next. Read this file, then `LOG.md` (tail ~200 lines) for tick-by-tick
state. Logan (Elriggs, logan.smith.5@gmail.com) runs this program; he prefers **spelled-out prose,
no internal abbreviations** when you report to him.

---

## 1. Current objective (Logan's steer, 2026-07-21)

An explicit **two-stage minimum-description-length decomposition of the embedding as read by the
query/key circuit**:

1. **Stage one — free merge.** Tokens that pay attention to the same things can be treated as the
   same token. This is a *vocabulary* reduction and it is (nearly) free, so it goes first.
2. **Stage two — sparse dictionary.** Decompose the merged rows as a **sparse linear combination of
   k atoms**, using sparse-autoencoder architectures (batch-top-k, matryoshka, and per-token top-k
   with orthogonal-matching-pursuit/least-squares coefficients) to find a good k and dictionary.

Get it working on the **first** query/key circuit (easy mode), then extend to deeper layers, which
are harder because their inputs are contextual rather than the raw embedding.

**Why this is new — verified, not assumed.** Query/key has been compressed by vector-quantization,
rank, rank-then-vector-quantization, used-subspace projection, and vocabulary-merge — but **never by
an overcomplete sparse dictionary, at any layer**. `codebooks.py`'s docstring literally reserves the
slot ("sparse bilinear dictionary — pending"). The merge-then-sparse pipeline with a matched-bits
comparison has never been run.

---

## 2. THE OBJECT DECISION — RESOLVED 2026-07-21: Option A (Logan, in chat)

Logan chose weight-only: avoid data-conditional objects; the first query/key circuit is the layer-0
fold versus the raw embedding, vocabulary-by-vocabulary. **Layer 1 is deferred until the layer-0
program is settled.** When it resumes, the intent is to propagate the embedding, block-0 attention
output, and bilinear-MLP output through the weights (object construction is the research question —
not conditional means). The original decision record is kept below.

The question was: what exactly are "the rows" that stage one merges and stage two sparse-codes?

### Option A — layer-0 exact fold (weight-only, vocabulary-by-vocabulary) — RECOMMENDED

At layer 0 the query/key input **is** the RMS-normed embedding (the block's lambda skip is killed by
the pre-attention norm), so the circuit folds in closed form:

```
q̂(t) = rms_norm_head( rms_norm(e_t) @ W_q^T )          # (V, 9, 128), unit-RMS rows
score(t_q @ i, t_k @ j) = q̂(t_q)^T R_{i-j} k̂(t_k) / head_dim
```

Implemented already: `tier2_folding.branch_factors(m, branch)`. **Fold gate passes at ~1e-15.**

The vocabulary-by-vocabulary matrix **is exactly the product of the two (V × 128) factor tables**
(per rotary frequency band, via the cosine/sine difference expansion in
`tier2_folding.scores_from_factors`). So decomposing the *factors* IS decomposing the
vocab-by-vocab map, losslessly — it is not an approximation of it. It also avoids materializing
50304² ≈ 2.5 billion entries per head-branch (~10 GB), which this program forbids outright.

Rows for the dictionary: `cat([q̂[:, h], k̂[:, h]], 1)` of shape **(V, 256)**, one per head-branch,
**18 head-branches** (9 heads × 2 bilinear branches).

### Option B — layer-1 conditional means (what an earlier draft wrongly assumed)

Layer 1's input is the post-block-0 residual, which is contextual, so it **cannot** be folded from
weights. `l1_condmean_qk.py` estimates per-token conditional-mean factor tables from ~524k tokens.

### Why Option A

| | layer-0 fold | layer-1 conditional means |
|---|---|---|
| Is the object exact? | **yes**, exact function of the token, gate ~1e-15 | no — a data estimate |
| Does data enter the *object*? | **no**, only the evaluation | yes, ~524k tokens |
| Floor before any compression | **0.000** | **+0.014** held-out |
| Description-length claim is about | **the model's weights** | a dataset-specific activation summary |
| "Embedding folded in on both sides"? | **yes, literally** | no |

Logan's caution — "by default we should avoid the data conditional mean stuff, or at least be careful
about it" — is well placed. At layer 1 the per-token table is **already a lossy model of the circuit
before you compress anything**, and its error is comparable to the effects being measured: the
vector-quantization-256 number there is +0.092, only about six times the +0.014 floor. So a real
fraction of every layer-1 compression number is estimation error, not compression cost. At layer 0
that confound does not exist. The bookkeeping is also cleaner: conditional-mean tables are *fitted*,
so the frozen convention forces you to report estimation tokens beside structural bits; at layer 0
the estimation-token count is zero.

**Layer 0 is also easy mode in the compressibility sense**, which supports calling it "easy":
effective alphabet finds 7 of 9 heads have marginal alphabet **1**, and joint vector-quantization at
256 classes costs **+0.008 at 165× description-length reduction**. Compare layer 1, where the same
move costs +0.092. Stage one really is nearly free at layer 0.

**The honest cost of Option A:** layer 0 and layer 1 are different circuits. The qualitative arc
Logan liked (grammatical categories, syntactic dependencies — findings F31/F36) lives at **layer 1**;
layer 0's classes are morphology / word-fragment flavored. Re-anchoring buys methodological
cleanliness and gives up continuity with those features.

**For deeper layers, do not default to conditional means.** Treat "how do you get a
vocabulary-by-vocabulary object when the input is contextual?" as the actual research question.
Candidates: (a) first-order-in-context (live layer-0 pattern × compressed content — this already
worked once, finding FO-1), (b) composed (current token, attended token) pair rows, (c) conditional
means *with* the estimation cost explicitly paid in the ledger.

---

## 3. State of play

| phase | status |
|---|---|
| 0. Planted-structure positive control | **DONE — passes selectivity 2/2**, committed |
| 1. Stage-one free merge | **DONE on layer 0** (`qk_merge_stage1_l0.py/.json`, tick 151); per-head-branch K=2048 merge is free-or-better at 4.2% raw |
| 2. Stage-two sparse dictionary (3 arms) | **DONE** (`qk_sae_dict.py/.json`, tick 152): all dict arms beat matched-bits SVD; two-stage ~free at 1.3% raw |
| 3. Matched-bits frontier | **DONE as part of Phase 2** (SVD r=8..128 + dict budgets + two-stage, one table) |
| 4. Convergence + robustness | **RUNNING** (`qk_sae_robust.py`, tick 152): wide 128-seq audit + seeds, targets the negative-dCE band |
| 5. Deeper-layer query/key | not started |

Nothing is currently running. GPU is idle.

### Phase 0 result (object-independent — carries over to either option)

`qk_sae_control.py` / `qk_sae_control.json`. Two plants with known ground truth, every arm at
**matched description length** (dictionary n=512, k=8 = 5.51 megabits; matched
singular-value-decomposition rank 40 = 5.45 megabits). Fraction of variance unexplained:

| plant | low-rank | token-linear | **token-OMP/LS** | batch-top-k | matryoshka |
|---|---|---|---|---|---|
| sparse (n_true=512, k_true=8) | 0.627 | 0.038 | **0.012** | 0.041 | 0.067 |
| low-rank (r_true=16) | **0.0003** | 4.370 | 0.0040 | 13.544 | 0.076 |

Selectivity 2/2. Atom recovery on the sparse plant: mean max cosine similarity **0.986** — it finds
the planted atoms almost exactly.

**Two pre-registered signals (stated before touching the real model):**
1. Orthogonal-matching-pursuit with least-squares coefficients is the **only** arm robust on both
   plants — it is the strong arm.
2. **Batch-top-k lands at fraction-unexplained above 1 on the low-rank plant** (13.5 — worse than
   predicting the mean), as does the linear-encoder top-k (4.37), because correlated atoms make
   raw-magnitude selection explode without a least-squares refit. This **reproduces the layer-0
   value-dictionary finding on known ground truth** and predicts batch-top-k will lose on the real
   query/key tables too. **Falsifiable: if batch-top-k wins there, the plant model is wrong.**

---

## 4. File map

### New in this program (the two-stage work)
- `basis_aligned/qk_mdl/qk_sae_control.py` / `.json` — Phase 0 planted battery. **Committed, passing.**
- `basis_aligned/qk_mdl/qk_merge_stage1.py` — Phase 1 merge frontier. **Uncommitted, written against
  Option B (layer-1 conditional means).** If Logan picks Option A, swap the estimation pass for
  `tier2_folding.branch_factors` — the merge, bit-accounting, and audit machinery transfer unchanged,
  and the +0.014 floor disappears from every downstream number.
- Still to write: `qk_sae_dict.py` (Phases 2–4), `qk_sae_layer2.py` (Phase 5).

### Model + fold machinery (reuse, do not rewrite)
- `basis_aligned/qk_mdl/tier2_model.py` — `load_elriggs`, `reference_forward` (logit soft-cap
  30·tanh(·/30), embedding RMS-norm, value-bus mixing), `build_eval_tokens` (pile-10k + GPT-2
  tokenizer), `rope_tables`, `apply_rot`.
- `basis_aligned/qk_mdl/tier2_folding.py` — `branch_factors` (**the exact layer-0 fold**),
  `scores_from_factors` (factors → scores, the patch path). Run as `__main__` to re-verify the gate.
- `basis_aligned/qk_mdl/mdl_accounting.py` — **frozen** description-length conventions (32 bits per
  float, log2(n) per discrete choice). No sparse-dictionary helper exists yet; the formula is inlined
  in three places. **Add `dl_sparse_dict` there rather than inlining a fourth copy.**
- `basis_aligned/qk_mdl/codebooks.py` — `fit_svd`, `fit_bicluster`, `fit_toeplitz`,
  `fit_conjunction`. Docstring reserves the sparse-dictionary slot.

### Sparse-dictionary implementations that ALREADY EXIST (do not rewrite any of these)
- `basis_aligned/qk_mdl/ov_sparse.py` — `train_topk_dict`; lines 143-172 are the frozen-support
  cross-entropy-training loop (supports frozen from least-squares, atoms+coefficients trained through
  the frozen model).
- `basis_aligned/e9_batchtopk.py` — `train_batchtopk`, complete batch-top-k with threshold EMA and
  global deployment threshold.
- `basis_aligned/qk_mdl/ov_matryoshka.py` — `train(nested=True)`, `fvu_topk`; nested prefixes.
- `basis_aligned/qk_mdl/ov_omp_batch.py` — `omp`, vectorized greedy orthogonal matching pursuit with
  least-squares refit; also the marginal-error batch allocation.
- `basis_aligned/qk_mdl/ov_dict_variants.py` — `train_dict(mode='token'|'batch')`, `encode_token`,
  `encode_batch`, and **`bits(...)`, the variable-nonzero description-length helper** (needed for
  batch-top-k and batch orthogonal-matching-pursuit).
- `basis_aligned/e7_pareto_dictionary.py`, `e7b_ce_finetune.py` (`DictEmbed`), `e7c_kl_distill.py`.
- Diagnostics on why batch-top-k underperforms: `ov_batch_probe.py`, `ov_train_curves.py`,
  `ov_sametest.py`, `ov_converged_ce.py`, `ov_routed_fair.py`.

### Existing merge / equivalence-class code (stage one)
- `basis_aligned/qk_mdl/effective_alphabet.py` — the sufficient-partition measurement; clusters the
  shared `[q|k]` rows per head-branch, sweeps k, reports weight-side and behavioral alphabets.
- `basis_aligned/qk_mdl/qk_cluster_vs_rank.py`, `qk_rank_then_vq.py` — clusters versus rank, and the
  composed rank-then-cluster code, both with explicit bit ledgers.
- `basis_aligned/qk_mdl/l1_condmean_qk.py` — layer-1 conditional-mean tables + `vq_tables` +
  `audit_ce` (Option B machinery).
- `basis_aligned/tn_gauge/bilin18_qk1_vocab.py` — the layer-1 vocabulary-merge script (finding F30).

### Documentation
- `basis_aligned/qk_mdl/LOG.md` — dated tick log, ~3300 lines. **Tail is current state.**
- `basis_aligned/qk_mdl/qk_mdl_spec.md` — binding spec, especially §6 anti-drift rules.
- `basis_aligned/tn_gauge/SUMMARY.md` — navigable synthesis of findings F1–F39.
- `basis_aligned/tn_gauge/GOALS.md` — per-finding detail.
- `basis_aligned/tn_gauge/folded_basis_features.md` — the qualitative decompositions
  (embedding → syntax, output-value → semantics, composed → dependencies). **Layer 1.**

### Data
- `/workspace/tensor_language/data_fineweb_tokens.npy` — 600 × 513 FineWeb tokens (the model's
  training distribution), GPT-2 tokenizer, from the `sample-10BT` stream.
- Pile via `build_eval_tokens` (NeelNanda/pile-10k).

---

## 5. The model

**bilin18** = `Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd`, 546M parameters.
- Two-branch **unnormalized bilinear** attention: `pattern = (q1·k1)(q2·k2)/head_dim²`, causally
  masked. **No softmax anywhere** — so there is no per-query gauge, and pattern entries can be
  negative. Do not row-center as a gauge fix.
- Bilinear multi-layer perceptrons; 18 layers, width 1152, 9 heads, head_dim 128, vocab 50304, RoPE.
- Per-head query/key RMS-norm **then** RoPE; rotation sign is opposite to the tiny models.
- Value-bus mixing: each layer's value mixes with block-0's, `v = (1−lamb)·v + lamb·v1`.
- Logit soft-cap: `30·tanh(logits/30)`.
- **Evaluation regime frozen at T=512** — the model's competent regime. Cross-entropy degrades badly
  past ~512 because the unnormalized score-product's row mass grows with context.

---

## 6. Program rules that bind (non-negotiable)

1. **Gates before claims.** Every patched or compressed forward must be verified against
   `reference_forward` (~1e-7 for patch harnesses, ~1e-15 for the fold). Report the gate.
2. **Held-out ΔCE is the binding metric.** Structural fraction-of-variance-unexplained is a search
   metric only. Audit tokens disjoint from fitting tokens (established split: `AUDIT = ALL[4:20]`,
   `TRAIN = ALL[20:]`).
3. **Matched bits.** Never compare compression schemes at matched class count or matched rank —
   only at matched description length.
4. **Positive controls before claims.** Every wrong headline in this program's history was caught by
   a known-answer control, never by inspection. Phase 0 exists for this reason.
5. **Convergence is part of verification** — loss plateaus, multiple seeds, seed variance reported.
6. **Negatives are results.** Report them plainly.
7. **Description-length convention:** structural bits and estimation tokens reported side by side.
8. **No V×V at 50k vocabulary.** Work on factors.
9. Generator scripts written as files (never shell here-document string surgery); absolute paths in
   background jobs.
10. Deviations from a pre-registered plan → raise a **QUESTION FOR LOGAN**, do not silently drift.

---

## 7. Corrections on record (do not re-introduce these errors)

- **The "128-dimensional, 9× essentially-free" claim was WRONG.** With enough data the layer-1
  query/key input covariance spans rank-at-99% = **969 of 1152 (~84%) and is still climbing**; the
  earlier ~312 figure was an artifact of estimating on 512 tokens. What *is* true: the spectrum is
  steeply concentrated — **effective rank ≈ 47**, and 90% of the energy sits in **≈334 dimensions**
  (both saturate by ~16k tokens). A 128-dimensional used-subspace projection is therefore **lossy but
  cheap** (+0.003 held-out), **never lossless**. Do not describe it as free.
- **The bilinear input null was understated once and corrected:** per unit,
  `hidden_i = (a_i·x)(b_i·x)` reads only the 2-dimensional span{a_i, b_i}, and the product zeros on a
  *union of two hyperplanes* — a variety, not a linear subspace. But measured, the query/key-relevant
  units collectively span nearly all input dimensions, so there is **no large weight-only linear
  null**; the big reduction is activation-weighted, and the product-carving is exactly why composed
  (joint current × attended) features beat individual ones.
- **Marginals do not compose.** Repeatedly: individually free ablations are jointly expensive
  (7 heads individually ~free, jointly +0.534); per-layer menus that each look cheap compose to
  +1.44 across the model. Always audit the joint arm.

---

## 8. Suggested next actions

**If Logan says layer 0 (Option A):**
1. Re-verify the fold gate (`python basis_aligned/qk_mdl/tier2_folding.py`, expect ~1e-15).
2. Adapt `qk_merge_stage1.py`: replace the conditional-mean estimation pass with
   `branch_factors(m, branch)`; rows become `cat([q̂[:,h], k̂[:,h]], 1)`, 18 head-branches. Keep the
   global-partition-versus-per-head-branch distinction and the bit accounting as written. The
   uncompressed arm now has ΔCE exactly 0 — no floor.
3. Run the merge frontier, then Phase 2 with all three encoder arms at matched bits.

**If Logan says layer 1 (Option B):** `qk_merge_stage1.py` runs as-is; report the +0.014 floor beside
every number and carry estimation tokens in the ledger.

**Either way:** Phase 0 stands and does not need re-running.

---

## 9. Note on commit trailers

The cron tick text asks for a `Co-Authored-By: Claude Fable 5` trailer. Use the trailer matching the
model that actually wrote the commit rather than copying that line blindly — misattributing
authorship is worse than a cosmetic inconsistency. This was flagged to Logan; he has not ruled on it.
