# QK computational-MDL decomposition (embedding folded in) + bilinear-attention version

**Purpose:** measure and exploit the sparse computational structure of attention QK
circuits by decomposing the VOCAB-SPACE score matrix (embedding folded in) under a
menu of structured "codebooks," scored by description length at matched
PATTERN-level distortion. Core claims to test: (1) most heads have structure
(biclusters, positional/frequency sparsity, conjunctions) that truncated SVD cannot
see, so computational MDL ≪ spectral MDL; (2) for bilinear attention heads, the
two branches factor the computation into interpretable conjuncts, recoverable from
weights alone and checkable against our causally verified induction circuit.

Companion docs: mechanism_decomposition_spec.md (method program + its post-mortem
lessons), vocab_decomposition_project.md (token trees, orderings, gotchas G1-G12,
which apply here too).

---

## 0. Architecture assumptions — VERIFY BEFORE ANYTHING ELSE

Our tiny models (the attn2/block* OWT checkpoints) use **RMSNorm and RoPE**, and
the attention heads are **bilinear** (two QK branches combined by Hadamard
product). Claude Code must verify all three in the actual code before relying on
the algebra below, and specifically must determine:

- **A1:** RMSNorm placement (pre-norm assumed) and whether norm has learnable
  scale γ (fold γ into E's columns... into the effective embedding, see §1).
- **A2:** RoPE details: which dims are rotated (all? half?), the frequency set
  {ω_f}, and whether queries and keys are both rotated (standard) or one-sided.
- **A3 (critical, changes the math):** where the two bilinear branches multiply:
  (a) PRE-softmax: score = s₁ ⊙ s₂, then one softmax; or
  (b) POST-softmax: A = A₁ ⊙ A₂ (elementwise product of two attention patterns),
      possibly renormalized.
  Both cases are handled below (§3), but they lead to different objects. Do not
  guess; read the module.

Report the verified answers at the top of the results log before Tier 0.

---

## 1. The folded object, and when it is exact

### 1.1 Standard (single-branch) head, layer 0

Layer-0 inputs are pure token embeddings. With pre-RMSNorm, the query/key inputs
are ê_t = γ ⊙ e_t / rms(e_t): a deterministic per-token map. Define the
**effective embedding** Ê (V × d) with rows ê_t, and the folded matrix

    M = Ê W_Q^T W_K Ê^T          (V × V; row = query token, col = key token)

With RoPE, the score also depends on relative offset Δ = j_key − i_query, and it
admits an EXACT finite expansion: R_Δ is block-diagonal 2×2 rotations, so

    score(t_q at i, t_k at j) = Σ_f [ cos(ω_f Δ) · C_f[t_q, t_k]
                                     + sin(ω_f Δ) · S_f[t_q, t_k] ]

where, per frequency f, C_f and S_f are V×V matrices of rank ≤ 2·(dims in band f)
built from the corresponding 2-dim slice of W_Q^T R W_K sandwiched by Ê. This is
not an approximation: for layer 0 under RMSNorm + RoPE, the family {C_f, S_f} IS
the head's entire QK computation. Two immediate structure axes fall out:
**frequency sparsity** (which f carry mass; prior work found heads use few bands,
high-f ≈ positional, low-f ≈ semantic) and **token structure within each band**
(the codebooks of §2 applied per C_f/S_f, or to the Δ-averaged / Δ=−1 slices).

**Tier-0 gate:** reconstruct the model's actual layer-0 attention scores from
{C_f, S_f} to machine precision (fp64) on random inputs. If this fails, the
folding algebra or A1-A3 assumptions are wrong; stop.

### 1.2 Gauge

Softmax is invariant to adding any per-query constant: M ~ M + v·1^T. ALWAYS
row-center folded matrices before decomposition/MDL accounting (analogue of the
W_U mean-row gauge). Also invariant: global positive scale (temperature) at the
level of top-k patterns; the distortion metric (§4) absorbs this.

### 1.3 Beyond layer 0 (path-folded matrices)

Layer-1 inputs are embedding + layer-0 head outputs, so exactness is lost. Define
PATH-folded matrices instead: e.g. for the induction head L1H2 with key
information arriving via L0H3's OV,

    M_path = Ê W_Q^T R_Δ W_K (W_OV^{L0H3} Ê)^T    (query reads embedding directly;
                                                    key reads L0H3-transported embedding)

This is exact along that path and ignores other paths + the second norm (state the
approximation when reporting). These are the framework's virtual weights, made
vocab-space. Tier-2 material; layer 0 first.

---

## 2. The codebook menu (each = a structural prior = a way to compress)

All applied to row-centered matrices; DL accounting in §4. Implement in this
order:

1. **Truncated SVD** — the baseline every other codebook must beat at matched
   distortion. DL = r(2V̂ + 1) numbers (V̂ = analyzed vocab subset size).
2. **MDL biclustering** (cross-associations, Chakrabarti et al. 2004): partition
   rows and columns SEPARATELY (from-role ≠ to-role) into blocks; DL = block
   means + partition + exceptions. This is the "months→numbers" codebook. Use the
   MDL-native algorithm (auto-selects #clusters), not k-means-and-pray.
3. **Two-sided tree / HODLR:** put the token tree from the vocab project on both
   axes; off-diagonal blocks constrained low-rank. Reuse the ordering machinery
   and its shuffled controls (G1). Tests whether ONE token tree serves E and the
   heads (the shared-registry claim).
4. **Sparse bilinear dictionary:** score ≈ Σ_m σ_m (a_m·q̂)(b_m·k̂) with (a_m, b_m)
   from a dictionary shared ACROSS heads; per-head DL = #active terms. Rank ≤
   d_head bounds terms per head; cross-head sharing is where corpus-level MDL
   wins. (This is the masked-projector machinery, scalar-output; reuse solver +
   its hard-won fixes: pinv/rowspace M-step, detach W, no_grad, gauge ‖a‖=‖b‖=1.)
5. **Positional codebooks:** frequency-sparsity of {C_f, S_f} (a head that uses 2
   bands = 2 small blocks); near-Toeplitz / shift structure in Δ-dependence where
   present (describe by a few Fourier/displacement coefficients — full-rank but
   tiny DL; SVD-invisible).
6. **Sym/antisym split** (Saponati-style) as a cheap PRE-CLASSIFIER, not a
   codebook: high-symmetric heads → try 2/3/4 first; high-antisymmetric → try 5.
7. **CUR / exemplar tokens** — readability layer only; report exemplars for
   whatever codebook wins, don't score it as compression.

## 3. Bilinear attention heads (the two-branch version)

Per branch b ∈ {1,2}, build M_b (or {C_f^b, S_f^b}) exactly as §1. Then:

**Case A3(a), pre-softmax product:** effective score matrix is M₁ ⊙ M₂.
- Rank accounting: rank(M₁⊙M₂) ≤ r₁r₂ (Khatri-Rao expansion: terms
  (u_a⊙w_b)(v_a⊙z_b)^T). SVD of the product sees up to d_head² undifferentiated
  directions; the natural codebook is a **sparse core over branch-factor pairs**:
  which (a,b) pairs carry mass. Decompose branches separately first, then fit the
  core.
- Semantics: elementwise product = soft AND. Block structure of the product =
  intersection of the branches' block structures; DL(conjunction) can be
  k₁+k₂ where a flat codebook on the product needs k₁·k₂.
- Identifiability bonus: the factorization into conjuncts is generically
  identifiable (the order-jump that makes CP unique), so per-branch structure is
  meaningful, not gauge.

**Case A3(b), post-softmax product:** log A = log A₁ + log A₂ + const(query), so
in LOG-pattern space the branches ADD: decompose each branch's pattern-logit
matrix separately and the head's DL is the SUM of two branch DLs. Cleaner for
MDL; conjunction reading unchanged (product of probabilities).

**The pre-registered ground-truth test (Tier 1 gate):** for the causally verified
induction head L1H2 (both K branches), hypothesis:

    induction ≈ (token-identity match: near-diagonal/identity-like block structure
                 in ONE branch's path-folded matrix, key side through L0H3)
              ∧ (positional/prev-token structure: high-frequency RoPE bands /
                 shift-like Δ-dependence in the OTHER branch)

Success = the two branches' best codebooks are respectively identity-plus-noise
and positional, AND ablating the identified structure reproduces the causal
retention table's qualitative pattern (only the identity-conjunct's removal
collapses match at the correct source). Failure = report as failure; do NOT
substitute a weaker proxy and declare success (see §6).
Context from Tier 1.5 of the previous program: branch magnitude misleads; the
identity conjunct may be small in norm but carry the SELECTIVITY (variance across
candidate sources). Score conjuncts by contrast at matched vs shuffled sources,
not by norm.

---

## 4. MDL accounting and the distortion metric (fix in Tier 0, never vary silently)

    DL(head) = min over codebooks [ DL(codebook params) + DL(exceptions) ]
               subject to pattern-distortion ≤ ε

- **Distortion is PATTERN-level, not logit-level:** primary metric = mean JS
  divergence between original and compressed attention distributions over a fixed
  eval token set (plus top-k overlap, k ∈ {1,5}, as secondary; plus downstream
  CE-loss delta for the tiny models where cheap). Rationale: softmax gauge +
  monotone slack make logit Frobenius the wrong loss; the computation is the
  pattern. Fix ε by calibration in Tier 0 (choose ε such that SVD at full rank -1
  is comfortably inside; report DL-vs-ε curves, not single points, for headline
  results).
- Count DL in floats for continuous params + log-counts for discrete structure
  (partitions, supports); document the exact convention once in code
  (mdl_accounting.py) and reuse everywhere. The interesting quantity is RATIOS
  (codebook vs SVD baseline at matched ε), which are robust to convention.
- Also report the per-head **logit-rank vs pattern-rank gap** (min rank preserving
  ε in logit-Frobenius vs in pattern distortion): the "computationally inert
  weight structure" statistic. Predicted large and unpublished.

## 5. Tiers

**Tier 0 — exactness + calibration (tiny models, layer 0).**
1. Verify A1-A3 from source; log answers.
2. Implement folding + RoPE expansion; gate: fp64 reconstruction of actual layer-0
   scores/patterns from {C_f^b, S_f^b} to ~1e-10. Both branches, both heads.
3. Gauge checks: row-centering leaves patterns invariant (numerically).
4. Fix distortion metric + ε; fix DL conventions; unit-test cross-associations and
   HODLR codebooks on synthetic planted-structure matrices (planted bicluster,
   planted Toeplitz, planted conjunction) — each codebook must WIN on its own
   plant and LOSE on the others' (this is the codebook-selectivity sanity check).

**Tier 1 — tiny bilinear models, full analysis (the main event).**
1. All layer-0 heads: per-branch folded matrices, frequency-sparsity profile,
   full codebook menu, MDL table (DL ratio vs SVD at matched ε).
2. The L1H2 conjunction test (§3), with path-folded key side. Pre-registered
   success criteria above. This is the headline result if it passes; the honest
   negative if not.
3. Positional heads: identify heads whose {C_f, S_f} mass is concentrated in
   high-f bands with near-constant token structure; compress to
   frequency+Toeplitz codebook; report the DL collapse.

**Tier 2 — a real small LM with RoPE (Pythia-160m/410m: rotary but LayerNorm —
Claude Code should check and adapt the norm folding; or a small
Qwen/OLMo-class model if RMSNorm is preferred; pick ONE, justify in log).**
1. Vocab subset selection (see G-mem below): word-filtered 5-10k tokens.
2. Layer-0 heads (exact modulo norm caveat): the 144-heads-style MDL table;
   headline = fraction of heads where a structured codebook beats SVD by >2× at
   matched ε, broken down by codebook type (≈ a taxonomy of QK computation).
3. Shared dictionary across heads (codebook 4): corpus-level DL vs per-head SVD.

**Tier 3 — extensions (only after 1-2 land).** Path-folded matrices for known
GPT-2-class circuits if a learned-abs-pos model is added (requires the 4-way
content/position split + norm approximation, both to be stated); joint QK-OV
with shared key-side dictionary; token-tree transfer test (does E's tree compress
the heads?).

## 6. Anti-drift rules (lessons from the previous autonomous run — binding)

1. **Exactness/reconstruction gates before structure claims.** No MDL number is
   reported from a pipeline whose Tier-0 gate isn't currently passing; re-run
   gates after EVERY solver/algebra change.
2. **Pre-registered decisive tests.** The L1H2 conjunction test and the Tier-2
   MDL table are the decisive outputs, with success criteria as written. If a
   test can't be run as specified, log a QUESTION FOR LOGAN and do the nearest
   cheaper thing labeled as a proxy — never promote a proxy to a verdict.
3. **Negative results are results.** "SVD is not beaten on head X / class Y" goes
   in the table, not in a rationalization.
4. **No silent metric changes.** ε, the distortion metric, and DL conventions are
   fixed in Tier 0; any change invalidates prior numbers and requires rerunning
   the affected tables (say so in the log).
5. Engineering hygiene from last time: detach weights from autograd before
   solver loops; @no_grad on solvers; fp64 for exactness gates; never materialize
   V×V at full V (see G-mem).

## 7. Gotchas

- **G-mem: V×V is huge.** Full-vocab folded matrices at V=50k are 2.5e9 entries
  per head per frequency term — never materialize. Tiny models: V is small,
  materialize freely. Real LMs: word-filtered subsets (5-10k rows/cols) for
  block/tree codebooks + implicit/randomized linear algebra (the factors
  Ê W_Q^T and W_K Ê^T are thin; everything SVD-ish can run factored).
- **G-freq: RoPE Δ-range.** {C_f, S_f} are Δ-independent, but any Δ-averaged or
  Δ-sliced summary depends on the context-length distribution; fix the Δ range to
  the training context and say so.
- **G-subset (inherits G3 of the vocab doc):** BPE junk dominates; word-filter,
  and report block contents, not just DL.
- **G-center:** row-center AFTER folding and per frequency term; centering the
  slices ≠ centering the assembled M(Δ) — center the object you decompose, and
  verify pattern-invariance numerically (Tier 0.3).
- **G-branch-gauge:** in case A3(a), (M₁, M₂) → (cM₁, M₂/c) is invisible; fix
  branch scale (unit Frobenius per branch, absorb into a head scalar).
- **G-tie:** if the tiny models share/tie any weights between branches, the
  identifiability claim in §3 weakens; check.

## 8. Deliverables

1. tier0 exactness report (assumptions A1-A3 verified, gates passing).
2. Tiny-model MDL table (per head × per branch × per codebook) + the L1H2
   conjunction verdict with the causal-table comparison.
3. Real-LM head-taxonomy MDL table + the logit-rank vs pattern-rank gap
   distribution.
4. mdl_accounting.py, folding.py (with RoPE expansion), codebooks/ (svd,
   cross_assoc, hodlr, sparse_bilinear, positional), all gate tests under tests/.
