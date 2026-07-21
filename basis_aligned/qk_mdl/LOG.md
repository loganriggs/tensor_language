# qk_mdl program log

Binding spec: [qk_mdl_spec.md](qk_mdl_spec.md) (esp. §6 anti-drift rules). One dated
entry per tick. Gate status must be current before any MDL number is reported.

---

## 2026-07-14 — tick 0 (spec §0: A1–A3 verified from source; program armed)

Verified directly from `model.py` (class `Attention`, `Rotary`) and the checkpoint
configs in `runs_owt/attn2-*`:

- **A1 (norm):** pre-RMSNorm with `elementwise_affine=False` — **no learnable γ** to
  fold; effective embedding is exactly ê_t = e_t / rms(e_t). Checkpoints have
  `norm="rms"` on. Residual is **lerp with scale 0.5**: x_out = 0.5·x + 0.5·o(z) —
  affects path-folded weights for layer 1 (embedding arrives at layer-1 with weight 0.5
  along the direct path).
- **A2 (RoPE):** rotate-half convention (chunk d_head into two halves, (a,b)→(−b,a)),
  **all d_head dims rotated**, base 10000, d_head=32 → **16 frequency bands**; both q and
  k rotated, in **both** branches (q1,k1,q2,k2 each pass through `self.rotary`).
- **A3 (CRITICAL — differs from both spec cases):** there is **NO softmax anywhere**.
  `pattern = (q1·k1)(q2·k2) / d_head² * causal_mask` — a multiplicative mask on the raw
  product of two bilinear forms; the model is polynomial in its inputs. Consequences:
  - No softmax gauge: the per-query-constant invariance of §1.2 does NOT hold; do not
    row-center as a gauge fix (there is no gauge). Global scale gauge between branches
    (G-branch-gauge) still applies.
  - §4's JS-divergence pattern metric does not apply as written (patterns are not
    distributions; entries can be negative).
- **G-tie:** q1/k1/q2/k2 are four separate `nn.Linear(d_model,d_model,bias=False)` — no
  tying; §3 identifiability claim stands.
- Models: attn2-* = 2 bilinear attention layers, d_model=128, n_head=4 (d_head=32),
  V=5120, n_ctx=256. V×V = 26M entries — materialize freely (G-mem satisfied for tiny
  models).

**QUESTION FOR LOGAN (spec deviation, per anti-drift rule 2):** with no softmax, the
pattern-level distortion metric must be chosen fresh. Provisional choice (to be
calibrated in Tier 0.4, flagged, not silently varied afterward): primary = relative MSE
on the masked pattern (‖P̂−P‖²/‖P‖² over the eval token set, per head), secondary =
downstream ΔCE of the tiny model with the compressed head patched in (cheap at this
scale, and it is the program's own preferred audit). Say if you want a different primary.

Next steps (tick 1): Tier 0.2 — implement `folding.py` with the exact RoPE expansion
{C_f, S_f} per branch (16 bands, rank ≤ 2 slices each: score(t_q@i, t_k@j) =
Σ_f cos(ω_f Δ)·C_f[t_q,t_k] + sin(ω_f Δ)·S_f[t_q,t_k], per branch, then product), and
the fp64 reconstruction gate (~1e-10) against actual layer-0 scores of
attn2-mix10-seed0, both branches, all 4 heads. Then Tier 0.4 planted-structure
synthetics = the ground-truth-MDL battery Logan asked for (each codebook must WIN on its
own plant, LOSE on the others').

---

## 2026-07-14 — tick 1 (Tier 0.2–0.3: folding + fp64 exactness gate — PASSING)

Built `folding.py` (exact {C_f, S_f} RoPE expansion per branch, rank-≤2 per band from
per-token factors; `band_mass` computes Frobenius band profiles without materializing
V×V) and `tier0_gate.py`.

**Gate initially FAILED at 1e-10 (errors ~1e-4–1e-6). Root cause found in MODEL SOURCE,
not the algebra:** `Rotary.__init__` computes its frequency tables with `.float()` (fp32),
so the checkpointed models' cos/sin tables carry fp32 precision even after casting the
model to fp64. Fix: the gate's expansion builds cos(ωΔ)/sin(ωΔ) from the model's own
cached tables via the exact difference identities (`use_model_trig=True`), which tests
the folding algebra against the deployed model; the analytic-fp64-ω variant
(`use_model_trig=False`, the right object for downstream folded matrices) deviates from
the deployed model by ~0.5–2.3e-4 on branch scores — a stated property of the
checkpoints, to be absorbed into ε calibration in Tier 0.4, never silently.

**GATE STATUS: PASS** on attn2-mix10-seed0, attn2-dense-seed0, attn1-seed0 (layer 0, all
4 heads, both branches): pattern max err 1.6–3.6e-15, branch scores ≤1.4e-13, branch-scale
gauge 9e-16, manual ê formula = module to 9e-16 (torch RMSNorm eps = finfo.eps).
Report: `tier0_report.json`.

Descriptive (no MDL claims): band-mass profiles show strong mid-band concentration —
e.g. L0H1_b2 puts 57% of Frobenius mass in band 7; several head/branches have top-3
bands ⊂ {5..9}. Frequency-sparsity (codebook 5) looks promising; quantify properly only
after Tier 0.4 fixes ε and DL conventions.

Next (tick 2): Tier 0.4 — fix distortion metric (provisional: relative pattern MSE +
downstream ΔCE; QUESTION FOR LOGAN from tick 0 still open) and ε calibration;
`mdl_accounting.py` with the DL conventions; then the planted-structure battery
(bicluster / Toeplitz / conjunction plants — each codebook must WIN its own plant and
LOSE the others') = the ground-truth-MDL component.

---

## 2026-07-15 — tick 2 (Tier 0.4: conventions FROZEN + planted battery PASSING)

Gate re-run first (anti-drift rule 1): **PASS** (unchanged, ~2e-15).

**Conventions frozen** in `mdl_accounting.py` (change = announce + rerun): DL in bits
(32/float + log₂ for discrete choices + 32+log₂(pool) per exception); matrix distortion =
relative Frobenius² (no centering — no gauge in the no-softmax models); model-level
distortion (provisional pending Logan, flagged): relative pattern MSE primary, ΔCE
secondary; battery ε = 1.5× plant noise floor.

**Codebooks implemented** (`codebooks.py`): svd (rank-minimal at ε), bicluster
(cross-associations flavor: separate row/col partitions, alternating minimization,
spectral init via k-means on top singular vectors, k doubled until ε met), toeplitz
(diagonal-profile + Fourier truncation). Pending: HODLR/tree, sparse-bilinear/conjunction.

**Battery** (`tier04_battery.py`, N=512, three plants with known true DL):

| plant | SVD | bicluster | toeplitz | true DL | winner |
|---|---|---|---|---|---|
| low-rank(8) | **262.4k (r=8, = true)** | fail | fail | 262.4k | svd ✓ |
| bicluster(8×8) | 229.6k | **12.3k (k=16)** | fail | 5.1k | bicluster ✓ |
| Toeplitz(6 modes) | 393.6k | fail | **0.4k (= true)** | 0.4k | toeplitz ✓ |

**SELECTIVITY: PASS** (3/3). Notes: (a) the battery CAUGHT a real solver bug on its first
run — random-init biclustering needed k=128 on its own k=8 plant and LOST to SVD; fixed
with spectral init (this is the positive-controls lesson doing its job). (b) Known
remaining solver gap: bicluster meets ε at k=16, not the planted k=8 (2.4× true DL) —
selectivity unambiguous, exact-k recovery would need more restarts; noted, not hidden.
(c) SVD pays 45× ground truth on the bicluster plant — the "computational ≪ spectral MDL"
direction, quantified on a plant.

**Gate status: PASS. Battery: PASS (3 of 5 codebooks; conjunction + tree pending).**

Next (tick 3): sparse-bilinear codebook (masked-projector solver per spec §2.4 with the
listed fixes) + conjunction plant (M₁⊙M₂ of two cheap-codebook matrices; owner must win),
and the HODLR/tree codebook if time. Then Tier 1.1 (real layer-0 heads, full menu, MDL
table) — ε calibration for real heads per §4 at that point.

---

## 2026-07-15 — tick 3 (Tier 0.4 complete for 4 codebooks: conjunction plant + codebook, battery PASS 4/4)

Gate re-run first: **PASS** (unchanged). Added `fit_conjunction` (M ≈ bicluster ⊙ Toeplitz
gate, alternating weighted LS; DL = DL(blocks) + DL(gate Fourier) + 1 scale float) and the
conjunction plant (bicluster(8) ⊙ positive 6-mode gate ∈ [0.2,1.8]).

**Battery: SELECTIVITY PASS 4/4** (tier04_battery.json):

| plant | svd | bicluster | toeplitz | conjunction | true DL | winner |
|---|---|---|---|---|---|---|
| low-rank(8) | **262.4k (=true)** | fail | fail | fail | 262.4k | svd ✓ |
| bicluster(8²) | 229.6k | **12.3k** | fail | 12.4k | 5.1k | bicluster ✓ |
| Toeplitz(6) | 393.6k | fail | **0.4k (=true)** | fail | 0.4k | toeplitz ✓ |
| conjunction | 1246.4k | fail | fail | **38.3k** | 5.6k | conjunction ✓ |

The battery caught the SAME solver-class bug a second time: fit_conjunction's inner
biclustering with random partition init needed k=64 on its own plant (25× true DL) and
failed outright on the pure-bicluster plant it should express trivially. Fixed with
spectral partition init on the gate-whitened matrix M/c₀ — after which conjunction wins
its plant by 33× over SVD and correctly loses to plain bicluster by exactly the
constant-gate overhead (12.4k vs 12.3k).

Honest gaps (logged, not hidden): (a) conjunction meets ε at k=32 vs planted k=8 → 7×
true DL (alternation recovers structure partially; win margin unaffected); (b) bicluster's
k=16-vs-8 inflation from tick 2 persists; (c) conjunction fails on the pure-Toeplitz plant
because that plant's gate oscillates through zero and the blind-from-product fit assumes a
positive gate (documented identifiability limit — per-diagonal signs cannot be absorbed by
block-constant factors; the REAL pipeline decomposes branches separately, spec §3, so
blindness never arises). SVD pays 33–240× the owner on structured plants — the
computational-vs-spectral-MDL direction on known ground truth, now across 3 structure types.

**Gate: PASS. Battery: PASS 4/4. Pending: tree/HODLR codebook; shared-dictionary
sparse-bilinear (Tier 1.3).**

Next (tick 4): Tier 1.1 — the real thing: all layer-0 heads of attn2-mix10-seed0, per
branch: materialize folded band matrices {C_f, S_f} (V=5120 fine), ε calibration per §4
(SVD at full-rank−1 comfortably inside; DL-vs-ε curves), full codebook menu, first real
MDL table. The tick-0 QUESTION (distortion metric under no-softmax) becomes load-bearing
here; provisional metric will be used AND labeled provisional in every table until Logan
answers.

---

## 2026-07-15 — QUESTION RESOLVED (Logan): distortion metric for real heads

Logan: "MSE and CE delta seem good for now. Would highlight the CE delta one."
Frozen in `mdl_accounting.py`: **ΔCE = headline + binding audit** (tables gated and ranked
by it); relative pattern MSE = search-loop metric + secondary column; ε_pattern calibrated
so the SVD baseline's ΔCE is comfortably small, then frozen. Tier 1.1 (next tick) proceeds
with the metric no longer provisional.

---

## 2026-07-15 — Tier 2 directed push (Logan: Elriggs models, 10h budget) — part 1

Target models identified from configs + state dicts + `jacclust/tt_model.py` (the modeling
code was already in-repo from the jacclust program):
- **bilin18** = gpt2-bilinear-sqrd-attn-18l-9h-1152embd (546M): TWO QK branches
  (c_q,c_k,c_q2,c_k2), pattern = (q1·k1)(q2·k2)/D² causal-masked UNNORMALIZED; bilinear
  MLPs. (The plain "gpt2-bilinear-18l" has bilinear MLPs but STANDARD attention —
  config bilinear_attn=false — not the target.)
- **sqrd12** = gpt2-sqrd-attn-12l-6h-768embd (162M): ONE branch, pattern = (q·k/D)²
  ROW-NORMALIZED (per-query positive-scale gauge EXISTS here) — "the smaller one,
  only squared attention" ✓.
- Family facts: modded-nanogpt lineage — global rms after embed, x0-skip lambdas, per-head
  QK rms-norm THEN RoPE (rotation sign OPPOSITE to tiny models → S_f = qb·ka − qa·kb),
  value mixing with block-0 v, logit cap 30·tanh(/30), bf16 rotary tables in source.

**CE gate (Logan: verify 3–4):** built fp-controlled reference forward
(`tier2_model.py`); verified EXACT vs tt_model.GPT (5.4967 = 5.4967).
- sqrd12: CE 3.497 @ T=1024 ✓.
- bilin18: CE 5.50 @ T=1024 — **out of range**, diagnosed via per-position CE: healthy
  3.3–3.9 through pos ~512, exploding to ~11 by pos 1000. The unnormalized score-product
  pattern has row mass growing with context; the model's competent regime ends ~T=512
  (CE@256 3.63, CE@512 ≈ 3.5). **Eval regime frozen at T=512 for both models** — gate
  satisfied there; context-degradation logged as a model property (FINDING for Logan).

**Layer-0 folding for this family (`tier2_folding.py`):** the per-head QK rms-norm makes
each branch's ENTIRE folded object a factor pair (q̂,k̂) ∈ (V×128)² of unit-RMS per-token
vectors (lambdas drop out at layer 0 — killed by the pre-attn rms-norm). All codebooks act
on factors; V×V never materialized; a pure positional head = token-VQ at k=1.
**Fold gate: PASS at 1e-15, both branches.**

Running: `tier2_mdl.py` — 9 heads × 2 branches, codebooks {svd-r, vq-k, band-m},
pattern-MSE search loop + ΔCE binding audit (per Logan's metric answer), T=512.

---

## 2026-07-15 — Tier 2 part 2: bilin18 layer-0 MDL results (gates passing; ΔCE-audited)

Baseline CE 3.2341 (T=512, pile-10k, 16 audit chunks). Full layer-0 QK DL = 884 MiB
(raw factors, frozen conventions). Per-head grid: {svd-r, vq-k, band-m, zero} × 9 heads ×
2 branches, ΔCE audited for every candidate (`tier2_audit_bilin18.json`).

**FINDING T2-1 (pattern-metric failure, real 546M model):** the pattern-MSE screen is
useless as a behavioral predictor here — vq16 fits with pattern-MSE 0.14–0.95 cost
|ΔCE| ≤ 0.011; L0H3's vq16 (pattern-MSE 0.139) IMPROVES CE by 0.011. Same moral as
basis_aligned e6/e10, now in attention weight-space of a real model.

**FINDING T2-2 (per-head marginals):** 7 of 9 heads can be individually ZEROED at
|ΔCE| ≤ 0.011 (several negative); only H3 (+0.034) and H6 (+0.010) resist, and each
compresses to vq16 (~1250× per head-branch). BUT—

**FINDING T2-3 (marginals do NOT compose):** jointly zeroing the 7 "free" heads costs
+0.534 nats (vs ~+0.03 summed marginals) — massive cross-head redundancy: individually
expendable, collectively load-bearing. Zeroing is the WRONG compression for redundant
heads.

**FINDING T2-4 (headline): the ENTIRE layer-0 QK computation is a ~256-token-class
computation, behaviorally.** Joint frontier (`tier2_joint_bilin18.json`):
all-heads vq256 → ΔCE **+0.0084** at **165× DL reduction** (5.4 MiB vs 884 MiB);
all vq16 → +0.042 at 1240×. Same ΔCE as keeping H3,H6 exact at 37× more DL.

**FINDING T2-5 (readability):** the vq16 classes are crisp token-type/morphology
structure — H3: digit class, punctuation classes, sentence-initial class (In/It/We/This),
an odd/even-flavored uppercase split (B,D,F,G,H,J,L,N vs A,C,E,K,M,O); H6: function
words, morphological suffixes (ion/ter/ers/ould/ines), a semantic-noun class
(people/government/women/police), determiners (their/its/these/every). Exemplars in the
session log; CUR/exemplar dump per §2.7 to be attached in the results doc.

Caveats: single eval distribution (pile-10k) at T=512 (the model's competent regime —
see part 1); vq classes fit on factors under L2 (not behaviorally optimized — the
basis_aligned e7 lesson says CE-trained codebooks would do better still); ε levels
reported as curve points {0.001,0.01,0.05}-ish rather than one number. sqrd12 run in
progress.

---

## 2026-07-15 — Tier 2 part 3: sqrd12 + synthesis (TIER2_RESULTS.md)

sqrd12 audited (baseline 3.372 @T=512): joint vq256 ΔCE +0.116 at 6.1e-3 DL — ~15× less
behaviorally compressible than bilin18 at matched ratio. No free head-zeros (H3 +0.356
ablated, but svd16 ≈ free — low-rank AND load-bearing). Contrast finding: two-branch
unnormalized 546M ≫ one-branch normalized 162M in layer-0 QK compressibility; candidate
explanations (head count/redundancy, row-normalization sensitivity, capacity) NOT
disentangled — logged as open.

Deliverables: TIER2_RESULTS.md + fig_tier2_frontier.png + tier2_audit_{bilin18,sqrd12}.json
+ tier2_joint_bilin18.json. All gates passing at time of report.

---

## 2026-07-15 — tick 4 (Tier 1.1: tiny-model layer-0 MDL table; reference gate exact)

Gate re-run: PASS. New mini-gate: tiny-model reference forward (with score patching)
reproduces the model bit-exactly (max logit diff 0.0e+00, fp64) and baseline CE 4.634 ≈
recorded 4.637. Full grid ΔCE-audited (`tier1_mdl_attn2-mix10-seed0.json`).

**FINDING T1-1: the tiny model is the STRUCTURAL OPPOSITE of the 546M model.** Layer-0
heads are rank-compressible (svd16 = half rank ≈ free on all 8 head-branches at
|ΔCE| ≤ 0.009; svd4–8 suffices for half of them; even svd1 costs only +0.02–0.18 on 5/8)
but NOT token-clusterable: vq1 costs +0.24–2.19 per head-branch and the joint token-class
frontier is terrible (all-vq256 +2.73 vs bilin18's +0.008; all-vq1024 still +0.25).
All-zero layer-0 QK: +16.7 (layer 0 is half the model). Interpretation: a 2-layer model
must carry fine-grained token identity through layer-0 QK; an 18-layer model's layer-0 is
a coarse token-type router. Scale/depth story for the taxonomy table.

Caveats: joint-vq curve non-monotone (vq16 +1.39 < vq64 +1.57 < vq256 +2.73) — k-means
seed variance suspected (single seed, L2-fit); flag, do not interpret the bumps. Joint
svd frontier not yet audited (next tick alongside L1H2).

**Next (tick 5): Tier 1.2 — the pre-registered L1H2 conjunction test** (path-folded key
side through L0 OV per §1.3/§3), success criteria as written in the spec; the reference
forward + patching machinery from this tick is the substrate. Also joint-svd frontier +
frequency profiles for the positional-head sweep (1.3).

---

## 2026-07-15 — tick 5 (Tier 1.2 attempt: pre-registered test BLOCKED as specified; substitute null + positive control; target re-anchored)

Gate re-run: PASS (3/3).

**DEVIATION (anti-drift rule 2): the pre-registered target `attn2-seed0` no longer exists
on disk** (runs_owt has no such run; mechdecomp's Tier-1.5 loaded it in a prior epoch of
the repo). Ran the nearest substitute + a positive control instead; no verdict promoted.

1. **attn2-dense-seed0 (nearest surviving relative): NULL.** No match-and-copy behavior
   at all — all L1 heads at/below chance on match@source (mass ~0.003, argmax ≤0.003);
   no identity structure in ANY (branch × L0-head) path-folded G matrix (hit rates ≈
   1/V chance, diag z ∈ [−0.13, +0.11]). This checkpoint does not implement the circuit;
   the null is about the checkpoint, not the hypothesis.
2. **Positive control (attn2-s30k-mix50-rp-dense-seed0, the genuine content-induction
   model with documented causal table): my screens recover the documented circuit** —
   L1H0/L1H3 = the redundant copy pair (match argmax 0.18/0.26 ≈ 25–30× chance), L0H1 =
   dominant prev-token head (0.147) matching its −99% causal rank. Machinery validated;
   P(copy) proxy 0.248 vs documented 0.748 — metric/data convention gap (theirs: tiled
   burst format + their copy metric), to reconcile before quantitative comparison.
3. **Design lesson (logged for §3): zeroing a branch is NOT a branch-specific
   intervention in product attention** — pattern = s1·s2, so kill_b1 ≡ kill_b2 ≡ kill
   head (identical CE 5.0313 observed). Branch-causal probes must REPLACE scores
   (mean/shuffle/structure-ablated), not zero them. tier12_conjunction.py's causal arm
   is redesigned accordingly for the rerun.

**QUESTION FOR LOGAN:** attn2-seed0 (the .434→.031 retention-table model) is gone from
runs_owt — do you have it elsewhere, or should the pre-registered conjunction test be
re-anchored to attn2-s30k-mix50-rp-dense-seed0 (genuine content induction, documented
multi-head causal table in mechdecomp/tier15_induction.py, screens reproduced here)?
Proceeding with the rp model next tick unless redirected.

Next (tick 6): full conjunction test on the rp model, heads L1H0+L1H3: per-branch
path-folded identity structure through each L0 head (chance-calibrated), positional/band
diagnostics, and score-REPLACEMENT branch interventions; reconcile the P(copy) metric
with tier15_induction's convention first.

---

## 2026-07-15 — tick 6 (Tier 1.2 re-anchored: conjunction test on the genuine induction model — PARTIAL PASS with a sharper structure than pre-registered)

Gate: PASS. Guard: base P(copy) 0.7467 ≈ documented 0.7483 ✓; copy heads L1H0/L1H3
confirmed (match argmax 0.123/0.122); conventions reconciled with tier15_induction
(uniform-random tokens tiled P=96, softmax-P(target) metric).

**Causal results (`tier12b_conjunction.json`, `tier12b_combos.json`):**

| intervention (positional-average = destroy token identity, keep Δ-profile) | ΔP(copy) |
|---|---|
| one branch of one head (any of the 4) | −0.026 … +0.001 |
| BOTH branches of one head (full token-lobotomy of one copy head) | +0.004 / +0.011 |
| the two **L0H1-key-fed** branches (H0.b1 + H3.b2) jointly | **−0.487** |
| the two diffuse branches (H0.b2 + H3.b1) jointly | −0.138 |
| all four | −0.517 |

Key-path ablations: H0.b1's and H3.b2's key inputs depend on **L0H1 alone** (−0.51/−0.49;
other L0 heads ≈ 0), while H0.b2/H3.b1 are diffuse (L0H0/L0H1/L0H3 all matter). The two
copy heads use OPPOSITE branches for the identity conjunct.

**VERDICT vs pre-registered criteria: PARTIAL PASS (structure sharper than hypothesized).**
- Conjunction structure EXISTS and is branch-specific: per copy head, exactly ONE branch
  carries the token-identity conjunct (key side through the prev-token head L0H1); the
  other branch is comparatively positional/diffuse. ✓ (spec's core claim)
- The pre-registered single-head collapse criterion FAILS — but for the documented reason
  (redundant copy pair): identity destruction must hit BOTH heads' identity branches to
  collapse the circuit (−0.487), and does. Circuit-level conjunction: ✓.
- Weight-space identity codebook: PARTIAL — the only strong generic-weights identity
  signal is (H3, b2, via L0H0) at 380× chance (z +2.05); (H0, b1, via L0H1) is weak
  (4× chance). Same generic-vs-data-conditioned gap mechdecomp Tier 1.5 documented:
  causal identity routing (via L0H1) is a data-conditioned minority direction in weight
  space. Data-conditioned structure metrics are the fix (future tick).

Tick-5 design lesson applied: all branch interventions are REPLACEMENTS (per-Δ means),
never zeros. QUESTION FOR LOGAN from tick 5 (attn2-seed0 whereabouts / formal
re-anchoring) still open; results above stand on the re-anchored model regardless.

Next: either (a) data-conditioned weight-space identity metric (condition G on induction
positions — predicted to move the L0H1 signal into both identity branches), or (b) return
to spec order: Tier 1.3 positional heads + tiny-model MDL table completion (joint-svd),
or (c) Tier 3 path-folded MDL. Cron default: (b) then (a).

---

## 2026-07-15 — tick 7 (Tier 1.3: positional-head sweep = clean NEGATIVE; mix10 joint-svd frontier)

Gate: PASS (3/3). Positional codebook = per-Δ score replacement (token structure
destroyed, Δ-profile kept), classification threshold |ΔCE| ≤ 0.01 (+ |ΔP(copy)| ≤ 0.02
for the rp model). Full sweep: 16 branches × attn2-mix10-seed0 + 16 × rp model
(`tier13_positional.json`).

**FINDING T1-2 (negative, per anti-drift rule 3): ZERO behaviorally-positional branches
in either tiny model.** Minimum cost +0.012 (rp L1H0b1); mix10 branches cost +0.07–2.18.
The spec's predicted positional-head DL collapse does not occur in this zoo. Two
sub-findings:
- **Pattern-positionality ≠ score-positionality:** the rp model's prev-token head L0H1
  (attends Δ=1 on average) LOSES the circuit when its scores are positional-averaged
  (ΔP(copy) −0.739): its score magnitudes are token-dependent and the identity branch
  reads its OV transport. A head can look positional in its pattern and be content-
  critical in its scores.
- rp L0H3 is extreme-content (+4.35 CE when positional-averaged) despite only −44% causal
  copy share; L0 branch pairs are near-symmetric in posavg cost (b1≈b2 to 3 decimals).
Cross-script consistency check: rp L1H0b1 posavg ΔP(copy) −0.0092 = tick 6's value ✓.

**mix10 joint-svd frontier (tick-4 leftover):** joint svd16 (half rank, all 8 layer-0
branches) +0.054; svd8 +0.202; svd4 +0.455; svd1/2 catastrophic (+3.5). Per-head svd16
was free (tick 4) → mild non-additivity (+0.054 joint), nothing like the 546M's vq
redundancy collapse. Confirms the depth-taxonomy: tiny = rank-structured, moderately
additive; big = token-class-structured, heavily redundant.

Tier 1 status: 1.1 ✓ (tick 4), 1.2 ✓ PARTIAL PASS re-anchored (tick 6; attn2-seed0
question still open), 1.3 ✓ NEGATIVE (this tick). Tier 1 complete pending Logan on the
re-anchoring. Next: data-conditioned weight-space identity metric (tick 6's open fix),
or Tier 3 path-folded MDL, or 546M layer-0 CE-trained codebooks (basis_aligned e7
lesson). Cron default: data-conditioned metric.

---

## 2026-07-15 — tick 8 (data-conditioned identity metric: tick-6 prediction CONFIRMED; Tier 1.2 upgraded to PASS)

Gate: PASS. Method: conditional-mean pre-rotary q/k vectors by token identity on tiled
induction data, key side decomposed by L0-head source with frozen empirical norm
(`tier12c_conditioned.py`, full 5120-token coverage).

**The pre-stated prediction (tick 6) is confirmed exactly.** Identity structure appears
in precisely the two causal identity branches, exclusively via L0H1:

| branch × source | identity hit rate (chance 0.0002) | diag z |
|---|---|---|
| L1H0.b1 via **L0H1** | **0.4443** (2200× chance) | +3.23 |
| L1H3.b2 via **L0H1**, gauge-corrected | **0.4227** | −3.22 (sign = branch gauge) |
| every other (branch × source) cell, incl. direct & L0H0/2/3 | ≤ 0.0004 | \|z\| ≤ 0.09 |

- The generic-vs-conditioned attribution gap is resolved as mechdecomp predicted:
  generic weights said (H3.b2 via L0H0); the data-conditioned metric says via L0H1 —
  matching the causal key-path ablations (tick 6) exactly.
- The sign flip between the two heads' identity diagonals is pure **branch-sign gauge**
  ((−s₁)(−s₂)=s₁s₂, spec §7 G-branch-gauge): |z| is the gauge-invariant statistic;
  under sign correction the two heads are near-identical (0.444 vs 0.423). The copy
  pair implements ONE identity conjunct twice, in opposite branches, opposite signs.

**Tier 1.2 combined verdict upgraded to PASS (re-anchored):** structure criterion ✓
(identity-plus-noise in exactly one branch per copy head, via the causal source, under
the data-conditioned codebook), causal criterion ✓ at circuit level (tick 6, −0.487
joint collapse), with the single-head redundancy caveat and the attn2-seed0 re-anchoring
question (still open for Logan) both documented.

Program state: Tiers 0, 1, 2 complete. Remaining spec items: Tier 3 (path-folded MDL
for deeper layers; joint QK-OV; token-tree transfer), CE-trained codebooks on the 546M
(basis_aligned e7 lesson), attn2-seed0 question. Cron default next: Tier 3 path-folded
MDL table for the rp model's layer-1 through the L0 paths (the machinery from this tick
is most of it).

---

## 2026-07-15/16 — tick 9 (Logan's directed batch: results/ folder, CE+KL codebooks, Tier-3 opener)

Gate status: PASS throughout (re-run at tick start). Three deliverables, all committed:

**1. `results/` subfolder (Logan's request):** per-experiment MD files (README + 6) with
method explanations, inline figures (Tier-2 frontier, conjunction causal bars,
conditioned-G identity diagonal, tiny-model frontier), and decomposition examples —
conditioned-match examples and the 546M vq16 token classes (clean linguistic categories:
determiners, derivational suffixes, abstract nouns, past-tense verbs, BPE fragments).

**2. CE-trained + KL-distilled codebooks (546M layer-0), the headline:**

| joint codebook | DL ratio | L2-fit | CE-trained | KL-distilled |
|---|---|---|---|---|
| all vq16 | 1240× | +0.044 | **−0.019** | — |
| all vq64 | 500× | +0.015 | **−0.032** | **−0.007** |
| all vq256 | 165× | +0.008 | **−0.039** | — |

Every CE-trained codebook OUTPERFORMS the original layer-0. The KL split shows faithful
compression alone reaches parity-or-better at 500× (−0.007 under pure imitation);
~−0.025 of the CE gain is domain adaptation. **A 64-token-class layer-0 QK is at least
as good as the trained 884 MiB computation.**

**3. Tier-3 opener (path-folded lookup codebooks): informative NEGATIVE**
(results/06_tier3_pathfold.md): replacing live layer-1 q/k with conditional-mean lookup
tables destroys the copy circuit (−0.62…−0.74 P(copy) held-out) even though those same
tables carry the identity structure at 0.44 hit rate. Structure-visible ≠
computation-sufficient: the circuit consumes context-dependent components (norm scales,
actual pattern weights, within-condition variance) that 0th-order-in-context tables
discard. Tier-3 codebooks must be ≥ first-order in context (live L0 pattern × quantized
OV content). Shared-table (joint QK) question unresolved (per-head tables already fail).
Logan's MLP-two-inputs note recorded for deeper tiers. (One artifact rerun: json crash on
tuple keys — fixed, rerun, numbers unchanged.)

---

## 2026-07-16 — tick 10 (OV circuit + bilinear-MLP blocks, Logan's steer)

Gate: PASS (block-split no-drop gate exact to 2.4e-7). `ov_blocks.py`,
`results/07_ov_blocks.md`.

**FINDING OV-1 (block importance, block-0 bilinear MLP):** drop self +1.291, drop CROSS
+0.840, drop source-pair +0.187. Logan's cross-term object (token × attention-out inside
the bilinear encoder) is a first-class computation; his near-one-hot intuition mostly
holds (source-pair 5–7× smaller) but source×source interaction is nonzero.

**FINDING OV-2 (selection/content dichotomy):** OV value tables are NOT coarsely
classable (vq64 +2.02, vq1024 +0.88, zero +4.36) — opposite of QK on the same model at
the same ratios. Selection is a ~256-class computation; content needs fine token
identity, like the raw embedding in basis_aligned e6. CE-training of OV tables running
(the e6→e7 move); results to follow in ov_ce_trained.json.

Next: V×V cross-block codebook (token × transported-token → hidden) as its own object.

Addendum tick 10: OV CE-training landed — vq1024 +0.917→+0.568, vq4096 +0.782→+0.475
(~38% recovery only; QK went negative under identical treatment). The selection/content
dichotomy is REAL, not metric mismatch. Exact basis_aligned parallel: hard vq fails on
content, sparse coding rescued the embedding (+0.87 vs +0.26) → next OV codebook = top-k
sparse coding of value tables. Queued with the V×V cross-block codebook.

---

## 2026-07-16 — tick 11 (Logan's requests: methods explainer + unified graph + pattern display)

Gate: PASS. Three deliverables in `results/`:
1. **00_methods.md** — every codebook method with its working code snippet, intuition,
   and where it won/lost (svd, vq/bicluster, band, toeplitz/positional, conjunction,
   conditional-mean lookup, CE/KL-trained).
2. **fig_methods_compare.png** — all families on ONE object (546M layer-0 QK, joint,
   ΔCE-audited). New joint arms filled in (`tier2_joint_families.py`): joint svd16 =
   +0.0045 at 12.5% DL (svd64 NEGATIVE at 50% — stronger than the per-head view
   suggested), band needs 48/64 bands, joint positional +1.47. Tidy decomposition: of
   layer-0 QK's ~2.5-nat contribution, ~1.0 is purely positional, ~1.5 token-selective,
   and token CLASSES capture nearly all of the selective part at 20× less DL than rank.
3. **08_pattern_display.md / fig_pattern_display.png** — attention patterns computed
   FROM the best method (vq256 CE-trained) side-by-side with the originals on real text,
   token-labeled. 48% pattern rel-MSE, better CE — the dissociation made visible.

---

## 2026-07-16 — tick 12 (Logan's methods questions: expanded explainer, class-annotated display, shared-registry test)

Gate: PASS. Deliverables:
1. **00_methods.md rewritten** — "factors" defined with full code (q̂,k̂ ∈ (V, d_head)
   per head-branch, pre-rotary, exactly generating all scores); every method's snippet
   expanded to include helper definitions; conjunction section rewritten step-by-step
   (weighted-LS alternation, monotone objective, identifiability caveat); FAQ added.
2. **Pattern display regenerated with class annotations** (`token·c17` labels): axes are
   sequence positions labeled by actual tokens; ENTRIES depend only on (class, class, Δ)
   — same-class tokens share pre-rotary factors, RoPE differentiates positions.
3. **Shared-registry experiment** (`shared_registry.json`) answers "reduce the Embedding
   itself in one class structure?": NO —
   QK own classes +0.008 / QK global +0.051 (selection robust to partition choice);
   OV own +1.383 / OV on QK's classes +1.813 / OV global +2.472 (content tolerates none;
   QK's classes are WORSE for OV than OV's own); both-global-256 +2.777.
   **FINDING SR-1: "which tokens are interchangeable" is CIRCUIT-SPECIFIC — no single
   privileged coarse structure exists on the embedding; each reader induces its own
   partition.** In forward passes no shared reduction is needed anyway: the QK codebook
   replaces only scores; v reads the full embedding (class-precision selection ×
   full-precision content).

---

## 2026-07-16 — tick 13 (OV sparse coding: prediction CONFIRMED, content compresses too)

Gate: PASS. `ov_sparse.py` / `ov_sparse.json`; results/07 updated.

**FINDING OV-3: sparse coding rescues OV content** (tick-10 prediction confirmed).
L2-fit top-k (512 atoms, k=16 signed coefficients per token, per head): ΔCE +0.034 where
hard vq256 cost +1.383. CE-trained (supports frozen, atoms+coeffs through the frozen
model): **+0.044 → −0.019 — better than the original values.**

Refined dichotomy: selection tolerates hard classes; content needs sparse combinations;
under matched behavioral training BOTH layer-0 circuits beat the original (QK −0.039,
OV −0.019). The basis_aligned e7 pattern (vq +0.87 vs sparse +0.26) reproduced on
attention circuits.

Queue: V×V cross-block codebook (block-0 bilinear MLP, justified by +0.84 importance);
first-order-in-context path codebooks (Tier 3); attn2-seed0 question still open.

---

## 2026-07-16 — tick 14 (Logan's advisor-message on clustering epistemics: CE-training procedure audited against the 3-tier ladder; tier-1 certificate computed)

Gate: PASS. New artifact: `tier1_certificate_vq256.json`.

**Our CE-training, stated precisely:** ALL model parameters frozen (requires_grad=False,
nothing else moves); discrete structure (token→class assignments / sparse supports)
frozen from WEIGHTS-ONLY k-means/top-k — data never selects the discrete structure; only
continuous tables train (QK centroid factors ~1.2M params; OV atoms+coeffs), each paid at
32b/float in the DL accounting; train chunks (pile-10k 20..147) disjoint from audit
(4..19); KL variant = teacher-CE to the ORIGINAL model.

Mapping to the ladder: model-side compensation channel CLOSED; codebook-side channel OPEN
by design (centroids drift from weight-derived values toward what the frozen downstream
prefers on-distribution) — which is why claims were already scoped to "on pile @T=512"
and why the KL arm exists (vq64: CE −0.032 vs KL-faithful −0.007 → adaptation ≈ −0.025,
quantified). MDL bookkeeping concern is narrower than the message fears: assignments are
data-free; only fully-paid floats are data-tuned.

**Tier-1 exhaustive certificate (computed, honest verdict: metric-dependent).** The
folded domain IS fully enumerated; for vq256 L2-fit, closed-form bound over ALL
(t_q,t_k,Δ): max ≤ 2.24, mean-case ≤ 1.21; exact sampled errors: mean 0.016–0.042,
p99 ≤ 0.17, sampled max 0.55. Typical scores are 0.018 → RELATIVE-error tier-1 FAILS
(generic-pair scores are ~100% wrong); selective peaks are ~1–2 → ABSOLUTE-ε tier-1
partially stands (all scores within ±0.55 sampled, ±0.04 mean, distribution-free). The
metric decides even the epistemic tier. Our headline numbers are tier-2/3 and were
scoped as such.

Corrections 1–2 status: gauge-centering is moot for the no-softmax families (tick 0;
applies to sqrd12 only); clustering pre-rotary factors = the recommended concatenation
across frequency slices automatically; our vq is both-sided by construction (one
partition on [q̂|k̂] per head → k×k effective core); cross-associations with separate
q/k partitions + MDL-native k selection remains the spec-codebook-2 upgrade, unrun on
the real model. Adopted framing: minimal k at fixed ε = the head's SUFFICIENT PARTITION /
effective alphabet — queued as a per-head measurement.

---

## 2026-07-16 — tick 15 (effective alphabets: the sufficient-partition measurement)

Gate: PASS. `effective_alphabet.py` / `effective_alphabet.json`.

**FINDING EA-1 (marginal alphabets, ε=0.01, bilin18 layer-0):** 7 of 9 heads have
behavioral alphabet **1** (token-independent factors suffice marginally — the redundancy
again); **H3 = 2**; H6 = 4. The weight-side alphabet is unbounded (k=4096 cannot reach
25% mean factor error) — geometrically unclusterable, behaviorally near-trivial: the
weight/behavior gap in its purest form. Caveat front and center: these are MARGINAL
(single-head-patched) alphabets; joint alphabets are ~16–256 per head-branch (tick 9's
joint audits; ≤16 with CE-trained centroids since joint vq16 CE-trained = −0.019).

**FINDING EA-2 (interpretable): H3's binary distinction ≈ "am I mid-word?"** — class 0
(7,867 tokens) is almost exactly the BPE word-fragment prefixes requiring continuation
(priv/conqu/ufact/Inqu/exting/depl/cogn/Acqu/disemb...), class 1 the complete
words/suffixes/rest. The most causally-important layer-0 head is, marginally, a
morphological continuation detector — matching its near-diagonal local attention in the
pattern display (multi-token word completion).

Queue unchanged: cross-block V×V codebook, cross-associations on real model, first-order
path codebooks.

---

## 2026-07-16 — tick 16 (cross-block + self-block codebooks: MLP-0 decomposition complete)

Gate: PASS (split-path exact-exact 1.19e-7). `cross_block_codebook.py/json`,
`self_block_codebook.py/json`; results/07 updated.

**FINDING XB-1:** the cross term's two input sides are independently class-tolerant
(k_t=256 → +0.043; k_s=256 → +0.055) with superadditive compounding (both → +0.206);
self block slightly finer (256 → +0.097, 4096 → +0.030). **FINDING XB-2 (the layer-0
synthesis):** every INTERACTION (QK selection, MLP self/cross blocks) is class-tolerant
at ~256–1024 classes; the only class-intolerant object is the direct value/residual
TRANSPORT (+1.38), which sparse-codes instead. Classing source content inside the cross
term: +0.055; classing it globally: +1.38 — content precision is consumed by transport,
not by interaction. Slogan: comparisons need classes; carriage needs identity.

Queue: cross-associations (separate q/k partitions, MDL-native) on the real model;
first-order path codebooks (Tier 3); per-block CE-training of the MLP-0 codebooks.

---

## 2026-07-16 — tick 17 (MLP-0 codebooks CE-trained)

Gate: PASS. `mlp0_ce_codebooks.py/json`, tables saved (`mlp0_tables.pt`).
Combined L2-fit self@256 + cross@256×256: +0.166 (sub-additive vs +0.097/+0.206 parts).
CE-trained (3 class tables, frozen assignments, frozen model): **+0.022** — 87% recovery.
Scoreboard (CE-trained): QK −0.039 · OV −0.019 · MLP-0 blocks +0.022.
Next: the grand-combined arm — QK vq256 + OV sparse + MLP-0 classed, all simultaneous,
joint finetune → "layer 0, fully codebooked" as one number; then cross-associations and
first-order path codebooks.

---

## 2026-07-16 — tick 18 (grand-combined arm: layer 0 fully codebooked = −0.019)

Gate: PASS. `grand_combined*.py/json`, results/09_grand_combined.md.

**FINDING GC-1:** component L2 errors compound superadditively (+0.455 vs 0.230 summed).
**FINDING GC-2 (flagship):** jointly CE-training all 9.9M table values (model frozen,
frozen discrete structure, 2.1M disjoint train tokens) lands at **−0.019 — the fully
codebooked layer 0 is slightly better than the original.** QK classes + OV sparse
dictionaries + MLP classed blocks, one forward pass.
**Protocol note:** 65k train tokens sufficed for ~1M-param component tables but the
9.9M-param joint run memorized (4500 steps → train CE 1.1, held-out +1.62); 2.1M tokens
fixed it. Logged for all future joint trainings.

Queue: cross-associations on the real model; first-order path codebooks (Tier 3);
attn2-seed0 question (open).

---

## 2026-07-16 — tick 19 (cross-associations + first-order path codebooks; continuous-execution mode armed)

Gate: PASS. Cron re-armed every 30 min (:17/:47, job 3ab8af57) with the chain-next rule
baked in (Logan's instruction: never wait on the cron; never leave the GPU idle).

**FINDING CA-1 (clean negative): separate from-role/to-role partitions do NOT beat the
shared partition** on bilin18 (`cross_assoc_real.json`): shared k=256 +0.0082 vs separate
+0.0089; shared k=1024 +0.0019 vs separate +0.0047 — and separate pays double index bits.
Per head, the query-role and key-role class structures are congruent; the spec-codebook-2
upgrade is not needed on this model.

**FINDING FO-1: first-order-in-context path codebooks fix the tier-3 failure**
(`first_order_path.json`, rp model): live layer-0 pattern × classed OV content degrades
GRACEFULLY (all-content k=64/256/1024: ΔP(copy) −0.18/−0.09/−0.04) where the 0th-order
lookup collapsed (−0.62…−0.74). The missing component in tick 9 was exactly the
context-dependent pattern weights, as diagnosed.
**FINDING FO-2 (consistency effect):** classing content ONLY in the identity-branch keys
is ~3× WORSE than classing it everywhere (k=256: −0.25 vs −0.09) — partial replacement
breaks internal consistency between coupled paths; uniform coarseness composes better
than mixed precision. (Third appearance of the composition theme.)

Running: sqrd12 grand-combined analog (QK vq256 + OV sparse, tick-18 training protocol).
Queue after: results-doc consolidation; L1 first-order codebooks on bilin18; pair-block
treatment; attn2-seed0 (blocked on Logan).

---

## 2026-07-16 — tick 20 (sqrd12 grand contrast; pair block; L1 goes 0th-order)

Gate: PASS (split-path exact-exact diff 1.19e-07 in pair-block harness).

**FINDING SQ-1 (model contrast for the flagship):** the sqrd12 grand-combined analog
(QK vq256 + OV sparse 512×16, jointly CE-trained, tick-18 protocol, 2.1M tokens) lands at
**+0.188** — vs bilin18's −0.019 with MORE components codebooked (QK+OV+MLP). Two
sub-findings: (a) sqrd12's L2 errors compose SUB-additively (qk +0.116 + ov +0.221 = 0.337
summed vs +0.275 joint) where bilin18 was superadditive — the row-normalization appears to
absorb part of the joint error; (b) CE training recovers only 32% of the L2 error on
sqrd12 (0.275→0.188) vs >100% on bilin18 (0.455→−0.019). The ~15× compressibility gap
between the models is a property of the models, not of the L2 fitting stage — behavioral
training cannot close it. `sqrd12_grand.py/json`.

**FINDING PB-1 (completes the MLP-0 block table):** classing the a⊙a pair block
(self+cross exact): k=64/256/1024 → +0.073/+0.058/+0.026. At k=256 the pair block (+0.058)
sits with the cross sides (+0.043/+0.055), well below self (+0.097) — every MLP-0 block
individually tolerates ~256 classes; importance order (self > cross > pair) does not
predict class-tolerance order. `pair_block_codebook.py`, pair_block_real.json.

**FINDING L1-1 (layer-1 selection is nearly token-deterministic):** layer-1 QK factors
cannot be folded from weights (inputs are contextual), so conditional-mean factor tables
q̄(t), k̄(t) per branch (post-QK-norm pre-RoPE, estimated from 524k tokens, unit-RMS
renormalized) were patched in via the same scores_from_factors machinery: **ΔCE +0.014**
against a +2.82 zero-scores control (layer-1 attention is heavily load-bearing). The
0th-order-in-context lookup that failed for OV *content* on the tiny model (tier 3) works
for real-model *selection* — third confirmation of selection-tolerates/carriage-doesn't,
now in the context dimension. Raw (un-renormalized) cond-means cost 3× more (+0.040): the
QK-norm shell is the right gauge for the tables. Coverage 91% of audit tokens (unseen →
global mean) — +0.014 includes that fallback cost. `l1_condmean_qk.py/json`.

**L1-2:** vq256/vq1024 on the cond-mean tables: +0.092/+0.064 L2-fit — classing costs more
at L1 than at L0 (+0.008). CE-training of the vq256 class tables running (l1_ce_codebook.py,
1M table floats, protocol-sized).

Results-doc consolidation done: CA-1 → results/04, FO-1/FO-2 → results/06, SQ-1 → 09+05,
PB-1 → 07, L1 → new results/10_layer1_condmean.md; README index updated.

Queue: harvest l1_ce_codebook → tick 21; L2+ recursion (cond-mean tables at deeper layers
— does 0th-order selection hold at all depths?); MLP-0 pair CE-trained (optional); attn2-seed0
(blocked on Logan).

---

## 2026-07-16 — tick 22 (depth sweep: selection is 0th-order ONLY where the model lets it be)

**FINDING DS-1 (the depth sweep breaks the uniform story):** cond-mean selection tables
per layer (zero-scores control in parens): L1 **+0.014** (+2.82) · L2 **+0.008** (+0.55)
· L3 +0.045 (+0.25) · L5 **+0.251** (+2.51) · L8 +0.069 (+0.050) · L12 +0.016 (+0.038)
· L17 +0.033 (+0.006). Three regimes: (a) bottom layers L1-L2: heavily load-bearing AND
~token-deterministic — tables are nearly free; (b) L5: the other load-bearing attention
layer, and the ONLY one whose selection is genuinely contextual (cond-mean recovers 90%
of the zero gap but leaves +0.25); (c) upper layers (L8, L17): barely load-bearing, and
the token-generic table is WORSE THAN DELETING THE LAYER — wrong-but-confident selection
injects inconsistent signal where silence is nearly free. Fourth appearance of the
consistency theme.

**Design consequence:** the all-layers compressed model is a per-layer MENU
{table, zero, live/trained}, not a uniform treatment. Greedy from this sweep:
table L1,L2,L3,L12 · zero L8,L17 · L5 needs first-order or CE-trained tables.
Complementary sweep of the remaining 10 layers running (`layers_condmean_sweep2.py`).

`layers_condmean_sweep.py/json`.

Queue: harvest sweep2 → complete the 18-layer menu → flagship: full-model compressed
attention (menu choices + joint CE repair, protocol-sized data); KL variant of L1-3
(optional); attn2-seed0 (blocked on Logan).

---

## 2026-07-16 — tick 23 (sweep2: full 18-layer menu; stage-A composition launched)

**FINDING DS-2 (completes DS-1's table; three more surprises):** remaining layers
(zero / cond-mean): L4 +0.479/+0.059 · L6 +0.094/+0.048 · L7 +0.095/+0.018 ·
L9 +0.045/+0.006 · L10 +0.011/**−0.016** · L11 +0.033/+0.015 · L12 (tick 22) ·
L13 +0.018/+0.008 · L14 **−0.035**/+0.014 · L15 +0.002/+0.010 · L16 −0.007/−0.010.
(a) Deleting L14's attention IMPROVES pile CE by 0.035 — the layer is actively harmful
on this eval; L16-zero also mildly negative. (b) L10 and L16 cond-mean tables BEAT the
live model — token-generic selection is a regularizer there. (c) Only TWO layers in the
whole model have genuinely contextual selection worth keeping live: L5 (+0.25 gap) and
nothing else above +0.06 — L1-L4 load-bearing but tabled ~free.

Full menu (argmin per layer): table L1-4,6,7,9-13,16 · zero L8,14,15,17 · live L5.
Sum of parts +0.234 (all-table) / +0.146 (menu). Stage A running (`all_menu.py`):
composed audits of all-table / menu / menu-static, tables streamed from CPU at fp32,
saved to all17_tables.pt for the stage-B joint CE repair.

Queue: harvest stage A → tick 24; stage B = vq256 everywhere + joint CE training
(protocol: ~19M params wants ~4M tokens — check pile-10k budget; batch memory needs
checkpointing or batch 2); results/10 depth-sweep section; KL variant (optional);
attn2-seed0 (blocked on Logan).

---

## 2026-07-16 — tick 24 (stage A: composition blows up 10x; stage B training launched)

**FINDING AM-1 (composition, fifth and largest instance):** composing the per-layer menu
across the whole model (`all_menu.py/json`): menu (12 tabled + L5 live + 4 zeroed)
= **+1.440** vs +0.146 sum of parts; all-table = +1.920 vs +0.234; menu-static (L5 tabled
too) = +1.806. The mechanism is distribution shift, not table quality: each layer's
cond-mean tables were estimated under the LIVE lower stack, and patching the lower layers
destroys that distribution — errors compound multiplicatively through 17 layers. (The
single-layer sweep numbers stay valid as marginals; this is the same marginals-don't-
compose behavior as GC-1, now at model scale.)

Stage B running (`menu_trained.py`): menu-static with vq256 class tables everywhere
(13 layers x 16 head-branch codebooks = 15.7M floats, assignments frozen from the all17
cond-mean tables), jointly CE-trained 4500 steps batch 2 on ~3.1M pile tokens (protocol-
scaled). Zero layers stay zeroed; L0 stays live (exact fold). If it repairs like the
layer-0 grand did, the headline is: NO live QK selection anywhere in the 546M model —
every attention decision a token-class lookup. Held-out checkpoints at 1500/3000 to
catch overfit (tick-18 lesson).

Queue: harvest stage B → tick 25 (+ results/10 depth+menu section, figure); KL variant
(optional); attn2-seed0 (blocked on Logan).

---

## 2026-07-16 — tick 25 (docs tick; stage B mid-run)

Stage B (menu_trained.py) at step ~300/4500, ~6h ETA at batch 2 — training CE noisy
(2.1 → 6.1 on single batches), held-out checkpoints at 1500/3000 are the real signal;
monitor armed. No completed runs to harvest, so this tick shipped the queued doc work:
results/10 now has the full depth-sweep table + fig_depth_sweep.png + stage-A section;
README retitled. Layer-5-is-special is worth a targeted follow-up (what does its
selection attend to that's irreducibly contextual? induction-like?) — queued as optional
behind the flagship.

Queue: harvest stage B → tick 26 (results/10 stage-B section + root LOG for Logan);
L5 mechanism probe (optional); KL variant of L1-3 (optional); attn2-seed0 (blocked).

---

## 2026-07-17 — tick 26 (stage B mid-run; L5 probe chained)

Stage B past step 1500 (~3h remaining); training CE noisy at batch 2 (3.9-7.5 band),
held-out checkpoint pending in output. Chained behind it: `l5_probe.py` — per-head
table/live decomposition of layer 5's contextual selection (which of the 9 heads owns
the +0.25 irreducible gap; uses saved all17 tables, absolute paths after the cwd-reset
bug bit a THIRD time on background launches — rule updated: absolute paths everywhere
in chained commands).

Queue: harvest stage B + L5 probe → tick 27; KL variant (optional); attn2-seed0 (blocked).

---

## 2026-07-17 — tick 27 (heartbeat; KL variant chained third in line)

Stage B at step ~2100/4500, held-out @1500 = +0.87 (from +2.43 L2-fit — repair on
track). Chain now: stage B → l5_probe → menu_kl.py (NEW: KL(teacher||student) variant
of stage B, same 15.3M params/steps, pure imitation — its gap to the CE-trained number
is the adaptation share for the flagship claim; ~12h with teacher forwards at batch 2).

Queue: harvest stage B → tick 28; then l5_probe; then menu_kl; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 28 (heartbeat; menu_kl loss-scale bug fixed pre-run)

Stage B between checkpoints (held-out @3000 = +0.789; plateauing vs layer-0 grand —
final ~+0.7 would itself be the finding: full-stack static selection has an irreducible
joint cost that behavioral repair can't close at this budget). Fixed in the chained
menu_kl.py BEFORE it runs: F.kl_div batchmean on (B,T,V) divides by B only → ~512x
gradient inflation; reshaped to per-token rows. Chain intact:
stage B → l5_probe → menu_kl.

Queue unchanged: harvest stage B → tick 29; l5_probe; menu_kl; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 29 (heartbeat)

Stage B at step 3900/4500; nothing completed since tick 28. Chain intact:
stage B → l5_probe → menu_kl. Next tick harvests the flagship number.

---

## 2026-07-17 — tick 30 (stage B final: the static-selection wall)

**FINDING MS-1 (flagship result, and it's a WALL, not a parity):** menu-static
(vq256 class tables at all 13 non-zeroed layers, zeros at 8/14/15/17, L0 live-exact;
15.3M trainable floats, 4500 steps, 3.15M tokens, batch 2) converges at
**ΔCE +0.757** (curve: +2.43 L2-fit → +0.87 @1500 → +0.79 @3000 → +0.76 final —
plateaued, not data- or step-starved). Contrast: the SAME protocol at layer 0 alone
repaired +0.455 → −0.019. So joint behavioral training closes 69% of the composition
blowup and then hits a wall: **a 546M model with every attention selection made
token-static costs ~0.76 nats, and training the table VALUES cannot buy it back.**
What layer-0 proved possible per-layer is NOT possible for the stack: the errors that
compound through 17 layers of static selection are not repairable in the tables'
continuous degrees of freedom (frozen discrete structure). Candidate residual causes,
in testable order: (a) L5's genuinely contextual selection (marginal +0.25, and its
inputs are now themselves degraded); (b) vq256 discreteness at the wrong layers;
(c) the zeroed layers' small costs interacting. l5_probe (running) addresses (a).
`menu_trained.py/json`, codebooks in menu_cbs_trained.pt.

Chain: l5_probe running → menu_kl (adaptation share of the +0.76).
Queue: harvest l5_probe → tick 31; menu variant with L5 LIVE (menu minus the wall's
suspected main brick — cheap audit reusing menu_cbs_trained.pt, worth running before
interpreting menu_kl); results/10 stage-B section; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 30b (L5 probe harvested: TWO heads own contextual selection)

**FINDING L5-1:** layer 5's +0.251 all-tabled cost decomposes onto exactly two of nine
heads: H7 (table-alone +0.104, live-alone leaves only +0.095) and H5 (+0.080, +0.165);
every other head tables for ≤ +0.009. So in the WHOLE 546M model, irreducibly contextual
selection lives in ~2 of 162 head-instances. l5_probe.py/json.

Sequencing: paused menu_kl to run `l5_pair.py` first (15-min audit, informs everything):
arm A = L5 tabled except H5+H7 (marginal), arm B = the trained menu with H5+H7 reverted
to live (does the wall crack?), B0 sanity re-audit of the trained menu. menu_kl
re-chained behind it.

---

## 2026-07-17 — tick 30c (pair audit: hot-swap fails; menu2 retrain launched)

**FINDING L5-2 (arm A):** L5 tabled except H5+H7 live = **+0.023** (vs +0.251 fully
tabled) — the two heads carry ~91% of the layer's (and the model's) contextual
selection cost. **FINDING L5-3 (arm B, consistency effect #5):** splicing live H5+H7
into the TRAINED menu is WORSE than the wall (+1.011 vs +0.757; sanity re-audit
+0.762 ✓) — the trained tables co-adapted around a fully tabled L5; components of a
jointly-trained compressed stack are not hot-swappable. `l5_pair.py/json`.

Decision (pre-registered logic: cheap-decisive-first): menu_kl paused again; launched
`menu2_trained.py` = stage B with L5 H5+H7 LIVE from step 0 (all else identical,
15.2M trainable). If the wall is mostly those two heads' missing context, menu2 lands
FAR below +0.757 and the flagship claim becomes "every attention selection is a
token-class lookup EXCEPT TWO HEADS". menu_kl re-queues after, against whichever wall
survives.

Queue: harvest menu2 → tick 31 (+ results/10 stage-B/L5 sections); menu_kl vs final
wall; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 31 (heartbeat; menu2-KL control written and chained)

menu2 in early training (step <300 of 4500, ~6h). Wrote `menu2_kl.py` — KL(imitation)
control on the menu2 config (supersedes menu_kl.py, which targeted the old static
config) — and chained it behind menu2. The plain menu_kl.py stays on disk unused unless
menu2 surprises high.

Queue: harvest menu2 → tick 32 (flagship rewrite + results/10); menu2_kl (chained);
attn2-seed0 (blocked).

---

## 2026-07-17 — tick 32 (heartbeat)

menu2 at step 900/4500; nothing completed. Chain: menu2 → menu2_kl. Monitor armed on
held-out checkpoints. Next harvest: @1500 checkpoint.

---

## 2026-07-17 — tick 33 (menu2 @1500 = +0.68; iterated re-estimation designed and chained ahead of KL)

menu2 held-out @1500 = **+0.681** (static run was +0.87 at the same step, final +0.757;
projection: menu2 final ~+0.58). Early read: the two live heads buy ~0.15-0.2, NOT the
bulk of the wall — residual suspects are distribution shift of the estimators vs vq
discreteness.

New experiment (chained ahead of the KL control because it's 4x cheaper and decides the
mechanism): `iter_reestimate.py` — bottom-up NO-TRAINING pass: re-estimate each tabled
layer's cond-mean tables UNDER the already-patched lower stack (menu2 config), audit
cumulatively after each layer. If composed dCE falls from ~+1.8 (one-shot untrained
tables) toward sum-of-parts ~+0.2, the wall is distribution shift and the fix is
estimation procedure, not capacity. Progressive curve localizes where compounding bites.

Chain: menu2 (running) → iter_reestimate → menu2_kl.
Queue: harvest menu2 final → tick 34; then iter curve; then KL; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 34 (heartbeat)

menu2 @3000 = +0.566 (vs static +0.789 at same step), final in ~1h. Chain intact:
menu2 → iter_reestimate → menu2_kl. Next tick: harvest menu2 final + write the
results/10 wall section with the complete menu-static/menu2/iter triple.

---

## 2026-07-17 — tick 35 (menu2 final: two heads buy 0.23; the wall persists at +0.53)

**FINDING MS-2:** menu2 (identical to menu-static but L5 H5+H7 live from step 0)
converges at **+0.530** vs the static wall's +0.757 (curve +1.97 L2-fit → +0.68 → +0.57
→ +0.53). So the model's two genuinely contextual heads account for ~0.23 of the wall —
consistent with their marginal (+0.25 as a full L5 gap, ~91% theirs) FINALLY composing
additively once they're present during training (unlike the hot-swap, L5-3). The
remaining +0.53 is NOT owned by any single layer's selection (every marginal was
≤ +0.07 outside L5): it is either compounding estimator bias (distribution shift) or
distributed vq discreteness. iter_reestimate (auto-started, chained) decides:
no-training bottom-up re-estimation; watch the cumulative curve. menu2_trained.json,
menu2_cbs_trained.pt.

Chain: iter_reestimate (running) → menu2_kl.
Queue: harvest iter curve → tick 36 (+ results/10 full wall section incl. MS-2);
menu2_kl; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 36 (iter curve 8/13 in: re-estimation does NOT fix the wall)

Interim IR-1 (final next tick): cumulative no-training re-estimation curve
L1 +0.12 → L2 +0.16 → L3 +0.37 → L4 +0.61 → L5 +0.73 → L6 +0.81 → L7 +0.96 →
L9 +1.01, already past BOTH trained walls (+0.757 static, +0.530 menu2) at 8 of 13
layers. Distribution shift is refuted as the wall's main mechanism: fresh estimators
under the degraded stack don't contain compounding — the increments concentrate in
L3–L6 (bottom-stack selection consumes context that only becomes visible when its
inputs are also tabled). The wall is genuine contextual information plus what only
JOINT training can co-adapt away. zeros_control.py chained (composed floor of the
4 zeroed layers, needed to read the curve; marginal sum +0.023).

Chain: iter (5 layers left) → zeros_control + menu2_kl.
Queue: harvest iter final + zeros → tick 37 = full wall write-up in results/10
(one table: sum-of-parts / one-shot / re-estimated / trained-static / trained-menu2,
plus curve figure); menu2_kl; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 37 (LOGAN REDIRECT: methods A–E for the composition blowup)

Logan's message maps the deep-layer input space as {embedding, attn-out_l, mlp-out_l}
streams and proposes: A conditional interaction clustering (condition on current token
→ co-occurrence collapses the joint space); B interaction-norm screening; C empirical
interaction-depth window (layer N barely affects N+2 directly → shifting window);
D sparse-code propagation from early layers (partially built: first_order_path on rp);
E MDL relative to the UNEMBEDDING, backwards (different optimum than embedding-relative).

Execution order chosen (cheap+informative first, each is a queue item):
  B stream_interactions.py — WRITTEN + CHAINED (exact stream decomposition of every
    layer's branch scores over stream pairs, 2 gates, energy map + window summary);
  C window interventions guided by B's map;
  D first-order propagated-code QK inputs on bilin18 (live patterns × classed content
    feeding later layers' QK — the mechanistic wall-fix candidate);
  E gradient/Fisher-weighted vq as drop-in for menu2 clustering (training run);
  A conditional tables (t_i, context-class) where B/C say conditioning suffices.
menu2_kl DEFERRED (chain canceled) — adaptation-share control postponed in favor of
the redirect; iter_reestimate + zeros_control still finishing (wall write-up pending).

Framing correction to log: the blowup is NOT "too many inputs to enumerate at L1+" —
per-layer cond-mean tables are near-free almost everywhere (DS-1/2). It's composition:
per-layer tables fail JOINTLY (+1.44), joint training walls at +0.53 (menu2), and
iter re-estimation (no training) proves the wall is genuine contextual information +
co-adaptation, NOT stale estimators (IR-1, final numbers next tick).

Queue: harvest iter final + zeros → wall write-up; B map → C design; D prototype
(L1-2 first); E weighted-vq menu3; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 37b (IR-1 final + Z-1: even deletions don't compose)

**FINDING IR-1 (final):** no-training bottom-up re-estimation, full cumulative curve:
L1 +0.12 → L2 +0.16 → L3 +0.37 → L4 +0.61 → L5 +0.73 → L6 +0.81 → L7 +0.96 → L9 +1.01
→ L10 +1.07 → L11 +1.11 → L12 +1.15 → L13 +1.25 → L16 **+1.41**. Better than one-shot
(+1.8-scale) but far above the trained walls (+0.757/+0.530). Distribution shift
REFUTED as the wall's mechanism; compounding concentrates in L3–L6. What joint training
buys (+1.41 → +0.53) is co-adaptation, not statistics. iter_reestimate.json/iter_tables.pt.

**FINDING Z-1 (zeros control):** the four "free-deletion" layers (8,14,15,17) composed
= **+0.114** vs +0.023 marginal sum — 5× superadditive even for deletions. Two
consequences: (a) the iter curve's first point (+0.123) is almost entirely the zeros
floor — L1's table adds ~+0.01, matching its marginal; (b) menu2's +0.53 sits on a
+0.114 floor from the zeros themselves → tabled selection proper costs ~+0.42.
Queue item added: menu3 = all-table, NO zeros, H5/H7 live (the zeros floor may be
buyable back). zeros_control.py/json.

Running: stream_interactions.py (Logan's method B, auto-chained).
Queue: B map → C window design; D propagated codes; E backward/weighted vq; menu3;
results/10 wall table+figure; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 38 (SI-1: the interaction map — short window below, diffuse+hub above)

**FINDING SI-1 (Logan's method B):** exact stream decomposition of every layer's branch
scores (gates: stream-sum ≡ x; pair-sum ≡ live score; max dev 2.0e-2 from bf16 rope —
fine for a norm map). Three regimes: (a) L1–L6: selection reads a SHORT WINDOW —
mlp(L−1)×mlp(L−1) dominates (L2: 87% of energy; L5: recent×recent = 89%), emb×emb ≈ 0
above L1 (the embedding's selection role is entirely mediated by MLP-0 — explains why
L0 folding is exact but L1+ tables must be data-estimated); (b) L8–L15: DIFFUSE — top
pair only 2–4%, long-range interactions everywhere, with attn5's output a persistent
HUB stream through the whole upper model (the contextual layer's output is globally
load-bearing); (c) L16–17: re-concentrates on mlp(L−1). Logan's window hypothesis (C)
holds in the bottom stack, breaks in the middle. stream_interactions.py/.pt/.json.

Launched: c_window.py (method C interventional): at L∈{2,5,9}, patch ONLY the QK read
(v + residual live): (i) mlp(L−1) stream → cond-mean table; (ii) all streams older than
L−2 tabled (window-only live); (iii) all streams tabled (0th-order QK read).

Queue: harvest c_window → D design (propagate codes through the window; hub stream
attn5 needs its own treatment); E weighted-vq; menu3 (no zeros); wall write-up
w/ figures; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 38b (C-1 harvested: the window is REAL; composed windowed-D launched)

**FINDING C-1 (method C interventional, v/residual live, QK read patched only):**
L2: mlp1-tabled +0.002 · window-live +0.000 · ALL-tabled +0.009 (≈ depth-sweep +0.008 ✓)
L5: mlp4-tabled +0.047 · window-live **+0.003** · ALL-tabled +0.231 (≈ sweep +0.251 ✓)
L9: everything ≤ +0.007 including ALL-tabled.
The bombshell is L5: its "irreducibly contextual" selection (the +0.25 gap, the H5/H7
heads, the wall's named suspect) needs only the LAST TWO LAYERS' streams live — deep
context tables away for +0.003. Combined with SI-1: selection everywhere reads (old
context ≈ token identity) × (recent window ≈ live computation).

Launched `d_composed.py` — Logan's D in window form, composed across ALL layers at
once, NO training: every layer's QK read = exact emb stream + tabled old streams
(created > W layers back; cond-means estimated at creation, λ-rescaled analytically)
+ the patched model's OWN live recent streams. Error chains bounded at depth W.
Arms: W=2, W=3, W=1, W=0 (composed control — should reproduce wall-scale blowup).
If W=2 composes near the sum of C-1 marginals (~+0.05 total), the wall is CRACKED
without training and the flagship architecture is: token-static long-range context,
live short-range computation.

Queue: harvest d_composed → tick 39 (wall write-up + this arc, results/10-11);
if W=2 works: MDL accounting for stream tables + vq/sparse compression of them (they
are (V,D) fp32 objects — the actual bits); E weighted-vq; menu3; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 39 (D-1: THE WALL IS CRACKED — windowed code propagation, no training)

**FINDING D-1 (flagship-grade):** composed windowed-D across all 17 layers, ZERO
training: W=0 control +2.27 (reproduces wall-scale ✓) · W=1 +0.86 · W=2 +0.43 ·
**W=3 +0.225** — the untrained windowed architecture beats BOTH trained walls
(static +0.757, menu2 +0.530). Selection's long-range context is token-static;
only a 3-layer local window of live computation is needed. Error chains bounded at
depth W decay ~2× per +1 of W. Logan's methods B→C→D executed in sequence produced
in one day what score-space tabling + 15M trained params could not.

Caveat for MDL: the stream tables are raw (V,D) objects (34 × 51M floats) — the bits
live there. Running now (`d_composed2.py`): W=4/5/6 asymptote + vq256/vq1024
compression of the stream tables at W=3/4 (tables also saved to stream_tables.pt).

Queue: harvest → tick 40: full wall-arc write-up (results/11: SI-1, C-1, D-1, wall
table incl. iter/zeros baselines, figures); then CE-polish of vq'd stream tables if
needed; E weighted-vq now optional (D route dominates); menu3 obsolete unless vq
tables disappoint; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 40 (D-2: asymptote + free vq; results/11 shipped; data-scaling control running)

**FINDING D-2:** W asymptote (untrained): W=4 +0.099 · W=5 +0.064 · W=6 +0.050 — cost
~halves per window step. **vq1024 on the stream tables is FREE** (W=4: +0.094, slightly
better than raw — quantization denoises the cond-means; W=3: +0.210); vq256 costs +0.04.
~50× table compression. Headline: **W=4 + vq1024 = +0.094, zero trained parameters**
(vs +0.757 for 15.3M trained score-table floats). d_composed2.json, stream_tables.pt.

Shipped results/11_windowed_codes.md (the full wall arc: MS → IR/Z → SI → C → D, with
fig_wall.png) — marked current flagship in the README.

Running: d_composed3.py — 6× estimation data (3.15M tokens), W=3/4: decides whether the
residual +0.09 is table noise or window-boundary error.

Queue: harvest d3 → CE-polish decision; root LOG update for Logan; E weighted-vq
(optional now); attn2-seed0 (blocked).

---

## 2026-07-17 — tick 40b (D-3: more data HURTS — region confound; controls running)

**FINDING D-3 (important caveat on the flagship):** 6× estimation data made D-composed
WORSE: W=4 +0.166 (was +0.099), W=3 +0.386 (was +0.225). Not sampling noise — the extra
5k chunks come from a different region of pile-10k than the early audit slice, so the
cond-means drifted off the audit distribution. Two implications: (a) the residual +0.09
is not estimator variance; (b) the stream tables are DISTRIBUTION-TUNED objects — the
+0.094 flagship number may be partly local to the audit's document region. Controls
running (`d_composed4.py`): A = early tables audited on LATE chunks (cross-region
generalization of the headline); B = late-estimated same-size tables on the early audit
(region-match vs amount). QUESTION FOR LOGAN (accounting): where should the
estimation-data/distribution term sit in the MDL story for data-estimated tables —
count estimation tokens as description bits, or report per-region numbers?

Queue: harvest d4 → honest revision of results/11 §5; root LOG for Logan; CE-polish
decision after; attn2-seed0 (blocked).

---

## 2026-07-17 — tick 41 (D-4: flagship generalizes cross-region; CE-polish launched)

**FINDING D-4 (controls resolve D-3 favorably):** A = early tables on LATE audit:
**+0.089** (headline generalizes; late-audit baseline 2.863); B = late-estimated
same-size tables on early audit: +0.184. Asymmetric: the early pile-10k slice is more
DIVERSE, so its tables are better everywhere; D-3's worsening was estimation-data
homogeneity, not region overfitting. Flagship stands: W=4 + vq1024 ≈ +0.09 on both
regions, untrained. Table quality tracks estimation diversity, not amount.
d_composed4.json; results/11 §5 revised.

Launched d_polish.py: CE-polish of the vq1024 ATOMS for bottom streams (attn/mlp 0–5,
12.6M floats, 3000 steps batch 2 on the diverse early 524k; held-out @1000/2000 +
late-region audit at the end). Residual +0.09 is structural (window boundary) — polish
tests how much of it the continuous DOF can buy back.

Queue: harvest polish → root LOG for Logan + results/11 final table; E weighted-vq
(optional); attn2-seed0 (blocked). QUESTION FOR LOGAN pending: MDL estimation-data term.

---

## 2026-07-17 — tick 41b (D-5: polish buys NOTHING — structure was already right; all-reads launched)

**FINDING D-5 (clean negative, closes the selection arc):** CE-polishing the flagship's
bottom-stream vq atoms (14.2M trainable, held-out checkpoints flat: +0.0942 → +0.0939 →
+0.0934 → final +0.0929; late audit +0.096) buys back essentially nothing. Sharp
contrast with score-space tables, which training improved 3× (+2.43 → +0.76). Reading:
the windowed-input architecture puts the discrete structure where the model's actual
computation lives, leaving nothing for continuous repair — the W-ladder (W=6 = +0.05)
is the only remaining knob, and it's a clean tradeoff curve, not a training problem.
d_polish.json.

Launched d_allreads.py: window ALL residual reads (v/content and MLP inputs, not just
QK) with the same tables — arms {v}, {qk,v}, {qk,v,mlp} × W∈{4,6}. If it composes, the
model's entire long-range information flow is token-static.

Queue: harvest all-reads → final arc write-up + root LOG for Logan + MDL bits table;
attn2-seed0 (blocked); QUESTION FOR LOGAN: estimation-data MDL term.

---

## 2026-07-17 — tick 42 (D-6/D-7: carriage windows for free and composes ADDITIVELY; MLP reads are the contextual core)

**FINDING D-6:** windowed v-reads (content/carriage): W=4 +0.019, W=6 +0.004 — nearly
free, MORE static than selection. And qk+v composes ADDITIVELY (W=4: +0.112 ≈ .094+.019;
W=6: +0.052 ≈ .050+.004) — the FIRST additive composition in the program. Resolution of
the old carriage-needs-identity theme: carriage needs token IDENTITY, which cond-mean
tables preserve exactly; it never needed context.

**FINDING D-7:** MLP-read windowing breaks it: qk+v+mlp W=4 = +0.864, W=6 = +0.325.
The bilinear MLP input is where long-range context genuinely enters the computation.
Localizer running (d_mlpread_probe.py): single-layer marginals {2,5,9,13,16} + bottom/top
composed — local fidelity vs knock-on (mlp_out is the next layer's dominant QK input).

Current best full result: **qk+v windowed at W=6 = +0.052 untrained** — all long-range
attention I/O (selection and content) in the 546M model is token-static.

Queue: harvest localizer → arc write-up (results/11 + root LOG + MDL bits table);
attn2-seed0 (blocked); Q-LOGAN: estimation-data MDL term.

---

## 2026-07-17 — tick 42b (D-8: MLP contextuality is TOP-of-model; final-arch arms running)

**FINDING D-8:** MLP-read windowing marginals: L2/L5/L9 ≈ 0.000 · L13 +0.005 ·
L16 **+0.146**; composed bottom L1-6 = +0.0004 (free) vs top L7-17 = +0.593. MLP
contextuality lives at the top of the model — the mirror image of selection (bottom-
heavy, L5). Coherent with SI-1: L17's selection reads mlp16×mlp16 — the upper MLPs
assemble genuinely contextual prediction features. The "irreducibly live" core of the
546M model is now: a ~W-layer local window everywhere + upper-MLP reads (L13-17).

Running (d_final_arch.py): qk+v all + mlp L1-12 at W=4/6, + mlp L1-15 at W=6 — the
final architecture numbers.

Queue: harvest → full arc write-up (results/11 §6-7, root LOG, MDL bits table);
attn2-seed0 (blocked); Q-LOGAN: estimation-data MDL term.

---

## 2026-07-17 — tick 43b (D-9: THE INVERSION — sqrd12 is EASIER under windowed-D)

**FINDING D-9 (transfer + inversion):** windowed-D QK-reads on sqrd12: W=6 **+0.011** ·
W=4 +0.040 · W=2 +0.204 · W=0 control +1.48. The architecture transfers, and INVERTS
the compressibility ranking: sqrd12 was ~15× HARDER than bilin18 under score-space
tables (results/05, 09) and is ~2× EASIER under input-space windowing (W=4: +0.040 vs
+0.099). Compressibility is a property of the (model, decomposition-family) PAIR, not
of the model — the strongest instance yet of the program's metric/representation-decides
theme. d_sqrd12.json, stream_tables_sqrd12.pt.

Running: d_sqrd12b.py (qk+v, +mlp arms — full final-arch transfer).
Queue: harvest → results/11 §8 transfer + memory update; attn2-seed0 (blocked);
Q-LOGAN: estimation-data MDL term.

---

## 2026-07-17 — tick 43c (D-10: sqrd12's ENTIRE long-range flow is token-static; +0.030)

**FINDING D-10:** sqrd12 full final-arch: qk+v W=6 +0.013 · +bottom-MLP identical ·
+ALL-MLP W=6 **+0.030** (bilin18 all-MLP was +0.325 — sqrd12's ReLU² MLPs barely
consume long-range context). Combined with D-9: the compressibility ranking between
the two models INVERTS with the decomposition family (score-space: sqrd12 15× harder;
input-space windowing: sqrd12 ~2-10× easier). results/11 §8 written. Ladder arms
(sqrd12 all-reads W=2/3/4) running as filler. d_sqrd12b.json.

Queue: harvest ladder → memory update + final commit sweep; attn2-seed0 (blocked);
Q-LOGAN: estimation-data MDL term.

---

## 2026-07-17 — tick 44 (arc closed; queue exhausted pending Logan)

Ladder final: sqrd12 all-reads W=2/3/4/6 = +0.96/+0.41/+0.18/+0.030 (d_sqrd12c.json,
added to results/11 §8 context). The windowed-D arc (D-1…D-10) is complete and fully
written up in results/11 + root LOG. Memory updated. GPU idle by design: remaining
queue items are blocked on Logan — (1) estimation-data MDL term convention;
(2) attn2-seed0 checkpoint location; (3) direction of the next arc (candidates:
within-window interpretability — name the live computations the window protects;
E backward-MDL variant; softmax-model transfer to a standard transformer).

---

## 2026-07-17 — tick 45 (WW-1: the contextual heads have names — H5 is induction, H7 is local-content)

**FINDING WW-1 (within-window arc, probe 1):** L5.H5 has the classic induction
signature — conditional pattern on "key follows my previous occurrence" is **16.8×**
its unconditional mean (copy 3.5×), with a nearly FLAT positional profile (2.4× decay
Δ1→64, vs 12-37× for the free heads) — a long-range content-matching head, exactly the
computation a token-static table cannot express. L5.H7 shows NO match signature
(1.2-1.35×) but a fat local profile (high through Δ≈4-8): content-dependent LOCAL
selection, syntax-like. So bilin18's irreducibly contextual selection = one induction
head + one local-content head. Stats caveat (logged): the over-random ratios for
H1/H2/H4/H6 are meaningless (signed means ≈ 0 in the denominator); conditional means in
the json are the honest numbers. l5_heads_function.py/json.

Running: l5_h5_causal.py — repeated-sequence (A+A) causal test: zeroing/tabling H5
should specifically hurt 2nd-half CE on repeats if it's the induction mechanism (H7 as
the non-induction contrast).

Queue: harvest causal → WW write-up (results/12); then H7 mechanism, window-content
naming (what do the recent-stream reads compute?); blocked-on-Logan items unchanged.

---

## 2026-07-17 — tick 45b (WW-2: causal test FLIPS it — H7 is the repeat-copier)

**FINDING WW-2:** repeated-sequence (A+A) causal test: zeroing H7 = **+6.68** on
2nd-half CE (+1.04 natural — L5's true heavy lifter); zeroing H5 = only +0.13 (+0.03
natural) despite its 16.8× correlational induction signature. The correlational/causal
dissociation strikes again (cf. the conjunction test's generic-vs-conditioned gap):
H5 *attends* induction-like on natural text but carries little; H7's natural-text
pattern looks local, yet it is THE causal mechanism for repeat copying — its pattern
must reorganize in repeat contexts (patterns are context-dependent; that is precisely
why these two heads resist tabling). Also noted: baseline repeat-2nd-half CE is 5.48
vs 3.23 natural — bilin18 is weak at literal copying overall. l5_h5_causal.py/json.

Running: signature probe re-run ON repeat data (l5_heads_function_rep.py) — do H7's
match statistics appear once the context demands them?

Queue: harvest → results/12 (within-window arc); H7 mechanism deep-dive; Logan items.

---

## 2026-07-17 — tick 45c (WW-3 synthesis; results/12 shipped)

**FINDING WW-3:** repeat-data signatures resolve the flip: H5's induction signature
INTENSIFIES on repeats (53× random) while H7 shows none in any context — H5 selects by
matching, H7 transports locally-chosen content into the attn5 hub stream the whole
upper model reads (SI-1). The selection/carriage split reappears INSIDE the contextual
core, and the correlational/causal dissociation repeats the conjunction test's lesson.
results/12_within_window.md shipped; README indexed.

Queue: H7 OV-side decomposition (what content features feed the hub?); Logan items
(estimation-data MDL term; attn2-seed0; next-arc priority).

---

## 2026-07-17 — tick 46 (H7 OV probe launched)

GPU was idle post-WW-3; launched h7_ov_probe.py (queue head): cond-mean v content per
source token at L5 for heads {7,5,0}, mapped through each head's c_proj slice,
logit-lens + embedding-alignment + decoded examples — names what the transport head
forwards into the hub (copy-content vs transformed). Crude-lens caveat logged in the
script docstring.

Queue: harvest probe → results/12 addendum; Logan items (estimation-data MDL term,
attn2-seed0, next-arc priority).

---

## 2026-07-17 — tick 46b (WW-4: H5 carries identity; H7's payload is contextual deviation)

**FINDING WW-4:** OV logit-lens on cond-mean content: H5's forwarded content decodes to
its SOURCE TOKEN (median rank 25/50k, top-1 15% — identity carriage; H5 is the complete
textbook induction head). H7's per-token mean is a near-constant generic direction
(decodes to '-'/' and'/'(' for every source; rank 4072) — its causal power (+6.7 on
repeats) lives in context deviations invisible to any 0th-order object. The pair now
fully explains itself: H5 = matchable+tableable-in-content but contextual-in-selection;
H7 = contextual in BOTH, mean content nearly empty. results/12 updated (WW-4 table +
examples). h7_ov_probe.py/json.

Running: h7_deviation_pca.py — PCA of H7's output deviations around token means,
lensed top directions + variance shares (H5 contrast).

Queue: harvest PCA → results/12 final; Logan items unchanged.

---

## 2026-07-17 — tick 46c (WW-5: H7 is a rank-1 gain head; causal rank-k test running)

**FINDING WW-5:** deviation-PCA around token-conditional means: H7's deviations are
(a) only ~5% of its output energy and (b) 63% ONE direction — which lenses to the same
generic connective/structure feature as its mean. H7 ≈ (fixed hub direction) ×
(context-dependent scalar gain). The model's un-tableable local transport reduces to
ONE SCALAR per position. H5 contrast: 92% deviation share spread over many directions
(diverse identity content — as a match-and-copy head must be). h7_deviation_pca.py/json.

Running: h7_rank1.py — causal: replace H7-out by its rank-k projection (mean + top-k
PCs, live coefficients), k∈{1,2,4,8}; H5 same as contrast (expect rank-1 hurts).
If H7 rank-1 is ~free on natural AND repeat audits, the WW arc closes with: bilin18's
contextual core = one match head + one scalar gain + top MLPs.

Queue: harvest rank test → results/12 final + arc close; Logan items unchanged.

---

## 2026-07-17 — tick 47 (WW-6: H7 causally rank-1; within-window arc CLOSED)

**FINDING WW-6:** live-coefficient rank-k replacement: H7 rank-1(+mean) = **+0.0001
natural** (repeat +0.049, vs +6.68 zero-ablation — 99% of function through a rank-1
bottleneck). H5 rank-1 hurts natural (+0.073) as predicted; SURPRISE: low-rank H5
content IMPROVES repeats (rank-2: −0.33) — the model under-cashes its own induction
signal; denoising the carried identity strengthens it. (H7 rank-4 repeat +0.20
non-monotonic — small-sample PCs; k=1 is the finding.) results/12 closed with the arc
conclusion: bilin18's irreducibly contextual attention = one match head (H5, identity
payload) + one scalar gain (H7, structure feature). h7_rank1.py/json.

Arc status: windowed-D (results/11) + within-window (results/12) both complete.
Queue: ALL remaining items blocked on Logan — (1) estimation-data MDL term convention;
(2) attn2-seed0 location; (3) next-arc priority (top-MLP mechanism / E backward-MDL /
softmax-transformer transfer / H5-denoising follow-up now added as a candidate).
GPU idle pending Logan's direction — the natural next default if no answer arrives:
top-MLP mechanism (the last unnamed contextual component).

---

## 2026-07-17 — tick 48 (top-MLP arc opened: exact stream-pair map for bilinear MLPs)

Logan still silent on next-arc priority → default arc launched: NAME the top-MLP
contextual computation (last unnamed live component; windowed MLP reads at L13-17 cost
+0.59 composed, L16 marginal +0.146). Key structural fact verified: bilin18's MLP is
PURE bilinear (Down(Lx ⊙ Rx), gated=False) — so MLP outputs decompose EXACTLY over
stream pairs, same machinery as the QK map (SI-1). Launched
mlp_stream_interactions.py: Down-weighted per-pair hidden energy at L∈{2,5,13,16,17}
(bottom layers as contrast), with the two exactness gates.

Queue: harvest map → targeted probes (deviation-PCA of top-MLP outputs? gain-like?);
Logan items unchanged.

---

## 2026-07-17 — tick 48b (TM-1: top-MLP input is DIFFUSE; output-rank probe running)

**FINDING TM-1:** exact bilinear stream-pair map for MLPs (gates pass): bottom MLPs
read a tight window (L2 recent×recent 99%, L5 94% — why windowed MLP reads were free
below); **L13 is diffuse (recent 19%, top pair 3%)**, L16 44%, L17 65%, with the attn5
hub reappearing in L16/17 pairs. The top-MLP contextual computation is broad
aggregation over many old streams — no single nameable channel, unlike selection.
mlp_stream_interactions.py/json.

Running: mlp16_rank.py (H7 playbook on outputs): token-mean + rank-k deviation
projection (live coefficients) for mlp16 and mlp13 — is the OUTPUT low-rank even
though the input consumption is diffuse?

Queue: harvest → TM write-up (results/13); Logan items unchanged.

---

## 2026-07-17 — tick 49 (TM-2: mlp16 factors through ~4-16 live scalars; results/13 shipped)

**FINDING TM-2:** mlp16's contextual output is LOW-RANK (dev PC shares 40/17/8%):
token-mean + rank-4 live projection = +0.040 (vs mean-only +0.141), rank-16 +0.024.
mlp13: individually cheap (+0.041 mean-only) and genuinely diffuse (PC1 4%) — the
composed top-MLP damage is interaction compounding. Synthesis table in results/13:
EVERYTHING contextual in bilin18 is token identity (H5 payload) or a small number of
live scalar gains on fixed directions (H7: 1; mlp16: ~4-16). Structural claim, not
compute reduction (live coefficients) — caveat in the doc. results/13 shipped, README
indexed.

Queue: name the mlp16 directions (lens+examples — next default); Logan items
unchanged (estimation-data MDL term; attn2-seed0; arc priority).

---

## 2026-07-17 — tick 49b (mlp16 direction-naming probe launched)

Launched mlp16_dirs.py (standalone — caught the import-runs-module trap before it cost
a rerun this time): top-8 deviation PCs of mlp16, each with logit-lens ± tokens and
extreme-coefficient context examples. Names the ~4-16 contextual gains of TM-2.

Queue: harvest → results/13 addendum; Logan items unchanged.

---

## 2026-07-17 — tick 50 (TM-3: the top-MLP context is DOCUMENT REGISTER; arc closed)

**FINDING TM-3:** mlp16's contextual directions decode cleanly: dir0 legal-citation
register (40%), dir1 prose (18%), dir2 legal captions (8%), dir3 XML/markup (5%),
dir4-7 blog/technical/numeric. The dominant top-MLP's irreducible context = document
register — slow document-scale state, exactly what token tables and local windows
cannot carry. Sample caveat logged (pile-10k early slice legal-heavy; structure is the
finding, not the ordering). results/13 complete — top-MLP arc CLOSED.

**Program state: all three arcs complete** (windowed-D results/11; within-window
results/12; top-MLP results/13). The 546M model's full contextual inventory:
token-static tables + local window + one induction match (H5) + one structure gain
(H7, rank-1) + a handful of register gains (mlp16, rank ~4-16) + diffuse small top-MLP
interactions. Queue: ALL blocked on Logan — (1) estimation-data MDL term; (2)
attn2-seed0; (3) next direction (candidates: register-swap causal test; softmax-model
transfer; E backward-MDL; H5-denoising follow-up).

---

## 2026-07-18 — tick 51 (TM-3 causal validation launched)

All arcs closed; Logan still silent → launched the register-swap causal test
(mlp16_register_swap.py), the cheapest decisive follow-up: within a rank-64 live
reconstruction of mlp16 deviations, patch ONLY the top-4 (register) coefficients:
(a) document-mean (slowness test — if register is document-level state, ~free);
(b) swapped across documents (wrong register — should hurt); (c) zeroed (reference).
Confirms or breaks the TM-3 interpretation.

Queue: harvest → results/13 addendum; then remaining candidates (softmax transfer,
backward-MDL, H5 under-cashing) pending Logan's priority; blocked items unchanged.

---

## 2026-07-18 — tick 52 (TM-4: slowness REFUTED — the "register" gains are fast local structure)

**FINDING TM-4 (negative, revises TM-3):** top-4 coefficient patching inside rank-64
live reconstruction: live +0.023 · doc-mean +0.103 · swapped +0.158 · zeroed +0.113.
Document-constant coefficients ≈ as bad as deletion → mlp16's dominant contextual
gains are NOT slow register state; they are fast-varying structural position within
register-specific syntax (the lens/examples showed the contexts, not the content).
results/13 corrected in place — TM-3's register naming demoted to correlational; the
structural claim (few live gains on fixed directions) stands. mlp16_register_swap.py/json.

Program remains at rest pending Logan (MDL estimation term; attn2-seed0; next
direction). Next default if silence continues: softmax-transformer transfer of
windowed-D (tests architecture-generality of the whole framework).

---

## 2026-07-18 — tick 52b (H5 under-cashing probe launched)

Launched h5_undercash.py: hypothesis A (content noise — H5's v replaced by clean
cond-mean identity content, live pattern) vs hypothesis B (amplitude starvation —
H5's pattern scaled α ∈ {1.5, 2, 4}), plus the A+B combination; natural + repeat
audits. Explains WHY low-rank filtering of the induction head's output IMPROVED
repeat prediction (WW-6).

Queue: harvest → results/12 addendum; softmax-transfer awaits Logan's scope OK
(new model family); other Logan items unchanged.

---

## 2026-07-18 — tick 53 (WW-7: under-cashing = content noise; program at rest)

**FINDING WW-7:** hypothesis A wins decisively — H5 v-content cleaned to cond-mean
identity: repeat −0.170 (natural +0.023); pattern scaling hurts monotonically
(α=4: +3.37) and adds nothing after cleaning. The induction head carries a NOISY copy
of token identity; the model's small gain is rational; denoising is the intervention
that helps. results/12 postscript added. h5_undercash.py/json.

**Program state: fully at rest.** All arcs closed (results/11-13), all follow-up
probes done (TM-4 negative, WW-7 resolved). The unblocked queue is EMPTY — remaining
items all need Logan: (1) MDL estimation-data term convention; (2) attn2-seed0
location; (3) next direction (softmax-transformer transfer needs a scope OK — new
model family; register/structural deep-dive; backward-MDL). Idling deliberately
rather than launching unqueued speculation (spec §6: no silent scope drift).

---

## 2026-07-18 — tick 54 (docs tick: window-ladder figure + README headlines)

No new Logan input; queue still blocked. Shipped fig_window_ladder.png (all W-ladders,
both models, trained walls as reference lines, log scale) into results/11, and a
headline-numbers block into results/README. GPU idle by design (empty unblocked
queue — rationale in tick 53).

---

## 2026-07-18 — tick 55 (gate-verified rest state)

tier0_gate.py re-run on the resting commit: ALL PASS (pattern ~3e-15, gauge ~2e-15 —
unchanged). Repo clean, all results committed. No Logan input; unblocked queue still
empty. Convention going forward: silent no-op ticks will not add LOG entries or
commits — the next entry will be a harvest, a Logan response, or a queue change.

---

## 2026-07-18 — tick 55b (cron retuned for rest state)

Rest-state housekeeping: 30-min cron (3ab8af57) replaced by hourly (96461de0, :23,
7-day expiry, session-scoped) with rest-state semantics baked into the prompt: silent
no-op while the unblocked queue is empty; auto-restore 30-min cadence the moment Logan
replies or the queue gains unblocked items. Continuous-execution guarantee intact,
no-op churn eliminated.

---

## 2026-07-18 — tick 57 (E-1 negative: diagonal Fisher loses to L2; E2 unembedding-metric chained)

**FINDING E-1 (negative for the first instantiation):** Fisher-whitened assignments
lose to plain L2 at both k (vq64: +0.171 vs +0.139; vq256: +0.116 vs +0.104) in the
W=4 composed harness. Marginal gradient-whitening distorts the cluster geometry more
than it helps — sampling noise (96 seqs) and diagonal-only structure are the suspects.
stream_fisher.pt saved for reuse. e1_backward_vq.py/json.

Launched E2 (`e2_unembed_vq.py`): the LITERAL unembedding-relative metric — cluster
table rows by their logit-space image (JL-sketched M = P·U, quadratic form M'M),
deterministic, same harness/k. If Logan's conjecture holds anywhere cheap, it's here.

Queue: harvest E2 → E arc synthesis (results/14); if E2 also loses, log the honest
conclusion (forward L2 on activations is already the right metric for these tables —
consistent with vq1024-is-free being a denoising story) and consult Logan for the
next E instantiation (CE-refined assignments?) or close the arc.

---

## 2026-07-18 — tick 57b (E-2: crossover — unembed metric WINS at k=64; stability check chained)

**FINDING E-2 (first positive for Logan's conjecture, with caveat):** unembedding-
relative clustering (rows compared through M = JL(512)·U) beats L2 at the aggressive
budget — vq64: **+0.124 vs +0.139** — and loses at k=256 (+0.139 vs +0.104), where it
is also NON-MONOTONIC vs its own k=64 (suspect: 512-dim sketch noise corrupting the
fine partition). Interpretation if it survives the check: the backward optimum differs
exactly where the metric has to choose what to keep (few atoms); with plenty of atoms,
activation-space fidelity dominates and the direct-logit metric misses what flows
through nonlinear paths. e2_unembed_vq.py/json.

Chained: e2b_stability.py — sketch 2048, two kmeans seeds, both k.
Queue: harvest → results/14 (E arc); then Logan consult on further E instantiations.

---

## 2026-07-18 — tick 58 (E-3: the crossover does NOT survive; L2 seed control chained)

**FINDING E-3 (corrects E-2):** stability check kills the k=64 advantage — unembed
vq64 spans +0.125…+0.150 across sketch/seed variants (E-2's +0.124 was the lucky
draw); vq256 unembed always loses (+0.131…+0.172 vs L2 +0.104). Seed-to-seed variance
(~±0.02) exceeds any metric effect at k=64. E-2's crossover claim RETRACTED pending
the L2 seed control (chained: e2c_l2seeds.py, two more L2 seeds at k=64 — needed to
state the null symmetrically).

Emerging E-arc synthesis (results/14 after the control): both cheap backward
instantiations (empirical Fisher, direct-U sketch) fail to beat plain L2 on
stream-table quantization; consistent with the program's denoising picture (vq1024
free; H5 denoising helps) — quantization error behaves as noise the downstream model
filters, so preserving activation-space geometry IS the right objective here.
Strongest untested instantiation: CE-refined assignments (behavioral Lloyd) — needs
training budget; consult Logan.

---

## 2026-07-18 — tick 59 (E arc closed: careful null; results/14 shipped)

**FINDING E-4 (closes the arc):** L2 seed control at k=64: +0.103/+0.139/+0.167 —
spread WIDER than unembed's; means indistinguishable (L2 ~0.137, unembed ~0.133).
Full synthesis in results/14: neither backward instantiation (Fisher, direct-U sketch)
beats forward L2; the one apparent positive was retracted under seeds; the null is
coherent with the denoising picture (quantization error = filtered noise → activation
geometry is already the right objective; direct-logit metrics can't see nonlinear
paths). QUESTION FOR LOGAN (in results/14): fund the CE-refined-assignments
instantiation (behavioral Lloyd — the only version that optimizes the discrete
structure against the binding metric directly), or accept the null and close E?

Program back to REST STATE pending Logan; unblocked queue empty.

---

## 2026-07-18 — tick 60 (LOGAN: pilot approved; behavioral Lloyd running; cron restored)

Logan approved the behavioral-Lloyd pilot. Launched `e3_behavioral_lloyd.py`:
bottom 12 streams, k=64, W=4 harness; start = best-of-3-seeds L2 partition (+0.1034);
4 iterations of {gradient pass through the PATCHED model (64k tokens; leaf =
creation-time gathered rows so all read layers + lambda chain aggregate), first-order
move scoring g_t·(C[c']−C[a_t]), damped moves (top 10%/stream, predicted-improving,
min-count 8), centroid recompute, held-out audit}; final cross-region audit.
Success bar: beat +0.103 by more than the ±0.03 seed floor. Approximations logged in
the docstring. Cron restored to 30-min (a0ca92b8).

---

## 2026-07-18 — tick 60b (pilot v2: fp16 reconciled; trust region added after iter-1 backfired)

Pilot v1 findings: (a) iter-0 discrepancy RESOLVED — the e-series' fp16 table storage
was the difference (+0.109 with fp16 round-trip vs +0.156 fp32; comparability
restored); (b) 10%-of-rows simultaneous moves BACKFIRED (+0.109 → +0.147): the
first-order scores don't survive 51k coupled moves — the predicted failure mode,
observed. v2 (running): MOVE_FRAC 0.02, 6 iterations, best-state tracking with
revert-and-halve trust region (Logan's own iterated-prune-revert protocol applied to
assignments), rowgrad freed per iteration (OOM fix).

---

## 2026-07-18 — tick 61 (E-5: pilot verdict — repairs, doesn't transcend; E ARC CLOSED)

**FINDING E-5:** behavioral Lloyd walked +0.142 → +0.103 (best held-out, iter 4;
late-region +0.106) — the good end of the L2 distribution, never below it; gains
decayed with half the steps reverting. Plus: identical-seed kmeans runs differ by
0.03 (GPU-atomic index_add nondeterminism) — the partition floor is chaotic, not
seed-stochastic. Full verdict in results/14 (updated): backward objective fails in
proxy AND direct behavioral form on this object; activation geometry was right all
along; behavioral refinement earns a place as a partition-repair tool only.
e3_behavioral_lloyd.py/json, e3_assigns.pt.

E ARC CLOSED. Queue: back to rest pending Logan (next-direction candidates:
softmax was vetoed; remaining: deeper fast-structure naming at top MLPs; anything
new Logan wants). Cron stays 30-min while conversation is active.

---

## 2026-07-18 — tick 62 (LOGAN REQUEST: explainer/glossary/examples doc set shipped)

Logan asked for a comprehensive methods doc set focused on what worked. Shipped three
files in results/, cross-linked and indexed from the README:
- EXPLAINER.md — objects/shapes/reductions (embedding, QK factors, OV, heads, bilinear
  MLP with the encoder/decoder split made explicit, streams), the nine methods with
  LaTeX, the five composition laws, the bits convention;
- GLOSSARY.md — ~35 working definitions;
- EXAMPLES.md — vq16 token classes, conjunction matches, the two contextual heads with
  decoded contents, mlp16 firing contexts + the TM-4 trap, figure pointers.
E arc remains closed (E-5); GPU idle pending Logan's read-through / next direction.

---

## 2026-07-18 — tick 63 (LOGAN REQUEST: visual explorer HTML shipped)

Built results/explorer.html ("Token tables & two heads") — self-contained visual
deep-dive: wall-routes bar chart, recent-share regime curve (QK + MLP), the W-ladder
log chart with trained-wall reference lines, vq16 token-class display, conjunction
match table (hits AND misses), L5 per-head signature-vs-causal scatter, H5/H7 detail
cards, mlp16 variance bars + firing contexts + the TM-4 trap verdict, and the
what-didn't-work section. All numbers from committed JSONs; hover tooltips;
light/dark themed. Published as a private artifact for Logan:
https://claude.ai/code/artifact/8763e1ff-ff6d-4cdc-bde5-7f0e5fd0f4b5

---

## 2026-07-19 — tick 64 (LOGAN FEEDBACK: explorer §2 rewritten — definitions + causal chart)

Logan's critique of explorer §2: undefined "interaction energy," invisible MLP markers,
and the causal paragraph (+0.003 vs +0.231) had no chart. Fixed in place (same URL):
(a) definitions panel with the exact formulas (stream sum → score pair-decomposition →
energy = mean squared pair term over sampled causal (i,j)); (b) MLP-hidden defined
(W_L x̂ ⊙ W_R x̂ pre-down-projection, down-column-weighted), markers enlarged with a
"5 layers measured" legend note; (c) NEW causal mini-chart: per-layer grouped bars for
L2/L5/L9 (window-only tabled / dominant-stream tabled / all tabled), tying the
observational energy map to the interventional ΔCE numbers. Re-published.

---

## 2026-07-19 — tick 65 (LOGAN SPEC: edge-ablation heatmap arc opened)

Logan's new spec: full lower-triangle module×module causal map — for every edge
(source stream → destination layer's reads), ablate the source IN THAT DESTINATION'S
READS ONLY, methods {zero, global-mean, PCA-1, PCA-4 (fixed subspace)}, dCE over
corpus; plot per-method heatmaps; plus a weights-only importance metric
(||R_dest·W_src||_F normalized) to verify empirically. Relation to prior work logged
in the reply: per-edge resolution is new (we aggregated by recency/layer); fixed-
subspace PCA ablation is new (H7/mlp16 rank-k kept LIVE coefficients); weights-only
screen never tested (history predicts partial failure — that's the point).

Launched edge_heatmap.py: 377 edges × 4 methods ≈ 1500 audits at 8 held-out chunks,
batch 8, resumable JSON (edge_heatmap.json), baseline×3 for the noise floor,
per-stream stats cached (edge_stream_stats.pt). ETA ~6h. Harvest = heatmap PNGs +
results/15 + weight-map correlation. Cron restored to 30-min (e459bfac).
Design deviations from Logan's list, logged: PCA-2 deferred (budget; addable
adaptively); "mean" here = global mean (our earlier tables were token-conditional
means — both will be discussed in results/15).

---

## 2026-07-19 — tick 66 (edge map harvested: FINDINGS EH-1..EH-4; results/15 shipped)

Sweep finished fast (~1h, 1508 audits). **EH-1 (sparsity):** 215/377 edges FREE under
zero-ablation; load-bearing structure = three families: within-layer attn→mlp (attn1→L1
+2.81, attn5→L5 +2.61), adjacent mlp→next (mlp16→L17 +3.89, mlp0→L1 +1.98), final
mlps→unembed (+1.30/+1.08). Windowed-D vindicated at edge resolution. **EH-2 (hub
dissociation):** attn5's mid-model energy presence is causally INERT (L7–L16 ≈ 0, some
negative); its real consumers are its own layer, L17, and the unembedding — energy maps
locate, ablations price (4th instance). **EH-3 (method ladder):** over big edges, zero
+0.302 → mean +0.176 → pca1 +0.156 → pca4 +0.144 — fixed-subspace PCA plateaus at half
the damage; reads want the TOKEN-CONDITIONAL component (cond-mean ≪ all of these).
**EH-4 (weights-only screen: total failure):** Spearman ρ=0.025 (p=.63) vs causal map —
the requested empirical verification is decisive; weight norms carry no edge information.
Also: real negative edges small (attn14→unembed −0.035 the largest). Plot-script NaN bug
in the first negative-edge extraction caught and fixed before write-up.
results/15 + fig_edge_heatmaps.png + fig_edge_weights.png; README indexed.

Queue: report to Logan; explorer §2 could absorb the edge map (ask/do next tick);
composed edge-set pruning (top-N free edges cut simultaneously — tests the standing
superadditivity law on the map's "free" set) as natural next GPU item.

---

## 2026-07-19 — tick 66b (EH-5: free-edge set does NOT compose; edge arc closed)

**FINDING EH-5:** 215 individually-free edges cut together = **+2.84** zero / +3.04
mean (149-edge stricter set: +0.54/+0.51) — strongest superadditivity instance yet
(marginals ≈0 → composed ~200× the sum). Mean worse than zero at scale (compounding
bias injections). Per-edge freeness ≠ pruning license; the map is a pricing tool.
Explains structurally why windowed-D (replace-with-conditional-summary, bound-by-
recency) succeeds where thresholded graph-cutting fails. results/15 §6.
edge_composed.py/json.

Edge arc complete (EH-1..EH-5). Queue: fold edge heatmaps into the explorer page
(offered to Logan); otherwise rest pending his direction.

---

## 2026-07-19 — tick 67 (explorer §8: interactive edge map; queue now empty)

Folded the edge arc into the explorer (same URL): new §8 with the full 377-cell
zero-ablation matrix as an interactive SVG heatmap (hover = exact per-edge ΔCE, data
inlined from edge_heatmap.json), the method-ladder and weights-null verdicts in prose,
and EH-5's composed-cut result as the closing verdict box. TOC renumbered (what-didn't-
work → §9). Re-published.

Queue: EMPTY pending Logan (edge arc EH-1..5 complete; all deliverables current).
Next silent tick retunes cron to hourly per policy.

---

## 2026-07-20 — tick 68 (LOGAN: table-MDL trio launched; circuit-card arc planned with guardrails)

Logan's asks: (1) run the three MDL-efficiency methods — shared codebook, low-rank
tables, edge-guided per-stream k; (2) qualitative circuit-finding through the
decompositions (cherry-picked OK); (3) requested my assessment of risks.

Launched e4_table_mdl.py: uniform-vq1024 baseline re-audited in-harness, low-rank
r∈{32,128}, shared codebook k∈{4096,8192} (per-stream RMS normalization + 37 scales),
edge-guided tiers (top-8 streams by causal weight k=4096 / mid k=1024 / tail k=64,
budget ≈ uniform). All W=4 audits; bits reported per arm.

Circuit-card design (next GPU slot), WITH the guardrails from my assessment: cards
trace ONE example through BOTH the token-static skeleton (table atoms per layer) AND
the named live components (H5 match, H7 gain, top-MLP gains) — tables alone would
show only the static part and miss the mechanism by construction; every card ships
with its SET-ablation check (cut the traced path as a whole; superadditivity law
makes per-edge traces unverifiable individually); cherry-picked labeled as such.
First target: induction copy of a repeated rare name (the one circuit already
causally mapped end to end).

---

## 2026-07-20 — tick 68b (table-MDL trio harvested: TM-MDL-1..3)

W=4 harness, uniform vq1024 re-audited in-run at +0.0888 (matches +0.094 within
re-cluster variance). **TM-MDL-1: low-rank r=32 WINS on dCE — +0.0741**, beating
vq1024 AND the full tables (+0.099): 36×/table compression that DENOISES (third
instance of the theme). r=128 no better than baseline at 4× the floats.
**TM-MDL-2: shared codebook k=4096 wins on bits — 4.7M atom floats (9× fewer) at
+0.0980**; k=8192 WORSE (+0.119, union-kmeans degradation). **TM-MDL-3: edge-guided
k allocation is a wash** (+0.0868 at more floats) — causal-importance budget tiering
didn't pay. Also fixed en route: mlp17 fp16 overflows (621 entries) now sanitized at
load (present in all prior vq runs; negligible impact, logged). Combo arm running
(r=32 basis + vq1024 on coefficients ≈ 2.6M floats + 18M idx bits — candidate
champion config). e4_table_mdl.py/json, e4b_combo.py.

---

## 2026-07-20 — tick 69 (combo champion + CIRCUIT CARD 1 shipped)

**TM-MDL-4 (combo):** r=32 basis + vq1024 coefficients = +0.089 at 2.5M floats + 18M
idx bits (~12 MB for the entire long-range flow) — bits champion; quality/bits frontier
is now {r=32 plain: +0.074 @59M floats} vs {combo: +0.089 @2.5M}. results/16 shipped.

**CARD-1 (first circuit card, format validated):** induction on 'Dunleavy...Dun'→'le'.
Selectivity ✓ (pair −3.38 vs random ±0.001). Honest content: H7 alone −6.39, H5 alone
−0.002 (WW-2 replicated at single-prompt level); pair ablation LESS damaging than H7
alone — non-additive interaction at TWO heads (composition law in miniature); skeleton
shows identity→class dissolution up the stack (emb peers 'Duncan/Dunham' → attn5 peers
generic name-prefixes). Bugs en route: tokenizer leading-space, emb not in tables
(analytic), both fixed. results/cards/card1_induction.md.

Queue: more cards (non-induction behavior; repeat-data prompt where H5 is load-bearing);
fold cards + table-MDL into explorer; Logan items.

---

## 2026-07-20 — tick 70 (LOGAN: contextual-circuits arc opened — n-gram ladder)

Logan's steer: get MORE CONTEXTUAL circuits, bottom-up; exploit co-occurrence (token ↔
its own attention-out); use the TN aspect; or dig into where/why the earliest layers
fail token-static and why weight heuristics can't help. Framing adopted: the context
ORDER ladder — 0th order = unigram tables (current program), next = BIGRAM-conditional
tables (frequent pairs + unigram backoff), then trigram/TT-factored. The earliest
streams are sequence-determined through attn0's pattern (a (t_i,t_j,Δ) contraction —
the TN object), so n-gram indexing is the natural refinement, and bigram rows are
nameable token-pair atoms (Logan's apple ↔ attention-for-apple co-occurrence).

Launched ngram_tables.py: 600k frequent bigrams (cnt≥4, ~6M-token estimation);
unigram vs bigram-backoff tables for {attn0, mlp0, attn1, mlp1}; per-stream R²;
audits at W=1 (unigram ref +0.861) and W=2 (+0.429), qk-reads. If bigram cracks the
W=1 gap, second-order context is n-gram-shaped → TT-factor the bigram tables next
(the tensor-network step) + bigram-indexed circuit atoms for cards.

Weight-heuristics question (to answer in the write-up): the needed objects are
data-measure-weighted contractions (pattern × value over the corpus distribution);
weights alone carry no measure — ρ=0.025 (EH-4) is the empirical face of that.

---

## 2026-07-20 — tick 70b (NG-1: bigram variance real, behavior null; pairclass variant chained)

**FINDING NG-1:** raw bigram-backoff tables: R² gains are LARGE (attn0 .19→.40, mlp1
.38→.60 — early-stream context IS substantially pair-shaped in L2 terms) but ΔCE gains
are ~nil (W=1 +0.888→+0.878; W=2 +0.443→+0.435). Two live confounds before concluding
the dissociation: coverage 53% (Zipf sparsity at 3.2M tokens, cnt≥4) and 4-sample row
noise — the ladder's structural problem is that ESTIMATION COST EXPLODES with context
order. ngram_tables.py/json/pt.

Chained: ngram2_pairclass.py — the TN-factored, dense version: correction table on
(class_prev × class_cur), 65k cells, ~50 samples each, 100% coverage, cells<8 shrunk
to zero; corr = pairclass-mean(z) − pairclass-mean(unigram-prediction). If the pair
signal survives class-factoring → estimable + nameable contextual atoms; if not, the
behaviorally-relevant early context is NOT local-pair-shaped (it's the long-range
selection structure the live window carries), and windowed-D's live window is vindicated
as the right treatment for exactly that residue.

---

## 2026-07-20 — tick 71 (NG-2/NG-3: ladder arc closed; results/17)

**FINDING NG-2:** pairclass-factored corrections (dense, 99.8% coverage, well-estimated)
retain ~2/3 of the bigram R² gain and STILL buy ~0.01 nats — confound-free dissociation:
local-pair context is variance-rich, behaviorally cheap. **NG-3 (the principled answer
to Logan's dig-in):** what early streams carry that matters is indexed by DYNAMIC
positions ("where my previous occurrence was"), not by the last k tokens — no n-gram
order captures it by construction; the live window is the correct treatment, not a
placeholder; and weight heuristics fail because the objects that matter are
data-measure-weighted contractions (weights carry no measure). Contextual-circuit route
= named live components + skeleton (cards), not finer context tables. results/17
shipped; README indexed.

Queue: report to Logan (his arc, decisive negative + the principled answer); rest
pending his steer. Candidates if he wants more: cards 2-3 (repeat-data H5 card;
non-induction card); pairclass atoms as descriptive layer in cards.

---

## 2026-07-20 — tick 72 (LOGAN DIRECTIVE: multi-hour autonomous arc; PUSHED; class-pair circuits launched)

Logan's standing directive: many hours autonomous; hardcore mech interp with maximal
TN use → more MEASURABLE MDL structure; verify reductions by falsifiable criteria
beyond dCE, esp. CAUSAL MONOSEMANTICITY (concentrated, cross-context-consistent
ablation effects); step back to this higher level every ~2h; keep everything pushed.

Housekeeping done: repo history REWRITTEN to drop 25GB of regenerable .pt caches
(filter-repo on the unpushed range; backup-pre-filter branch kept locally; *.pt
gitignored) → PUSHED to origin/main (112 commits). Cron prompt now carries the
directive + push-every-commit + 2h step-back.

New arc launched: cp_circuits.py — TN-native class-pair circuit atoms at layer 0:
coarsen the exact pattern tensor P(t_i,t_j,Δ) by embedding-classes (256), rank
(head, class_q, class_k) blocks by data-weighted pattern-energy mass, causally probe
the top 14 (zero the block only) and score each effect vector for MONOSEMANTICITY:
concentration (top-20 |Δlogit| share) + cross-context consistency (mean pairwise
cosine of per-position Δlogit vectors) + named promoted/suppressed tokens. Falsifiable:
diffuse or inconsistent effects kill the atom.

Queue after harvest: cards from the best blocks; block-sparse pattern MDL (keep top-B
class-pair blocks, ΔCE-vs-bits); extend monosemanticity scoring to existing atoms
(vq classes, H7 dir, mlp16 gains) for a cross-decomposition comparison.

---

## 2026-07-20 — tick 72b (cache restoration + attn2-seed0 FOUND)

filter-repo's checkout wiped the working-tree .pt caches along with history; ALL
restored from the local backup-pre-filter branch (no regeneration needed). Side
discovery during restore: **runs_hop/attn2-seed0/model.pt exists** — the "missing"
attn2-seed0 model was under runs_hop/ (the anchor scripts searched runs_owt/).
Logan closed the item as skip, but it's available if the original-anchor conjunction
test ever wants a re-run. cp_circuits relaunched.

---

## 2026-07-20 — tick 73 (CP-1: energy-selected blocks FALSIFIED as monosemantic; positive-control round running)

**FINDING CP-1 (round 1, honest negative):** all 14 energy-top layer-0 class-pair
blocks score concentration ≈0.00 and consistency ≤0.30 — diffuse, inconsistent
effects. AND the selection was compromised: pattern-energy mass ranks junk-token
classes (unicode debris, katakana) because the unnormalized bilinear pattern blows
up on rare tokens — energy-vs-causal mirage #5. Two live explanations: layer-0
blocks genuinely aren't output-monosemantic (plausible: layer 0 does transport, not
output-aligned features), or the metrics are too harsh (top-20-of-50k concentration
punishes class-level effects). cp_circuits.py/json.

Round 2 running (`cp2_controls.py`), per the positive-controls discipline: score
KNOWN-GOOD atoms (H7 rank-1 dir, H5 head, mlp16 dirs 0/3) + a random-direction
control + frequency-filtered content blocks, with refined metrics: participation
ratio, top-output-CLASS mass share, fire-conditioned consistency (top-decile
effect positions). Metric validates iff knowns pass and random fails.

---

## 2026-07-20 — tick 73b (CP-2: the metric's own mirages caught by controls; null-calibrated round 3 designed)

**FINDING CP-2:** positive-control round: (a) PR + class-share DON'T discriminate
knowns from random (all ~0.6-0.8 / ~0.01 — mean-vector-based, wrong object);
(b) fire-consistency discriminates but is CONFOUNDED by output proximity: ANY fixed
direction ablated at L16 yields mechanically consistent Δlogits (∝ U·d̂) — random
control 0.69, dir0 0.98, layer-0 blocks 0.03-0.16; (c) the decoded token lists DO
carry signal: mlp16 dir3 suppresses markup tokens (=\" , fmt, []) matching its firing
contexts; dir0 suppresses capitalized sentence-starters. The falsifiability loop
worked exactly as intended — on the ruler first. cp2_controls.py/json.

Round-3 design (next tick): NULL-CALIBRATED monosemanticity — every atom scored as a
percentile against N matched random atoms of the SAME TYPE AND SITE (random directions
at the same layer; random class-pair blocks at the same head), which absorbs the
mechanical baseline; per-position PR before averaging; and the where-fires↔what-pushes
alignment made quantitative (overlap between an atom's firing-context token classes
and its effect-token classes).

STEP-BACK (per Logan's 2h rule): the arc is producing exactly what he asked —
falsifiable verification machinery being validated before use. Priorities stay:
(1) finish the calibrated metric, (2) re-score all atom families with it,
(3) cards for survivors, (4) block-sparse pattern MDL still queued.

---

## 2026-07-20 — tick 74 (round 3 launched: null-calibrated monosemanticity)

cp3_calibrated.py running: 7 candidate atoms (mlp16 dirs 0/1/3, H7 principal dir,
3 content-class L0 blocks) each scored as percentiles against 8 matched random atoms
of the same type at the same site — the null distribution absorbs the mechanical
consistency confound (CP-2). Metrics: fire-consistency, median per-position
participation ratio, and ALIGN (share of effect mass on the atom's top-5
firing-context classes — the quantitative where-fires↔what-pushes). ~1.5h.

Queue after: re-score all atom families with the validated metric; cards for
survivors; block-sparse pattern MDL.

---

## 2026-07-20 — tick 75 (CP-3 + BS-1: monosemanticity arc closed with one survivor; selection rulebook found)

**FINDING CP-3 (arc close):** null-calibrated round: exactly ONE atom beats its
matched-null band — mlp16 dir0 (cons 0.98, pct 1.0). dir1/dir3 score BELOW null
(contextual gains legitimately vary — consistency-vs-null detects output-aligned
constancy, not meaning); H7 indistinguishable; L0 blocks decisively falsified.
Metric taxonomy + the three-round story in results/18. Every round was saved by a
control — the falsifiability loop worked, casualties in the right order (rulers first).

**FINDING BS-1 (TN-MDL positive):** the layer-0 selection tensor is block-sparse at
3% density — top-2048 class-pair blocks/head = +0.0004; 97% of class interactions
hard-zeroable. ~32k bits/head of rulebook structure. Composes (kept mass dominates;
cf. EH-5). Jointly with CP-3: layer-0 attention = a class-interaction ROUTER, not a
feature bank — its blocks are selection-meaningful, not output-monosemantic.
results/19. bs_pattern.py/json.

Queue: human-readable rulebook (top blocks named with exemplars) + density curves at
higher layers via cond-mean factors; behavior-targeted cards 2-3; step-back due next
tick (~2h mark).

---

## 2026-07-20 — tick 76 (STEP-BACK + rulebook/depth-density launched)

STEP-BACK (2h mark, per directive): the session has delivered against the higher goal —
TN-derived measurable MDL structure (windowed-D ladder; 12MB champion tables; BS-1's
3%-density selection rulebook) and a validated falsification loop (CP-1..3: metrics
audited before atoms; one survivor). Assessment: the "atom-first monosemanticity"
route is exhausted for this model (layer-0 = router, not feature bank); the productive
routes are (i) legible structure (rulebooks, cards with set-ablations) and (ii) ΔCE-
measured MDL ladders. Queue re-ranked accordingly: rulebook naming + depth density NOW
(running), cards 2-3 next, then per-layer rulebook bits into the MDL accounting.

Running: rulebook_density.py — (a) results/cards/rulebook_L0.md: top-8 blocks/head
named with class exemplars; (b) block-density ladders at L1/5/12/16 on LIVE patterns
(is 3%-sparsity universal or a layer-0 specialty?).

---

## 2026-07-20 — tick 77 (BS-2: universal 3% sparsity; 0.66MB whole-model routing; rulebook named)

**FINDING BS-2:** depth ladder — 3.1% density costs ≤+0.008 at every tested layer
(L1/L5/L12/L16); 12.5% free everywhere; 0.8% cheap in uppers, resisted by L5 (+0.25,
the contextual heads' tail). Whole-model attention routing ≈ 0.66 MB of rulebooks.
rulebook_L0.md: top blocks read as SAME-KIND matching + structure anchors. results/19
extended; root LOG updated for Logan. rulebook_density.py/json.

Queue: card 2 (repeat data, H5 load-bearing) next tick; card 3 (non-induction);
rulebook bits into MDL accounting; step-back done this cycle.

---

## 2026-07-20 — tick 78 (card 2 launched: the denoising paradox at single-sequence resolution)

card2_denoising.py running: 8 rare words repeated; arms = live / H5-zero / H5-content-
cleaned / H5-rank-2-filtered / H7-zero (catastrophic control) / random-head-zero (null
control); plus H5's attention target displayed at a worked position. The honest framing
per WW-2/7: the match head attends correctly, removal barely hurts, cleaning HELPS —
the card makes the program's strangest true fact legible on one sequence.

---

## 2026-07-20 — tick 78b (CARD-2: the denoising paradox has a BOUNDARY)

**FINDING CARD-2 (revises WW-7):** on a natural-word repeated sequence, cleaning H5's
content HURTS (−0.163; rank-2 filter −0.217) though both helped on uniform-random
repeats; controls held (H7 −2.774, random +0.007). Resolution: H5 carries CONTEXT-
MIXED identity — the context component is noise on degenerate data, signal on real
text; "under-cashing" is a degenerate-context statement. Card verdict rewritten to
match its own data; results/12 postscript amended. Cards are functioning as regression
tests on corpus claims — exactly the falsifiability behavior the directive wants.
card2_denoising.py, results/cards/card2_denoising.md.

Queue: corpus-scale natural-repeat cleaning arm (confirm the boundary beyond one
sequence); card 3 (non-induction); rulebook bits into MDL accounting.

---

## 2026-07-20 — tick 79 (H5-B: boundary confirmed at scale)

**FINDING H5-B:** natural-text A+A repeats, cleaning arm: +0.0344 (hurts) vs the
random-repeat reference on the SAME harness: −0.1701 (improves; WW-7 reproduced).
The card-2 boundary holds at corpus scale — H5's carriage is context-mixed identity,
noise only on degenerate data. results/12 updated with the numbers.
h5_boundary.py/json.

Queue: card 3 (non-induction behavior); rulebook bits into the MDL accounting;
explorer refresh with the directive-session findings (18/19 + cards).

---

## 2026-07-20 — tick 80 (CARD-3 shipped + STEP-BACK)

**FINDING CARD-3:** mlp16 dir0 on a legal citation: fires at citation-structure
positions (peak ` also`, 53k coeff); ablation moves case-name continuations +0.275 =
~100× both controls (dir1 −0.001, random +0.002); sign shows it is an INTRA-REGISTER
distribution shaper (suppressing boilerplate like ` JUSTICE` in favor of case-name
starts when removed), not a token booster. The validated atom now has a legible card.
results/cards/card3_dir0.md, card3.json.

STEP-BACK (tick-80, 4-tick cadence): three cards shipped, each yielding a finding
(non-additivity at two heads; the denoising boundary + scale confirmation; the
register-shaper reading of dir0). Structure results: universal 3% rulebooks (0.66MB
whole-model routing), champion tables (12MB). The directive's loop — TN structure →
measurable MDL → falsifiable verification — is now demonstrably operating end to end.
Remaining queue, re-ranked: (1) consolidated MDL accounting incl. rulebook bits (the
"we can in fact measure" deliverable — one table: every description layer, its bits,
its ΔCE); (2) explorer refresh (results/18-19 + 3 cards); (3) further cards as
behaviors suggest themselves. Then a Logan-facing session summary.

---

## 2026-07-20 — tick 81 (total-system audit launched)

combined_final.py running: the accounting table's headline number — windowed-D
(W=6, qk+v everywhere + mlp L1-12 reads, champion combo tables) COMPOSED with the
3%-density rulebooks at all 18 layers. Arms: rulebooks-only / tables-only / TOTAL.
Also the superadditivity test BETWEEN reduction families (marginals: rulebooks
~+0.01-0.02 summed, tables +0.059 — does the composition hold?). results/20
(consolidated accounting) written at harvest with this number on top.

---

## 2026-07-20 — tick 82 (TS-1/2/3 harvested; results/20 consolidated accounting SHIPPED)

**TS-1:** all-layer rulebooks compose 6× superadditively (+0.190 vs ~0.03 summed) —
whole-model routing quoted honestly at +0.19. **TS-2:** FIRST cross-family additivity:
tables (+0.042) + rulebooks (+0.190) → +0.256 (interaction +0.024) — different error
channels. **TS-3:** combo tables at W=6 = +0.042, better than raw tables (+0.059) —
denoising survives composition; new best windowed number. results/20 = the one-table
accounting (every description layer: bits, est. tokens, ΔCE, verification-beyond-ΔCE)
with the capstone sentence. combined_final.py/json.

Queue: explorer refresh (18/19/20 + cards) as the remaining deliverable; then
session-summary for Logan; further cards opportunistically.

---

## 2026-07-20 — tick 83 (explorer refreshed; session summary shipped; deliverables current)

Explorer §9 added (rulebooks + the bill + the verification loop), TOC renumbered,
republished at the same URL. Root LOG carries the Logan-facing session summary.
All results/01-20, cards 1-3, GLOSSARY/EXPLAINER/EXAMPLES, and the explorer are
current and pushed. Queue: opportunistic (more cards; per-layer rulebook naming;
sqrd12 rulebooks) — will continue generating in-scope experiments per the directive
unless Logan redirects.

---

## 2026-07-20 — tick 84 (SR-1/SR-2: rulebook generality split on sqrd12)

**SR-1:** per-layer block-sparsity + same-kind-matching flavor generalize to sqrd12
(3.1%: +0.027/+0.008 single-layer; blocks read identically). **SR-2:** composed
rulebooks do NOT (+0.569 all-layers vs bilin18's +0.190; +1.82 at 0.8%) — row
normalization couples blocks through the denominator. The (model × decomposition)
compressibility dependence recurs at family-subtype resolution. results/19 extended.

Queue: opportunistic per directive — per-layer rulebook naming; more cards; or
consolidation. Session deliverables all current and pushed.

---

## 2026-07-20 — tick 85 (STEP-BACK + SR-2 mechanism test launched)

STEP-BACK: session ledger — CP-1..3, BS-1/2, TS-1..3, H5-B, CARD-1..3, SR-1/2 across
results/18-20 + 3 cards + explorer §9-10, all pushed. Remaining in-scope value ranked:
(1) close SR-2's mechanism claim falsifiably (RUNNING: sqrd12_coupling.py — mask
numerator, keep ORIGINAL row sums; prediction: composed cost drops toward bilin18-like
if denominator coupling is the mechanism); (2) card 4 on the card-1 interference
mystery (which component compensates when H7 dies?); (3) per-layer rulebook naming
(descriptive, lower priority).

---

## 2026-07-20 — tick 85b (SR-3: coupling mechanism REFUTED)

**FINDING SR-3:** raw-denominator arm is slightly WORSE than renormalized (+0.687 vs
+0.569 @3.1%; +1.891 vs +1.822 @0.8%) — the denominator-coupling explanation for
sqrd12's poor rulebook composition is refuted; renormalization mildly repairs.
Leading alternative logged as open: head/branch redundancy (6×1 vs 9×2; wider-spread
per-head energy). results/19 corrected in place — the program's record stays honest
about its own conjectures. sqrd12_coupling.py/json.

Queue: card 4 (the card-1 interference mystery: which component compensates when H7
dies?); per-layer rulebook naming; else consolidate.

---

## 2026-07-20 — tick 86 (CARD-4: the interference is prompt-idiosyncratic)

**FINDING CARD-4 (negative, useful):** the card-1 two-head interference does not
reproduce at corpus scale — H7+H5 dead is ordinarily worse than H7 alone (+7.15 vs
+6.68); cleaning no better (+7.07); a mild interference shows with H0 instead (+6.25).
Co-ablation interactions are context-idiosyncratic; no stable compensation mechanism.
Second demonstration (after card 2) that the card→scale pipeline is the evidence unit.
results/cards/card4_interference.md, card4.json.

Queue: per-layer rulebook naming (descriptive) or consolidation; session deliverables
current; will continue per directive.

---

## 2026-07-20 — tick 87 (RD-1: diagonal matching fades with depth; L17 pronoun→temporal rule)

**FINDING RD-1:** diag-block energy fraction: L0 .31, L1 .44 (peak), L5 .19, L12 .21,
L16 .10, L17 .21 — same-kind matching is a bottom-of-model behavior; upper selection
is cross-class. Named L17 rule: pronouns→temporal-conjunctions. Rulebooks for
L1/L5/L16 written to cards/. Caveat: raw-energy names junk-polluted except L17;
freq-filtered naming queued if pursued. results/19 extended.

Queue state: descriptive-work marginal value declining; next tick's step-back decides
between freq-filtered naming, further cards, or resting pending Logan. All pushed.

---

## 2026-07-20 — tick 88 (CARD-5: the L17 rule is real, selective, featherweight, suppressive)

**FINDING CARD-5:** the pronoun→temporal block at L17: ablation effect +0.0016 on the
temporal class = ~5× matched controls, ~2× target-selectivity — causally real; but
milli-logit scale and SUPPRESSIVE in sign. Verdict: top-layer rulebook entries are
individually featherweight; the rulebook's meaning is aggregate routing structure.
Closes coherently with CP-1 + BS-1/2. results/cards/card5_l17rule.md, card5.json.

Session assessment: five cards, 20 results files, four arcs — the marginal value of
further self-generated descriptive items is now clearly below the session's earlier
work. Next tick: final step-back; likely consolidate + wind to rest-state cadence
pending Logan unless a high-value item emerges.

---

## 2026-07-20 — tick 89 (FINAL STEP-BACK: session consolidated; winding to rest)

Assessment: five cards, results/18-20, four arcs, every conjecture tested — remaining
self-generated items (freq-filtered naming polish, more per-layer listings) are below
the session's value bar. Per directive ("only rest if genuinely blocked"): the
genuinely high-value nexts are direction choices that belong to Logan (new model
family / deeper top-MLP / publication shaping). Consolidation shipped: root-LOG
capstone, README refresh, memory updated. Cron → hourly rest-state; 30-min restores
on his reply.

---

## 2026-07-21 — tick 90 (LOGAN'S PICK: top-MLP anatomy arc opened; cron restored)

Logan chose the top-MLP fast-structure deep-dive. Key structural opening: the gain
coefficient of any mlp16 output direction d is the EXACT weight-derived quadratic form
c_d(x) = x̂ᵀM_d x̂ + d·b, M_d = Σ_j (W_Dᵀd)_j W_L[j]⊗W_R[j] — no estimation needed.
Arc plan: (1) GATE the form against live coefficients; (2) eigen-anatomy of sym(M_d)
for dirs 0-3 (effective rank — is the fast structural state a few quadratic
features?); (3) name top eigenvectors (emb-NN + lens); (4) exact stream-pair split of
c_d over data (which stream interactions compute the state); (5) rank-r causal
replacement (ΔCE + coefficient R²); then upstream: what WRITES the driving streams.
mlp16_anatomy.py (exp 1: gate + eigen-anatomy + naming) launched; cron restored to
30-min (84478c16).

---

## 2026-07-21 — tick 90b (MA-1: form exact, weight-space DENSE; whitened anatomy running)

**FINDING MA-1:** the quadratic-form identity GATES exactly (rel-max 8.3e-07) — every
mlp16 gain is a closed-form weight object. But sym(M_d) is HIGH-rank (eff. rank
580–640 of 1152) for all four dirs: the rank-4-16 behavioral gain is NOT a weight-
space property; it must be DATA concentration (the L16 residual distribution excites
a thin slice of a dense form). Weight-space ev0 names are already suggestive (dir0:
sentence-final punctuation '.',').','!','?' — citation-boundary structure).
mlp16_anatomy.py/json.

Running: exp2 (mlp16_anatomy2.py) — data-whitened spectrum Σ^½M_dΣ^½ (does eff-rank
collapse to ~4-16?), top data-space features named. Then: exact stream-pair split of
c_d; rank-r causal replacement.

---

## 2026-07-21 — tick 91 (MA-2: data-whitening collapses the forms; boundary-feature quadratics)

**FINDING MA-2:** whitened eff-ranks: dir0 25.6 · dir1 36.7 · dir2 51.7 · dir3 24.3
(from ~600 weight-space) — the low-rank behavioral gain is DATA concentration of a
dense weight form, now quantified. Top whitened features decode as punctuation/
boundary structure across all four dirs (dir0 ev0: '.', ').', ':', ','; others:
newline/dash/quote). The fast structural state = quadratic interactions among
boundary features of the residual. Chain of description now: weights (dense form,
exact) → data metric (~25-50 quadratic features) → output behavior (rank 4-16).
mlp16_anatomy2.py/json.

Queue (exp3): rank-r FORM replacement in the live forward (c ≈ top-k whitened
features; ΔCE + coefficient R² — the causal check on MA-2) + exact stream-pair split
of c_d (which streams feed the boundary features). Then upstream: what writes them.

---

## 2026-07-21 — tick 92 (anatomy exp3 launched: causal rank-r forms + stream-pair split)

mlp16_anatomy3.py running: (a) live-forward replacement of all four dirs' coefficients
by rank-k whitened-form approximations (k=64/16/4), ΔCE + dir0 coefficient R² — the
causal check on MA-2's ~25-50-feature claim; (b) exact stream-pair covariance split of
dir0's coefficient (which stream interactions feed the boundary features — expected:
mlp15×mlp15 + attn5 pairs per SI-1/TM-1, now at coefficient resolution).

---

## 2026-07-21 — tick 92b (MA-3 harvested; results/21 shipped — the mechanism chain complete)

**FINDING MA-3:** rank-64 whitened forms for all four dirs run live at ΔCE +0.028
(dir0 R² 0.954; rank-16 +0.033); dir0's coefficient variance is fed by mlp15⊗mlp15
(dominant) + attn5⊗mlp15 — coefficient-resolution confirmation of the SI-1/TM-1
energy picture. results/21 ships the complete chain: exact weight form (gate 8.3e-7)
→ ~25-50 boundary-feature quadratics → mlp15+hub feeders → rank-4-16 gains →
register shaping. Logan's question ("what computes the fast structure, from where")
is answered at mechanism level.

Queue: recursion into mlp15 (same anatomy on the feeder's own bilinear form — walk
upstream until grounded in token-static structure); root-LOG update for Logan.

---

## 2026-07-21 — tick 93 (MA-4: recursion rung 1 — exact but BROADENING; grounding measure needs redo)

**FINDING MA-4:** L15 form for the boundary feature gates at 7.8e-7 (recursion sound);
whitened eff-rank ~113 at L15 vs 25-50 at L16 — anatomy BROADENS upstream; sharp
structure is composed from broad structure, so upstream walks fan out. Honest flag:
token-cond grounding R² = −0.19 is an estimation artifact (<2 samples/token); correct
grounding = windowed-table substitution inside the coefficient input (queued).
results/21 extended. mlp15_recursion.py/json.

Queue: windowed grounding of c_e (correct measure); step-back due next tick.

---

## 2026-07-21 — tick 94 (STEP-BACK + grounding curves running)

STEP-BACK (4-tick cadence): the anatomy arc has answered Logan's question at mechanism
level in 4 findings (MA-1..4, results/21); the remaining item is the corrected
grounding measure (running: ground_coeff.py — windowed-input R² of both coefficients,
W∈{0,1,2,4,6}; file-generation had the heredoc-apostrophe failure AGAIN — process
note: generator scripts now via Write-to-scratchpad + clean python, no heredoc string
surgery with escapes). After harvest: arc likely complete → consolidate + offer Logan
next directions (recursion is a fan-out per MA-4, so deeper walking is low-value;
better candidates: same anatomy for OTHER top-MLP dirs/L17; publication shaping).

---

## 2026-07-21 — tick 95 (MA-5: grounding curves; ANATOMY ARC COMPLETE)

**FINDING MA-5:** L16 consumer coefficient: R² 0.19 (W=0) → 0.96 (W=6); L15 feeder:
−0.81 → 0.77 (negative below W=4 — windowing corrupts the broad form off-manifold).
Sharpening IS grounding: the model distills downstream. Arc complete: MA-1..5 in
results/21 answer Logan's question end to end (exact forms → boundary quadratics →
mlp15+hub feeders → causally sufficient at rank-64 → grounding curves).

Queue: consolidation done; direction menu to Logan (other dirs/L17 anatomy;
publication shaping; anything new). Rest-state cadence at next silent tick.

---

## 2026-07-21 — tick 96 (Logan Q&A + OV-dictionary-variants experiment; cron restored)

Logan asked for: the exact layer-0 OV dictionary construction; a batch-top-k variant
swept over k; a routed/block-sparse variant (per-word-group dictionaries, his "8-of-64
vs 8-of-128" picture); code snippets + forward-pass explanation in temp_explainer.md.
Delivered: explainer §5 (dictionary definition, the one-line forward-pass change =
table lookup by token id, and code for all three schemes). Running: ov_dict_variants.py
— per-token top-k / batch-top-k / routed(G=8), swept, ΔCE + structural bits. First
number in: per-token n=512 k=4 = +0.277 (matches the sweep anchor). Cron restored to
30-min (dd2ab765). temp_explainer.md committed for durability.

---

## 2026-07-21 — tick 97 (OV dictionary variants harvested: routed WINS at matched bits)

**FINDING OVD-1:** three sparse-coding schemes for the layer-0 content tables, swept.
Per-token top-k: k=4/8/16/32 → +0.277/+0.218/+0.072/+0.001. Batch-top-k (flexible
per-token sparsity, same average): +0.413/+0.188/+0.064/+0.015 — BEATS per-token at
avg-k≥8 but worse at k=4 (starvation when budget is tight; flexibility needs slack).
**Routed/block-sparse (8 embedding-class groups, own dict each): +0.134 (uniform
8-of-128) / +0.123 (adaptive n_g) at ~180Mbits — clearly beats single-dict schemes at
matched or larger size** (+0.19-0.22), despite less training per dict. Logan's
per-word-family-dictionary intuition validated; routing is the efficient frontier for
content. Sweep table + all three code snippets in temp_explainer.md §5.
Caveat: L2-fit, 1200-step dicts (anchor was 3000) — relative comparison valid,
absolute above the trained anchor. ov_dict_variants.py/json.

Queue: CE-train the routed champion (does it go negative like the shared dict did?);
more Logan Q&A as it comes.

---

## 2026-07-21 — tick 98 (Logan's SVD-baseline question: OVD-2, sparse beats low-rank ~10x)

Logan: head_dim → SVD to rank d_head is lossless, so SVD is the honest baseline. Ran it
in-harness (ov_svd_baseline.py): head_dim=128 (his 64 guess corrected); SVD rank
8/16/32/64/96/128 = +2.24/+1.35/+0.59/+0.13/+0.036/+0.000 (rank-128 exactly lossless,
confirms rank≤128). **FINDING OVD-2:** sparse dictionary beats SVD ~order of magnitude
at matched bits everywhere. Cleanest: rank-16 SVD (one shared 16-dim subspace) +1.35 vs
per-token 16-of-512 (each token its own 16 dirs) +0.072 — same 16 coeffs, ~18x less
error. Content = UNION of low-dim subspaces, not one low-dim subspace; that's why
per-token sparse coding wins and SVD (optimal single subspace) can't. explainer §5
updated with the side-by-side. ov_svd_baseline.py/json.

Queue: CE-train routed champion; more Q&A.

---

## 2026-07-21 — tick 99 (OVD-3: batch-top-k "worse" is NOT a bug — resolved 3 ways)

Logan's concern: batch-top-k should be >= per-token top-k but is sometimes worse
(convergence/bug worry); + batch-size hypothesis; + wants adaptive+batch-topk routed.
BUG found & owned: the FIRST train-curve job used a RELATIVE path and cwd-reset, so it
never ran (stale 'done'). Re-run with absolute path.
**FINDING OVD-3 (three-part resolution):**
(1) Convergence: at 2500 steps FULL-BATCH the three shared schemes are within ~3% FVU
(per-token k=16 0.354, batch full-batch 0.364, batch minibatch 8192 0.367) — the big
sweep gaps were undertraining + the batch-size/threshold mismatch Logan guessed.
(2) Batch size WAS a factor: batch-top-k's threshold is per-minibatch at train but
full-vocab at eval; full-batch training removes the mismatch (0.367 → 0.364).
(3) DECISIVE same-dictionary test (ov_sametest.py): one dict, same k*V budget, encode
both ways → per-token 0.353 vs batch 0.403; batch STARVES 639 words to 0 atoms (per-row
count min 0 / median 14 / max 81). So the premise "batch >= token" is FALSE for
reconstruction: per-token gives each word its LOCALLY OPTIMAL k-term code; a global
budget can only help heterogeneous per-word needs and HURTS starved words. Batch-top-k
wins in SAEs (learned encoder + heterogeneous activation) but not for fixed-vector
reconstruction where content is fairly uniform in per-word complexity.
Routed done per Logan: adaptive group atoms (64-251 by size) + batch-top-k within each
group; all 8 groups converge cleanly (per-group FVU 0.019-0.103). Training curves in
results/fig_ov_training_curves.png. ov_train_curves.py, ov_sametest.py.

Process note (again): background jobs MUST use absolute script paths (cwd resets) —
this bit a 4th time. Added to the discipline.

---

## 2026-07-21 — tick 100 (converged matched-bits ΔCE comparison launched)

Following OVD-3: re-running all OV dictionary schemes with CONVERGED dictionaries
(4000-step full-batch) and the REAL cross-entropy audit, so scheme choice rests on
binding ΔCE at convergence (not the undertrained sweep). Arms: per-token top-k k=8/16,
batch-top-k full-batch k=8/16 (threshold now consistent train/eval), routed
adaptive+batch-top-k k=8. ov_converged_ce.py. Settles whether routed still wins and
whether batch closes the gap once converged + threshold-matched.

---

## 2026-07-21 — tick 100b (OVD-4: converged ΔCE corrects the story — batch loses, routed near-tie)

**FINDING OVD-4 (converged, real ΔCE, threshold-matched):** per-token top-k k=8/16 =
+0.125/+0.053; batch-top-k full-batch k=8/16 = +0.174/+0.056 (batch STILL loses at ΔCE,
confirming OVD-3 at the binding metric); routed adaptive+batch k=8 = +0.120 @192Mbit.
**Two corrections to the earlier sweep:** (1) batch does not beat per-token even
converged — per-token is locally optimal for fixed-vector content. (2) The "routed
crushes single-dict" claim was largely per-token UNDERTRAINING: per-token k=8 went
+0.218 (sweep) → +0.125 (converged), collapsing routed's landslide to a near-tie
(+0.120 @192M vs +0.125 @167M — same frontier, not a win). Methodological lesson logged:
compression comparisons must be at convergence. Fair routed test = per-token WITHIN
groups (not batch) — queued. explainer §5 corrected. ov_converged_ce.py/json.

---

## 2026-07-21 — tick 101 (fair routed test launched: per-token within groups + bits-matched reference)

OVD-4 left one loose end: the routed arm used batch-top-k within groups (the weaker
encoder). ov_routed_fair.py: all arms per-token top-k k=8, converged 4000-step
full-batch, real ΔCE — single dict n=512, routed uniform n_g=128, routed adaptive n_g,
plus a bits-matched single dict n=1024 (routed uses ~8x128 atoms vs single 512, so the
fair single-dict reference should also have more atoms). Settles whether routing
genuinely beats a single dictionary at MATCHED bits with a MATCHED (strong) encoder.

---

## 2026-07-21 — tick 101b (OVD-5: routing LOSES at matched bits — full correction; step-back)

**FINDING OVD-5 (clean negative, overturns the routing claim):** fair converged
matched-bits per-token comparison — single dict n=512 +0.125@167M; routed uniform
n_g=128 +0.101@183M; routed adaptive +0.128@192M; **single dict n=1024 +0.079@190M**.
The bits-matched single dictionary BEATS both routed variants (+0.079 vs +0.101/+0.128)
at same-or-fewer bits, despite routed's cheaper indices. Routing HURTS: group-confining
each word to 128 atoms wastes budget vs a shared 1024 any word can draw from —
union-of-subspaces content doesn't align with embedding-class partition. Adaptive sizing
also worse than uniform (small groups starved). FULL correction of OVD-1's "routed
wins" (which was undertraining + atom-inflation). Final recommendation: single shared
dict + per-token top-k, scale n & k — batch and routing both fail to help. explainer §5
final. ov_routed_fair.py/json.

STEP-BACK (tick ~4 cadence): the OV-dictionary Q&A arc is COMPLETE and self-correcting —
OVD-1..5 walked from "routed/batch win" to the honest "single per-token dict is best,"
each correction forced by convergence + matched-bits + decisive controls (same-dict
test, bits-matched reference). Deliverables: temp_explainer.md §5 (definition, forward
pass, all schemes, SVD baseline, training curves, converged corrections), 2 figures,
7 experiment scripts. Queue: await Logan (more Q&A / new direction). GPU idle at rest.

---

## 2026-07-21 — tick 102 (OVD-6: batch-top-k resolved — Logan RIGHT for orthonormal, overcompleteness breaks it; 0-atom red herring)

Logan's persistent (correct) intuition that batch-top-k should match per-token. Resolved
(ov_batch_probe.py, head 0, k=8, FVU): **ORTHONORMAL basis: batch 0.603 <= per-token
0.625 — TRUE, his argument is provable there** (error = sum dropped coeff², batch keeps
globally-largest = optimal; per-token is constrained). **OVERCOMPLETE 512-dict: batch
0.483 > per-token 0.467** — linear-encoder coeffs are non-optimal + magnitudes not
comparable across words, breaking the guarantee. Fixes tried (all fail to beat per-token):
min-1-atom floor 0.483, per-word-normalized select 0.489, warm-start 0.482. **0-atom words
are a RED HERRING** (I owed the correction — flagged starvation earlier): 1493 zero-atom
words have LOWER error (5679 vs 6374) because they're small-norm (content-norm 5797 vs
13370) and the bias fits them; batch is correctly not spending on them. Real fix would be
least-squares/OMP coefficient refit (comparable marginal error) — expensive. Practical:
per-token top-k right for overcomplete + linear encoder. Answered: V=50257 inputs/head ×9.
Histogram (0-38 atoms, median 7) + batch-topk code snippet added to explainer.
fig_batchtopk_hist.png. ov_batch_probe.py/json.

---

## 2026-07-21 — tick 103 (OVD-7 + OVD-8: least-squares batch VALIDATES Logan; Matryoshka hierarchy works)

**FINDING OVD-7 (Logan's intuition validated in full):** with least-squares coefficients
(greedy OMP + LS refit) and marginal-error global allocation, batch BEATS per-token even
overcomplete — per-token OMP k=8 FVU 0.4007, batch OMP marginal-error avg-k=8 FVU 0.3887
(0.389 < 0.401). The earlier reversal was ENTIRELY the linear encoder's non-comparable
coefficients; proper marginal error restores the orthonormal-regime guarantee. (OMP
per-token 0.401 also >> linear-encoder per-token 0.467 — LS helps everyone; flexibility
is the extra ~0.012.) OMP is iterative (analysis tool, not cheap inference encoder).
ov_omp_batch.py/json.
**FINDING OVD-8 (Matryoshka hierarchy):** nested-prefix-trained 512-atom dict — truncated
to first 32 atoms FVU 0.687 vs plain 0.832; first 128: 0.558 vs 0.692; full 512: 0.486 vs
0.467 (small full-dict cost, the usual Matryoshka trade). Real coarse-to-fine hierarchy:
a nested family of dictionaries for the price of one, enabling structured adaptive DEPTH
(easy words short prefix, hard words long) — cheaper-to-index + more interpretable than
arbitrary supports. figs: fig_matryoshka.png. ov_matryoshka.py/json. Both written into
explainer §5.

---

## 2026-07-21 — tick 104 (STEP-BACK + CE-confirmation of OMP/Matryoshka running + Logan's cluster-vs-rank Q chained)

STEP-BACK (4-tick): the OV-dictionary Q&A has become a rich self-correcting thread
(OVD-1..8): sweep → SVD baseline → convergence diagnosis → same-dict control →
routing-loses → orthonormal-vs-overcomplete → LS/OMP validates batch → Matryoshka.
Every claim gated by convergence + matched-bits + decisive controls; two of my own
earlier claims (routing wins; batch can't win) were overturned by Logan's pushes.
Deliverables: temp_explainer.md §5 (comprehensive), 5 figures, ~11 experiment scripts.

Running: ov_omp_matry_ce.py — binding ΔCE confirmation (all heads) of OVD-7 (OMP batch)
and OVD-8 (Matryoshka), since those were reconstruction-FVU only. Chained behind it:
qk_cluster_vs_rank.py answering Logan's new Q — QK number-of-clusters vs matrix rank
(clustering k∈{16..1024} vs SVD rank r∈{8..128}, real ΔCE, + effective-rank of the
clustered tables to show clusters != rank). Answered inline: QK factor tables are
V×128, rank≤128=head_dim; 256 clusters ≈ free (+0.008); clusters are a stricter
DISCRETE constraint than rank (k-cluster table has rank≤min(k,128)); inputs = V=50257
per head × 9 heads, 128-dim each (clarified in explainer).

---

## 2026-07-21 — tick 105 (OVD-9: CE confirmation — LS is the win, batch's reconstruction edge does NOT survive to ΔCE)

**FINDING OVD-9 (corrects OVD-7's "validated in full"):** binding ΔCE, all heads, k=8:
linear-encoder per-token +0.117; OMP per-token (LS) +0.062; OMP batch (LS, marginal-
error) +0.071; Matryoshka per-token +0.105. Two honest results: (1) the LARGE win is
LS coefficients (OMP), nearly halving loss (+0.117→+0.062) — batch-vs-per-token is a
sideshow. (2) Batch WON reconstruction (head-0 FVU 0.389<0.401) but LOSES at ΔCE
(+0.071>+0.062) — reconstruction≠behavior dissociation (same as pattern-MSE being a
useless ΔCE predictor). Logan's intuition validated for RECONSTRUCTION, not the binding
metric. Recommendation: per-token OMP (or per-token top-k cheap). Matryoshka ΔCE ~ plain
linear (+0.105); its value = hierarchy/adaptive depth, not peak loss. explainer §5
corrected. ov_omp_matry_ce.py/json. (qk_cluster_vs_rank.py still running behind.)

---

## 2026-07-21 — tick 106 (QCR-1: QK is LOW-RANK not cluster-shaped — rank-16 SVD beats 256 clusters; Logan's Q answered + deepened)

**FINDING QCR-1 (answers Logan's cluster-vs-rank Q, and deepens the dichotomy):**
per-branch [q|k] factor reduction, real ΔCE. Clustering k=16/64/128/256/1024 =
+0.041/+0.018/+0.012/+0.002/+0.010 (eff-rank of clustered table 12/42/68/82/98 — proves
clusters≠rank: k=256 → eff-rank 82 NOT 256). SVD rank r=8/16/32/64/128 =
+0.008/−0.002/−0.006/−0.005/−0.012. **Rank DOMINATES clustering: rank-16 SVD (−0.002,
improves model) beats k=256 clusters (+0.002) at 16 continuous dims vs 256 discrete
cells.** Selection is genuinely LOW-RANK (~16-32); the "256-class" headline is
behavioral-true but not minimal — low-rank is minimal. Sharpens the selection/content
dichotomy into GEOMETRY: selection = single low-dim subspace (SVD wins); content = union
of subspaces (sparse dict wins, SVD poor — rank-64 +0.13). fig_qk_cluster_vs_rank.png;
explainer §6. qk_cluster_vs_rank.py/json.

---

## 2026-07-21 — tick 107 (backward-direction Q&A + the positive test: when does backward win?)

Logan asked why the backward (unembedding-relative) direction failed (method E).
Answered in explainer §7: backward only helps under ADVERSARIAL error consumption
(some directions matter far more than others); method E's objects (token-static tables)
have noise-filtered error (uniform robustness — vq1024 free, low-rank improves), so no
asymmetry to exploit; also the unembedding metric sees only the direct linear path, not
the deep nonlinear one; empirically Fisher + unembed + behavioral Lloyd all = forward L2.
Positive prediction TESTED (backward_when_wins.py): forward SVD vs OUTPUT-GRADIENT-
whitened (backward) SVD of the layer-0 value table (content is behaviorally SENSITIVE —
carriage needs identity — the regime where direction SHOULD matter). rank r∈{4..64},
real ΔCE. If backward < forward at small r → direction matters here, converting the
method-E null into a characterization of WHEN backward wins.

---

## 2026-07-21 — tick 108 (BWD-1: backward loses even on sensitive content; the two-condition rule)

**FINDING BWD-1 (deeper null, completes the backward characterization):** forward SVD vs
output-gradient-whitened backward SVD of the layer-0 value table, real ΔCE: forward
+0.020/+0.021/+0.014/+0.009/+0.002 vs backward +0.022/+0.021/+0.014/+0.013/+0.003 at
r=4..64 — backward does NOT beat forward (within noise, forward slightly ahead). So even
behaviorally-sensitive content doesn't benefit from output-importance subspace choice.
RESOLUTION: backward wins only when BOTH (a) good shared low-rank basis AND (b) adversarial
error consumption in that basis. Selection has (a) but not (b) (noise-filtered); content
has (b)-ish but not (a) (union-of-subspaces, no shared basis — QCR-1). Neither circuit
gives both, so backward never wins here — not by accident but by structure. explainer §7.
backward_when_wins.py/json. Closes the method-E backward-direction question cleanly.

---

## 2026-07-21 — tick 109 (QCR-2: units error corrected; rank-then-VQ composes; sign-rank theory)

Logan caught a units error in QCR-1: I said rank "more compact" but rank-16 = 512
bits/token vs VQ-256 = 8 bits/token — VQ is ~60× cheaper/token; "16 dims vs 256 cells"
mixed dims and DL. CORRECTED in explainer §6: rank ~16-32 = intrinsic dimensionality
(geometry), k~256 = effective alphabet (cardinality), different questions. VQ dominates
per-bit (earlier finding stands). **FINDING QCR-2 (composed, his proposal):** rank-then-VQ
(VQ inside the rank-r subspace) real ΔCE + bits: pure VQ256 +0.003@45Mbit; pure rank16
−0.002@466Mbit; rank16+VQ256 +0.013@**12Mbit** (4× cheaper than pure VQ); rank32+VQ256
+0.011@17M; rank16+VQ1024 +0.009@21M. Cheaper on bits (validates composition) but NOT
strict domination — projection discards rank>16 residual, small ΔCE cost; new cheap
frontier point. THEORY (Logan's framing, data-supported): selection = scalar-per-pair
ranking → sign-rank-limited → low-rank (O(k²logV) dims); content = many discriminations →
union of subspaces → SAE regime. "Selection = sign-rank-limited, content =
union-of-subspaces." fig_qk_rank_vq_frontier.png; explainer §6. qk_rank_then_vq.py/json.

---

## 2026-07-20 — tick 110 (NEW THREAD: TN-gauge + Logan's overcomplete-Φ code propagation; toys)

Logan opened a new direction (joint/iterative TN-pure interaction-sparsity; then a
full overcomplete shared-dictionary **code-propagation** construction) and asked to
work on TOYS for fast iteration, default to running not waiting, and make a
goal-list. New subdir `basis_aligned/tn_gauge/` (GOALS.md = roadmap + his
construction as a testable ladder; PLAN.md = gauge findings).

**F1 — gauge primitives (toy_gauge_probe.py, block2 = [attn,mlp,attn,mlp]).** Exact
checks: a global residual rotation IS a gauge (RMSNorm-equivariant) but the embedding
has rank d, so pinning embed/unembed forces it to identity — **the shared residual
bond has ZERO interior freedom; the two boundaries pin the whole trunk** (no DMRG
sweep, no deep-layer SAE). Real freedoms are per-layer PRIVATE and independent: OV =
full O(d_head) (exact; an L1 rotation sparsifies it cleanly, CE unchanged); QK =
RoPE-constrained (a free head rotation blows CE by 18 nats → input-anchored, this is
why backward-from-unembed misses QK); MLP hidden = pinned by elementwise ⊙ (only
perm+scale, NOT rotation). Weight-only cross-layer composition DAG is uniform (1.1×
spread) → "which layers interact" needs data-contrastive scoring, not weight norms.

**F2 — shared-Φ code propagation gate (toy_code_propagation.py), NEGATIVE at m=512.**
One overcomplete dict Φ (m=512) coding EVERY bond, LS-refit coeffs, real TinyStories
(baseline CE 1.729). G1 FVU rises with depth (k=16: 0.067/0.104/0.142/0.229 bonds
0–3). G2 (binding): coding every bond costs ΔCE +2.71/+2.05/+1.52/+1.12/+0.59 at
k=4..64 — even 64/512 atoms/bond costs +0.59 nats. Naive shared dictionary does NOT
cheaply preserve the model; sets up gate 2 (dictionary size, shared vs per-bond). G3
MLP error amplification 1.0×(shallow)→1.4×(deep): below Logan's 2× worst-case bound
but depth-increasing (Step-5 mechanism holds directionally). fig_code_propagation.png.

NOT yet done (Logan's direct Q): propagating the layer-0 QK *measure* forward (gate 3).
Next: gate 2 fidelity/bits floor (m∈{512,2k,8k}, shared vs per-bond) then gate 3.

---

## 2026-07-20 — tick 111 (F3: gate 2 — code-propagation regime viable but reveals a propagation/fidelity TENSION)

Gate 2 (toy_fidelity_floor.py): F2's negative was mostly UNDERPOWERED, not fatal.
End-to-end ΔCE (baseline 1.729, k=32): shared m=512 +1.17 / per-bond m=512 +0.58 /
shared m=2048 +0.52 / per-bond m=2048 **+0.19** (bits 21/27/31/57 Mbit). Capacity and
per-bond both cut ΔCE ~6×.

**F3 (load-bearing):** Logan's Step-4 additive propagation (codes flow, no per-input
solve) REQUIRES one shared Φ (x_{l+1}=x_l+write ↦ code addition only if writer/reader
share Φ). But shared Φ is exactly the lossy config; per-bond buys fidelity (+0.19) at
the cost of re-encoding each bond = regime (a)/(b), NOT the free-propagation regime (c)
the construction targets. Cheap-propagation and faithful are in opposition on this toy.
Gate 2b (running, toy_shared_scaling.py): does scaling a SHARED Φ (m→8192, k∈{32,64})
reach ΔCE<0.05 or plateau? Then gate 3 = propagate the layer-0 QK measure forward.
GOALS.md F3 table. Chained: shared_scaling running.

---

## 2026-07-20 — tick 112 (gate 2b: toy atom-birth REFUTED but size-artifact; flagship premise HOLDS)

Logan refined the theory (mid-turn): additivity forces COMPATIBILITY not identity →
nested growing dictionary Φ_{ℓ+1}⊇Φ_ℓ (shared core + per-bond atom BIRTHS from
manufactured features); depth-degrading FVU = closure assumption failing. Gave the
decisive diagnostic (project bond residual onto upstream WRITE-mechanism span) to run
BEFORE any "regime is the limit" verdict, + calibrations.

**F4 (gate2b_writespan.py, toy):** coding residual is ISOTROPIC (eff-rank ~125/128),
write-span captures it ≈ random (0.25–0.30) « its own best-32-dim (0.35–0.43). Atom-birth
REFUTED — no structured missing subspace. BUT the cause is that d=128 activations are
near-full-rank (act eff-rank ~110–120/128): the toy is TOO SMALL to have the low-rank
stream the regime assumes. Verdict scope-limited, NOT "regime dead" (honors Logan's
over-claim warning). Calibration (b) bond-0-exact barely helps (+0.478→+0.464).

**F5 (bilin18_actrank.py, flagship):** bilin18 residual stream IS low-rank —
rank@90%-var ~150–260 of 1152 (13–22%), eff-rank ~530–650; most compressible mid-network
(bond 6 rank@90%=151). Premise HOLDS on the flagship; the toy verdict was a size artifact.

Chained: bilin18_writespan.py (gate 2b on the flagship, dictionary on middle bonds) —
the real atom-birth test now that there's genuine low-rank structure. GOALS.md F4/F5.

---

## 2026-07-20 — tick 113 (methodology correction: rotation FIRST; regime-1 floor; ladder reordered)

Logan (2 msgs): gate 1/2 = decoder-only dictionary learning on activations (SAE minus
encoder) — legal search, useful representability upper bound (~93% in 512 atoms), but
NOT the construction (skipped the zero-CE baseline, tier-2/3 not tier-1, silently absorbs
manufactured features). Reorder: rotation sweep → floors/budgets → weight-informed births
→ propagation → activation-audit. Flagship write-span (bonds 3/6/10/17: write-span
0.058–0.069 ≈ random 0.056) is CONFOUNDED by activation-training — NOT recorded as an
atom-birth verdict.

**F6 (toy_regime1_rotation.py):** exact per-head OV gauge Q∈O(d_head) maximizing
||oQ||₄⁴+||Qᵀv||₄⁴; applied to all heads ΔCE=−2e-6 (exact). OV L1 drops only 5.8–7.8%
(Hoyer 0.20→0.26) — OV bonds largely ROTATION-INCOMPRESSIBLE; ~93% L1 survives = the
zero-CE floor / superposition measure; remaining sparsity must come from regime-2 births.
**Positive control caught a dead optimizer:** L1-subgradient Cayley gave 0.3% on a
planted-sparse control (should recover ~78%); switched to L4 ascent (plant 78%, random ~0)
and rediscovered the true 7% floor. [[positive-controls-catch-solver-bugs]] again.

DEVIATION FLAGGED: sweep runs on PRIVATE bonds not residual bonds — end-pinning both
boundaries pins the shared residual interior (Q_ℓ=I), so residual sparsity comes from
births not rotation; and private bonds being independent, regime 1 is parallel not swept
(DMRG coupling enters in regime 2). Awaiting Logan confirm. Next: per-bond atom budgets
from floors → weight-informed births (dedup/orthogonalized). GOALS.md F6 + reordered ladder.

---

## 2026-07-20 — tick 114 (flagship regime 1: value bus shared across depth + rotation-incompressible)

Ported regime-1 OV rotation floor to bilin18 (independent of the open flagged question;
OV bonds are unambiguously private). Two findings, both caught/verified by the ΔCE gate:
(1) naive PER-LAYER OV rotation is NOT a gauge (max|Δlogit|=16.8) — bilin18 mixes every
layer's value with block-0's (v=(1-lamb)v+lamb·v1, tier2_model L87-89), so the value
head-subspace is SHARED across all 18 layers (the value bus, like the residual bus, is
shared — here for a concrete architectural reason). (2) the correct SHARED-per-head gauge
IS exact (max|Δlogit|=5e-4) but rotation buys ~0% (L1 drop 0.01-0.06%, Hoyer flat 0.22):
one 128-dim rotation can't jointly sparsify 18 layers, so the flagship OV subspace is
fully ROTATION-INCOMPRESSIBLE (floor ≈100%) vs toy 7%. => on bilin18 ALL OV sparsity must
come from overcompleteness (regime-2 births); the square-rotation baseline is empty there.

The gate caught a wrong per-layer gauge assumption (2nd gate-catch this session after the
dead L1 optimizer). bilin18_regime1.py/json; GOALS.md F7.

QUEUE: regime 2 (per-bond budgets + births) BLOCKED on Logan confirming the flagged
private-vs-residual-bond question (its design depends on the answer). Independent next:
QK constrained-rotation floor (RoPE-commuting subgroup), done carefully with controls.

---

## 2026-07-20 — tick 115 (regime 1 COMPLETE: QK RoPE-torus floor 1.4%; step-back — rotation baseline is nearly empty)

Finished regime 1 with the query/key bond (toy_qk_torus_floor.py). QK rotation is a gauge
only if it commutes with RoPE -> for rotate-half RoPE the commuting subgroup is a 16-angle
TORUS per head/branch (one 2D rotation per frequency plane), vs OV's full O(32). L4 ascent,
GATED by a planted-torus control (recovers the known optimum 96.4; a first miscalibrated
threshold 'FAILED' the passing optimizer — fixed to 'recovered known optimum'). QK floor =
1.36% L1 drop, exact gauge ΔCE -1e-7.

STEP-BACK (regime-1 summary, fig_regime1.png, all gauges ΔCE≈0): toy OV 7.0% | toy QK 1.4%
| flagship OV ~0% (value bus shared across depth). The square-rotation baseline is NEARLY
EMPTY: no private bond yields much sparsity to an exact orthonormal change of basis. So the
whole sparsity budget must come from OVERCOMPLETENESS (regime-2 births); regime 1's
deliverables are the zero-CE anchor (a denominator for the overcomplete arm's ΔCE) and the
proof that rotation alone can't compress these bonds. Two shared-bus facts surfaced (residual
via embedding-pinning, value via lamb-mixing), both gate-caught.

Regime 2 remains BLOCKED on Logan confirming the flagged private-vs-residual bond question.
Nothing running. Next independent options if Logan silent: flagship QK torus floor; or the
representability-vs-overcompleteness curve on a single bond (regime-2 prep that doesn't need
the birth-seeding decision). GOALS.md F8 + step-back.

---

## 2026-07-20 — tick 116 (regime 1 fully closed: flagship QK 0.22%; regime 2 first step: un-confounded births SUPPORTED)

Harvested flagship QK RoPE-torus floor: 0.22% L1 drop (uniform across depth), exact gauge
(max|Δlogit| 3e-5). Regime 1 now COMPLETE across both bonds x both models: toy OV 7% | toy
QK 1.4% | flagship OV ~0% | flagship QK 0.22% — the square-rotation baseline is empty
everywhere; sparsity must come from overcompleteness.

**F9 — un-confounded births test (toy_births_seed_test.py).** Regime 2's first step, and the
un-confounded fix to F4's confounded write-span: SEED atoms from weights (never trained),
compare seedings by fixed-dict sparse-code FVU. Deep-bond mean: WRITE 0.389 < TOKEN 0.439 <
RANDOM 0.518 (5 subsamples, std ~0.005). Write-seeded reliably beats token+random and the
gap GROWS with depth (bond2 write 0.352 vs token 0.502) -> weight-informed births SUPPORTED,
un-confounded. Gate note: a single-sample random draw was flukey-good (0.16); the 5-subsample
check corrected to 0.519 before any claim (falsifiable-verification rule earned its keep, 3rd
gate-catch this session). Proceeded with births-hypothesis validation (not the full birth
construction) since it doesn't depend on the still-open residual-vs-private flagged question.

Next: flagship births test; then nest births over the rotation basis with orthogonalization
(clean DL). Regime-2 CONSTRUCTION specifics (which bond, nesting) still want Logan's flagged
confirm; the hypothesis test did not. GOALS.md F8/F9; fig_regime1.png.

## 2026-07-20 — tick 116b (flagship confirms F9: write-seeded births decisively beat token/random)

bilin18_births_seed_test.py (bonds 3/6/10/17): WRITE 0.692 < TOKEN 0.850 < RANDOM 0.918 mean
FVU (std ~0.01) — same ordering as the toy, larger gaps on the real low-rank stream (F5).
Weight-informed births decisively supported on the flagship, un-confounded. Next: nest births
over the rotation basis (orthogonalize for clean DL) and measure the sparsity/ΔCE they buy.

---

## 2026-07-20 — tick 117 (regime 2 binding metric: write>token>random survives ΔCE, but seeds are an init)

toy_births_dce_test.py — F9 (reconstruction) lifted to ΔCE (binding rule). Fixed seeded
dicts, bond0 exact, bonds1-3 coded, m=512 k=32: write +2.81 < token +2.90 < random +3.47
(3 seeds). Ordering SURVIVES at ΔCE (write clearly beats random 0.66; beats token marginally
0.09 ~1.5std, attenuated from reconstruction). But absolute ΔCE catastrophic (+2.8 vs trained
+0.19-0.52 in gate 2): fixed seeds destroy the model -> write-seeding is the right DIRECTION
but seeds are an INITIALIZATION not a solution. Proceeded (unblocked: hypothesis/metric test,
not the full-construction bond-choice that wants Logan's flagged confirm).

Chained: write-init + training vs random-init (do good seeds give faster/better convergence?).
GOALS.md F10.

## 2026-07-20 — tick 117b (REVERSAL F11: write-seeding is a good fixed dict but a BAD training init)

toy_births_init_test.py: write-init vs random-init, overcomplete m=512 k=32 per bond.
write-init loss@50 0.817 -> ΔCE +0.50; random-init loss@50 0.108 -> ΔCE +0.35. Random init
trains 8x faster AND to a better optimum. Cause: write directions are clustered/rank-limited
-> redundant seed atoms, poor coverage; random spreads better. REVERSAL of F9/F10 (where
write won as a FIXED dict). Regime-2 seeding arc (F9-F11) synthesis: write directions identify
the right SUBSPACE but are a poor OVERCOMPLETE atom set; overcompleteness needs diversity, not
write-seeding; trained+random-init dict is the practical winner (+0.35, matches gate-2).
Orthogonalizing write seeds (Logan's dedup) collapses them to a <=d basis, killing
overcompleteness - the same tension from the other side. GOALS.md F11.

Regime-2 seeding thoroughly characterized. Next step (full nested construction / which bond /
DL accounting) is a design fork that wants Logan's flagged confirm; the seeding sub-question
is answered. Nothing running.

## 2026-07-20 — tick 118 (F12 closes F11: write-info useless for a trained dict; regime-2 seeding arc CLOSED)

toy_births_ortho_init_test.py: random ΔCE +0.349 (loss@50 0.108) | clustered-write +0.497
(loss@50 0.817) | ortho-write +0.351 (loss@50 0.102). De-clustering (write-PCA, diverse)
RESCUES convergence to random level (clustering WAS the F11 handicap) but confers NO advantage
(+0.351 ~ +0.349). Verdict: training finds the write subspace unaided; weight-informed init
offers nothing for the trained dict.

REGIME-2 SEEDING ARC (F9-F12) CLOSED: write dirs capture the right SUBSPACE (fixed-dict F9/F10)
but are useless for the TRAINED overcomplete code (clustered/rank-limited F11; de-clustered just
matches random F12). Practical faithful code = trained dict, any diverse init, ΔCE ~+0.35 @
m=512/k=32 (matches gate-2). Real cost, a Pareto point not a free reduction.

STATE: regime 1 (rotation floors, empty everywhere) + regime 2 seeding (F9-F12) both fully
characterized. Remaining work is a DESIGN FORK needing Logan's steer (full nested construction /
which bond / accept trained-dict-code and pivot to interpreting its atoms / monosemanticity),
plus the still-open flagged private-vs-residual question. Pausing tn_gauge grind at a clean
consolidation point rather than spawning marginal experiments. Nothing running. GOALS.md F1-F12.

## 2026-07-20 — tick 119 (Logan STEER: layer-1 QK source-interaction graph — sparse, M×M-dominated)

Logan steered: focus layers 0-1; optimize attn2 (layer-1) QK to depend SPARSELY on upstream
sources {E=embedding, A=attn0 OV output, M=mlp1 bilinear output} with good CE; one bond ->
stronger methods OK. Also bank regime-1 MDL as the baseline.

F13 (toy_qk1_interactions.py): QK score bilinear in x2=E+A+M splits EXACTLY into a 3x3 source
graph (gate sum-of-blocks=real to 3e-4, full ΔCE=0). Frobenius mass: MxM 0.70 dominant, MxE
0.10, ExM 0.07, MxA/AxM 0.09, ExE 0.01, A-pure ~0. Causal ΔCE: MxM ALONE +0.062 (usable); all
other single blocks catastrophic (+1.8); cumulative by mass MxM+MxE+ExM=+0.008, 6 of 9 blocks
= +0.0001. => layer-1 selection runs almost entirely on the bilinear output self-interaction
(MxM), weakly modulated by embedding; attn0 output A not directly read. Sparse interpretable
source graph = coarsest version of Logan's ask.

Regime-1 MDL baseline banked: rotation ~0 sparsity everywhere (F6-8) => regime-1 DL ~ raw
weight bits; layer-1 QK raw ~2.1 Mbit (4 x 128x128 x 32), rotation doesn't reduce it.

Next: decompose M (and E) into atoms, sparsify the fine MxM/MxE atom-interaction graph, then
its MDL vs the 2.1 Mbit baseline. GOALS.md F13.

## 2026-07-20 — tick 120 (F14: per-source rank for layer-1 selection — E/A compress, M does not in variance basis)

toy_qk1_source_rank.py: PCA-decompose each source, project to rank r, ΔCE (gate full=0).
E low-rank (r8 +0.021, r2 +0.038), A negligible (r2 +0.005), M HIGH-rank (r8 +0.65, r16 +0.40,
r64 +0.11 - does NOT compress by variance). Redirect: PCA optimizes variance not interaction;
M's high PCA-rank includes selection-irrelevant variance. Interaction-sparse basis = QK-singular
(M×M diagonal per head); QCR-1/2 showed the QK form is low-rank ~16-32. Next: low-rank-reduce
the layer-1 QK maps (= decompose M in the interaction basis), ΔCE vs rank + MDL vs 2.1Mbit. F14.

## 2026-07-20 — tick 120b (F15: layer-1 QK ~rank-64; steered-task synthesis F13-F15)

toy_qk1_lowrank.py: low-rank the attn2 QK maps, ΔCE/bits: r2 +0.93, r8 +0.28, r16 +0.16,
r32 +0.07 (1.05Mbit=50% raw), r64 +0.012 (2.1Mbit=100%), r128 0. Layer-1 QK ~rank-64 - a
Pareto trade not free; HIGHER rank than layer-0 (QCR-1 rank-16 free) since selection reads the
richer bilinear output M.

STEERED SYNTHESIS (F13-15): layer-1 QK is SPARSE at the SOURCE level (M×M dominant, A droppable,
E minor low-rank; 6/9 blocks recover model) but NOT compressible WITHIN M (M high-dim ~rank-64
in both variance F14 and interaction F15 bases). Clean finding = WHICH sources interact (layer-1
selects on bilinear-output self-interaction), not a low-atom code for M. Remaining optional
avenue: a LEARNED interaction-sparse basis (direct optimization) vs low-rank/variance. GOALS.md F15.

## 2026-07-20 — tick 121 (F16: M high-RANK but SPARSE in a learned basis - the interaction decomposition EXISTS)

toy_qk1_learned_basis.py (the "stronger technique"): L4-optimize a full O(D) rotation of the
M-input basis to sparsify attn2 QK reads [q1;k1;q2;k2]. Gated by planted control (recovers 87%
of 89% optimum; random -1.8%). Reads sparsify 24.7% L1, Hoyer 0.24->0.43 (vs regime-1 head-dim
7%/1.4%). BINDING (prune reads, ΔCE): keep 50% learned +0.057 vs original +0.17; keep 25%
learned +0.14 vs original +1.95 (14x better); keep 12.5% learned +0.45. => M is high-RANK
(F14/F15 can't low-rank it) but SPARSE in the right basis - a sparse-not-low-rank structure
variance+SVD both miss. The interaction-sparse decomposition Logan wanted EXISTS. MDL honest:
keep-25% reads + basis V ~1.0Mbit @ +0.14 ~ low-rank r16 (0.52Mbit @ +0.16) - not a bits win
but the sparse interpretable structure, prunes far better than naive basis. Updates F15
intrinsic-lean -> basis-dependent. GOALS.md F16.

Steered task (F13-F16) COMPLETE: layer-1 QK sparse at source level (M×M dominant) AND M sparse
in a learned basis (high-rank but sparse-not-low-rank). Optional next: flagship confirm; or the
per-source atom-interaction GRAPH (which M-atoms interact with which E-atoms) now that a sparse
M-basis exists.

## 2026-07-20 — tick 122 (F17: FLAGSHIP OVERTURNS F16 - bilin18 layer-1 QK already sparse in standard basis; step-back)

bilin18_qk1_learned_basis.py: binding-metric generalization of F16 to bilin18 h[1]. Control
passes (planted 86%). Learned input-basis rotation barely sparsifies reads (1.3% L1 vs toy
24.7%) and does NOT help pruning - ORIGINAL basis prunes BETTER: keep50% -0.003 (improves),
keep25% +0.009, keep12.5% +0.055 (learned: +0.003/+0.026/+0.107). => bilin18 layer-1 QK is
ALREADY sparse in the standard basis (drop 75% weights for +0.009, no rotation); F16's learned-
basis win was a d=128 TOY ARTIFACT (tiny model packs QK densely). Flagship check overturned the
toy conclusion (program's own lesson). MDL: keep-25% ~ 25% raw QK bits + indices @ +0.009 - a
genuine flagship sparsity reduction.

STEP-BACK (F13-F17): layer-1 selection decomposition settled. (1) source-level sparsity real
(M×M dominant, attn0-out droppable, F13). (2) within-source compression MODEL-DEPENDENT: toy
high-rank needs learned basis (F14-16); flagship QK directly ~75% sparse in standard basis (F17)
- the clean real-model result. Toy rotation machinery was compensating for small-model density.
Next optional: flagship source-level graph; or accept direct QK sparsity. GOALS.md F17.

## 2026-07-21 — tick 123 (F18: FLAGSHIP CONFIRMS F13 - layer-1 selection runs on the bilinear output)

bilin18_qk1_sources.py: causal source ablation on bilin18 h[1] QK (per-head QK-norm blocks the
exact bilinear split -> ablate). Decompose xin1=E+A+M, remove each from QK input only, ΔCE.
Gates: E+A+M=xin1 1.5e-5; inline forward = reference EXACTLY (Δ=0, after fixing 2 bugs the gate
caught). Result: remove M (block-0 bilinear/mlp out) +0.676 (essential); remove A (attn out)
-0.0002 (droppable); remove E (embedding) -0.011 (slightly helpful). => flagship layer-1
selection runs almost entirely on the BILINEAR OUTPUT - F13's interpretive finding GENERALIZES
(contrast F16 compression = toy artifact per F17). Durable model-general result: layer-1 QK
selects on what the bilinear layer computed, not raw tokens/attn output.

Gate discipline: reference-CE gate caught broken inline forward TWICE (omitted value-bus mixing;
omitted embedding-RMSNorm) before any claim. GOALS.md F18.

## 2026-07-21 — tick 124 (F19: bilinear-output selection is LAYER-1-SPECIFIC; deep selection distributed)

bilin18_depth_sources.py (forward=reference exactly). Ablate block(L-1)'s mlp/attn write from
block L's QK input across depth. Remove preceding MLP: L1 +0.676, L2 +0.066, L3 +0.027, L6 +0.016,
L9 +0.001, L12 +0.004, L17 +0.005. So F18's "layer-1 selects on preceding bilinear output" is
strongly LAYER-1-SPECIFIC, decaying fast with depth; deep layers (9/12/17) barely depend on any
single preceding write (distributed, read accumulated residual). L6 minor exception (preceding
ATTN +0.060 > MLP). Honestly bounds F18 - real but layer-1-scoped, not universal. GOALS.md F19.

tn_gauge layer-1-QK arc (F13-F19) complete: layer-1 selection runs on the bilinear output
(toy F13 + flagship F18), sparse at source level; the within-source compression is model-
dependent (toy learned-basis F16 = artifact; flagship directly sparse F17); and the phenomenon
is early-layer-specific (F19). Gate discipline caught multiple bugs (dead optimizer, per-layer
gauge, forward x2). Nothing running.

## 2026-07-21 — tick 125 (F20: layer-1 selection mechanism - predominantly long-range content-based)

bilin18_layer1_pattern.py (forward=reference): h[1] attention read-weight by relative offset
(|pat| normalized per query). Local(<=2) 0.23, long-range(>8) 0.62. Most heads (0/2/4/6/8)
long-range (65-84% beyond offset 8); head 1 local (0.63 within <=2); heads 3/5 prev-token-ish
(peak@offset1 ~44%). So the special layer-1 selection (runs on bilinear output, F13/F18)
implements a predominantly LONG-RANGE CONTENT-BASED read, not positional/induction - consistent
with reading M richly (F14-16). Sanity check (fraction>1 impossible) caught a normalization bug
before the wrong 'local' verdict. GOALS.md F20.

Layer-1-QK arc now fully characterized (F13-F20): sources (bilinear output), depth-scope (layer-1
-specific), mechanism (long-range content-based). Nothing running.

## 2026-07-21 — tick 126 (F21: layer-1 QK MDL frontier - the banked baseline Logan asked for)

bilin18_qk1_mdl_frontier.py + fig_qk1_mdl.png: layer-1 QK (raw 169.9 Mbit) compression frontier,
matched-bits, ΔCE binding, index bits side by side (MDL convention). Low-rank: r64 +0.13@11%,
r128 +0.06@22%, r256 +0.03@44%. Prune: keep50% -0.003@82%, keep25% +0.009@41% (42.5val+27.0idx),
keep12.5% +0.055@20%, keep6.25% +0.24@10%. Layer-1 QK compresses to ~40% raw near-free (+0.009)
or ~22% for +0.06. Methods cross ~20%: low-rank wins low-budget, prune wins high-budget. keep-50%
improves CE (half the QK weights removable-with-benefit). Regime-1 rotation = 0 compression (raw)
-> frontier is what future methods must beat. Banked baseline. GOALS.md F21.
