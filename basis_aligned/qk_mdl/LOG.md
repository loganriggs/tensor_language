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
