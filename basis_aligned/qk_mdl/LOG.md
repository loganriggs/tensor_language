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
