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
