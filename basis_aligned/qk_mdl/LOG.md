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
