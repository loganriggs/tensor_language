# jacclust — Jacobian clustering & operator-SAEs for bilinear layers

Research program on decomposing **bilinear layers** `y = D(Lx ⊙ Rx)` by the *operation* they
apply per datapoint. Two lines: (1) the exact per-datapoint Jacobian kernel `⟨J(x),J(x')⟩_F = xᵀGx'`
and clustering by it; (2) **operator-SAEs** — sparse dictionaries of the per-token secant
`M = y·x⁺` (Logan's bilinear-secant SAE). Everything is on the `Elriggs/gpt2-bilinear-*` checkpoints
(bilinear/squared-attention transformers) plus tiny in-repo models and synthetic toys.

## Start here (docs)

| file | what |
|---|---|
| **`SUMMARY.md`** | **Authoritative synthesis — read first.** Solid results S1–S20 with provenance, retraction table, one-paragraph bottom line. Trust this over `results.md`. |
| `results.md` | Chronological tick-by-tick log (57 ticks). Retracted claims sit next to survivors — use `SUMMARY.md` for what is actually true. |
| `LOG.md` | Batched questions / status / decisions for Logan. |
| `mechanism_decomposition_spec.md` | The original methods writeup (the spec this program tests). |

## Core modules (imported by experiments — do not move)

| file | what |
|---|---|
| `metric.py` | Exact Jacobian kernel: `jacobian`, `gram` (weights-only `G` with `⟨J,J'⟩=xᵀGx'`), `embed`, `frobenius_cosine`. Verified to 1e-14. |
| `surrogate.py` | Per-cluster linear-surrogate validation; `projected_gram` (G-top projection recipe), `random_spectrum_gram` (the `G_rand` spectrum-matched control). |
| `tt_model.py` | Tensor-transformer model classes (GPT, `CausalSelfAttention`, `CausalBilinearSelfAttention` = two-QK, `Bilinear` MLP) extracted from modded-nanogpt; loads the Elriggs checkpoints. |
| `bilinear_sae.py` | **The bilinear-secant SAE** (Logan's architecture): reconstruct the secant `M=y·x⁺` as a sparse sum of rank-1 atoms with the Dooms expanded quadratic loss. Toy self-test in `__main__`. |
| `dgp.py` `dgp_c.py` `dgp_e.py` | Synthetic generators: gated linear experts (DGP-A/A′/D), two-layer composition (DGP-C), no-control-stream quadratic gate (DGP-E). |
| `intervene.py` | Intervention (patch/ablate) validation helpers on a real MLP. |

## Experiments (script → purpose → tick → key result)

### Theory & toys  (also SUMMARY S1–S13)
| script | purpose | tick | result |
|---|---|---|---|
| `hier_geom.py` | hierarchical expert-family geometry | 32 | J\|cos\| recovers 2-level tree, ARI 1.0 both levels (S10b) |
| `jacsae.py` | SAE on the Jacobian object, gated toy | 37 | restricted-J SAE → mechanism (gate-purity 0.965); only restricted J works (S19) |
| `jacsae2.py` | non-degenerate superposition-of-operators SAE | 38 | operator dictionary MMCS 0.926 in genuine superposition |

### Real bilinear MLP — clustering
| script | purpose | tick | result |
|---|---|---|---|
| `real_mlp_gtop.py` | G-top projection recipe on real block2 MLPs | 41 | NULL — `G_rand` control explains any lift; closes priority-1 |

### Attention (two-QK bilinear / squared, OV, signed, secant)
| script | purpose | tick | result |
|---|---|---|---|
| `ov_squared.py` | squared-attn cos² kernel + OV pathway | 26 | OV is an independent (WHERE×WHAT) axis |
| `ov_multi.py` | OV pathway across heads | 26 | — |
| `twoqk.py` | genuine two-QK bilinear, both query matrices | 27 | including both matrices helps on causal heads (~1.5–2.4×) |
| `twoqk_screen.py` | weights-only pre-screen for the two-QK gain | 28 | no weights-only predictor; effect head-dependent |
| `twoqk_ctrl.py` | matched-dim control for the two-QK gain | 28 | survives at 3/4 causal heads |
| `twoqk_ov.py` | OV / WHERE×WHAT on the 500M two-QK model | 29 | partial replication (model-dependent) |
| `twoqk_signed.py` | why two QK: signed + conjunctive attention | 30 | ~49% of attention mass subtractive; conjunctive sharpening universal |
| `twoqk_ablate.py` | causal ablation of signed attention | 31 | load-bearing at ~7 heads (L1H1 standout) |
| `twoqk_full.py` | full attention module as one tensor / VK read | 33 | contracted read is null; 16 MB factored vs 1 EB dense |
| `twoqk_ovk.py` | OVK factor readouts as clustering features | 34 | key readouts dilute; OV write helps modestly |
| `attn_secant.py` | the `y@x⁻¹` secant for attention | 35 | **unification**: query readout = gate-restricted Jacobian of the per-query bilinear op |
| `attn_jacblock.py` | attention-Jacobian contamination direction | 36 | J leans toward context/output (weak) |
| `attn_sae_real.py` | real-model attention SAE (query vs residual) | 39 | head-dependent (query wins 3/4) |

### Operator-SAE (bilinear-secant SAE) — the main real-model line
| script | purpose | tick | result |
|---|---|---|---|
| `bilinear_sae.py` | architecture + toy | 42 | works; loss-expansion exact to machine precision |
| `real_bilinear_sae.py` | block2 MLPs | 43 | non-null positive (FVU 0.22–0.35 vs 0.99 random); no outliers |
| `real_bilinear_sae_500m.py` | 500M scale | 44 | feasible at d=1152; layer-dependent (L6 strong) |
| `secant_lowrank_ctrl.py` | sparse vs dense low-rank control | 45 | sparse genuinely beats low-rank (not a low-rank artifact) |
| `secant_depth_sweep.py` | depth profile — **superseded by `depth_corrected.py`** | 46 | (rosy train-FVU + wrong hook) |
| `secant_honly.py` | analysis-vs-transcoder fork (does it need y?) | 47 | needs y exactly where operators are genuinely sparse |
| `dict_stability.py` | dictionary stability across seeds | 48 | stable/canonical (MMCS 0.58–0.79) |
| `dict_autointerp_data.py` | autointerp data (blind + random controls) → labeled by subagent | 49–50 | interpretable at L0 (token features); opaque deep |
| `sae_losscurve.py` | **corrections**: held-out FVU + true MLP-input hook | 51 | earlier FVUs were ~0.05 optimistic (train-on-12k) |
| `bsae_scaled.py` | BatchTopK + lottery-ticket (m-sweep) + k-sweep + complexity hist | 52 | lottery-ticket saturates; BatchTopK complexity split |
| `depth_corrected.py` | **faithful** depth profile (supersedes tick 46) | 53 | mean held-out FVU 0.405; two-regime split holds |
| `complexity_split.py` | what drives the complexity split | 54 | it's magnitude (ρ 0.94 with `‖M‖=‖y‖/√d`) |
| `magfree.py` | magnitude-free BatchTopK | 55 | split ~90% magnitude; small real directional residual |
| `transcoder_vs_secant.py` | transcoder vs secant, tied vs untied | 56 | transcoder wins on linear readout; **untying = Goodhart** (stability 0.71→0.43) |
| `full_mlp_arch.py` | best sparse dict for the full bilinear MLP | 57 | bilinear transcoder > linear; secant-cost scales with input dim |

## Artifacts

- `logs/` — raw stdout from runs (one per script; ephemeral, findings live in `results.md`).
- `figures/` — PNGs: `sae_losscurve`, `bsae_scaled` (lottery-ticket/k-sweep/hist), `depth_corrected`, `complexity_split`, `magfree`.
- `data/` — autointerp token lists (`autointerp_L{0,8,10}.json`).

## Models & data (top-level repo, outside `jacclust/`)

- `deep_model.py`, `model.py` — tensor-transformer definitions for the small in-repo checkpoints (`block2` etc.).
- `runs_owt/`, `runs_lm/` — trained small models (e.g. `runs_owt/block2-dense-seed0/model.pt`).
- `data_text/` — tokenized corpus (`val.bin`, `train.bin`, `tokenizer.json`, vocab 5120).
- The 124M/500M bilinear checkpoints are `Elriggs/gpt2-bilinear-*` on HuggingFace (auto-downloaded to the HF cache).

## Running

```
cd /workspace/tensor_language && source /venv/main/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python jacclust/<script>.py
```
Scripts add `/workspace/tensor_language` to `sys.path` and import `jacclust.<core-module>`; run from the repo root.

## Standing methodology rules (carried from mechdecomp — they reversed 5 headlines)

- Every clustering/dictionary claim needs a **control that could fail** (matched-dim random projection, spectrum-matched `G_rand`); chance measured, never assumed 0.
- No single-seed / top-k / max-over-sample statistics for high-variance metrics (≥5 seeds, report mean±sd). SAE FVU is smooth (±~0.003) so fewer seeds are noted where used.
- Verify every construction against a known identity before clustering with it.
- State confounds; report held-out (not train) FVU with the correct hook (see tick 51).
