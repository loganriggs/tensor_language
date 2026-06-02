# tensor_language

Training tensor-network transformers of various sizes on language & toy languages.

Every architecture here stays a **tensor network** (polynomial / foldable) so it can be
contracted after training. The goal: build a clean ladder of models — from a 0-layer
bigram up to a 2-layer bilinear transformer — confirm that **adding components lowers
loss**, then find the datapoints each setting learns best.

## Architecture

```
tokens ─▶ Embed ─▶ [ bilinear-attn (+ bilinear-MLP) ] × n_layers ─▶ final_norm ─▶ Unembed
```

| Component | Form | Tensor-network status |
|---|---|---|
| Embed / Unembed | linear | ✅ |
| RoPE | fixed per-position rotation | ✅ |
| Bilinear attention | `(Q₁x·K₁x)(Q₂x·K₂x)/d_h²`, causal | ✅ degree-4 polynomial |
| BatchNorm on Q/K | per-channel affine at inference | ✅ folds into Q/K weights |
| Bilinear MLP | `D(Lx ⊙ Rx)` | ✅ degree-2 polynomial |
| ReZero scalar (`--rezero-init`) | learnable `α` in `x + α·branch(x)` | ✅ folds into `o`/`D` |
| `final_norm=layernorm` | `1/√var(x)` (per-sample) | ❌ does **not** fold |
| `final_norm=static-rms` | `/ running_rms` (fixed scalar) | ✅ folds into Unembed |
| `final_norm=none` | identity | ✅ |

`d_head` is fixed at 32, so `n_head = d_model // 32`.

## Sweep (`train_sweep.py`)

Variants (components added left→right — loss should fall monotonically):

| variant | layers | components |
|---|---|---|
| `embed_unembed` | 0 | Embed→norm→Unembed (**bigram** floor — predicts next token from current) |
| `attn1` / `attn2` | 1 / 2 | bilinear attention |
| `xf1` / `xf2` | 1 / 2 | bilinear attention + bilinear MLP |

Swept over `--widths` (d_model) and `--norms` (`layernorm,static-rms,none`).

```bash
python train_sweep.py --smoke                       # tiny wiring check, cached data, ~seconds
python train_sweep.py --data cached --steps 1500 --lr 3e-3 --no-compile   # overfit ordering demo
python train_sweep.py --steps 6000 --widths 128 --norms layernorm,static-rms,none   # real Pile sweep
python train_sweep.py --data pile --steps 6000 --top-tokens 10 --save-checkpoints    # + interp outputs
```

- `--top-tokens N` logs, per config, the N `(seq, pos)` datapoints with the lowest next-token CE
  (what each setting learns best) into `runs/<ts>_sweep/sweep.jsonl`.
- `--save-checkpoints` writes per-config `state_dict`s (torch.compile-unwrapped) to
  `runs/<ts>_sweep/checkpoints/<variant>_d<width>_<norm>.pt` for later mech-interp.

- Optimizer defaults to **AdamW** (`--muon` opt-in; `muon` pkg not installed). Pure-AdamW
  needs a higher lr than the legacy scripts' `3e-4` (that was the AdamW *aux* rate while
  Muon drove attention at `0.02`). Use `--lr 1e-3`–`3e-3`.
- `--data cached` trains on the 500-seq Pile val tensor itself (wiring/overfit checks only).
  `--data pile` streams DSIR-filtered Pile (needs `datasets`+`transformers`+network).

## Status (2026-06-02)

Wiring **verified correct** on cached data, and the loss ordering is now **confirmed on real
streaming Pile** (every streamed batch is fresh → already a held-out/generalization measure;
eval is on the fixed `dsir_pile_val_ctx512.pt` tensor).

**A TN-pure (foldable) stack trains and is monotonic.** 6k-step sweep, `d=128`, `n_ctx=128`,
`lr=3e-3` (streaming Pile, eval on held-out cached val):

| final norm | foldable? | embed_unembed | attn1 | attn2 | xf1 | xf2 |
|---|---|---|---|---|---|---|
| `layernorm` (reference) | ❌ | 5.914 | 5.649 | 5.588 | 5.570 | **5.497** |
| `none` (purest TN) | ✅ | 6.034 | 5.838 | 5.782 | 5.758 | **5.694** |
| `static-rms` | ✅ | 5.915 | 5.673 | 5.639 | 5.666 | **5.873** ⚠ |
| `static-rms --rezero-init 0.25` | ✅ | — | — | — | — | **5.725** |

Reproduce: `python train_sweep.py --data pile --steps 6000 --widths 128 --n-ctx 128 --norms layernorm,static-rms,none`

### Resolved — the `static-rms` "instability" was a cached-overfit artifact
The earlier report (`static-rms` collapses to the uniform floor with attention) only happens
on the **cached-overfit** task. That task rewards *memorization*: driving loss → 0 needs extreme
per-token output confidence, which **only per-token LayerNorm** can supply — a foldable global
scalar divides every token by one value dominated by the few exploding tokens, washing the rest
to uniform logits. It is impossible to match LayerNorm there *by construction*, and irrelevant:
on fresh streaming data (no memorization) `static-rms` is monotonic and within **~0.07 CE** of
LayerNorm. **Don't judge foldable norms on the cached set — use streaming Pile.**

The one real residual effect is **depth**: the deepest variant (`xf2` = 2-layer + bilinear MLP)
under `static-rms` slowly *drifts up* during training (U-shaped: 5.63 @4k → 5.87 @6k). It is
*not* a tuning fix — lower lr diverges harder (6.60 @ `lr=1e-3`). The clean fixes are:
- **`final_norm=none`** — fully stable and monotonic at every depth (purest tensor network), ~0.2 CE behind LayerNorm.
- **`--rezero-init 0.25`** — a learnable per-branch residual scalar (`x = x + α·branch(x)`, ReZero/SkipInit).
  It folds into `o`/`D` at inference (stays TN-pure) and tames the `xf2` drift to ~`none` level.
  `static-rms` still edges out `none` for the *shallow* variants, so `static-rms + rezero` is the
  best all-round foldable default.

### Next steps
1. ~~Fix `StaticRMSNorm` stability~~ / ~~real Pile run~~ / ~~log `most_learned_tokens()`~~ /
   ~~checkpoint saving~~ — **done** (see table; `--top-tokens`, `--save-checkpoints`, `--rezero-init`).
2. Scale up: widths `128,256,512` and longer steps to widen the variant gaps (they're ~0.1–0.4
   apart at 6k/d128) and approach the d=512 reference val of 4.72.
3. Confirm `static-rms + rezero` keeps its shallow-variant edge across a full ladder, then pick
   the canonical TN-pure default (`none` vs `static-rms+rezero`).
4. Use the saved checkpoints + `top-tokens` for the mech-interp pass (which datapoints each
   component learns first).

## Provenance
Consolidated from `tensor-mars/workspaces/logan/`: self-contained bilinear+BatchNorm attention
(`train_bilinear_bn_nobias_ctx512_d512.py`, the only full-5B Pile run, val 4.72) + `BilinearMLP`
(`train_2layer_transformer.py`). `cached_tokens/dsir_pile_val_ctx512.pt` copied from there.
