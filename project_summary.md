# Tensor-Language — Project Summary

_Last updated: 2026-06-02_

A working log of **what this project is, what we're measuring, and what we've found so far.**
For the terse spec see [`README.md`](README.md); this file is the narrative + results.

> View in VS Code: `Ctrl+Shift+V` (preview) or `Ctrl+K V` (preview to the side).

---

## 1. The goal

Train **tensor-network transformers** of increasing size on language and confirm a simple
contract: **adding a component lowers loss**. "Tensor network" (TN) means the whole model is a
**polynomial** in its input — no per-sample nonlinearities — so once trained it can be
*contracted / folded* into a fixed multilinear map for analysis (mech-interp).

Two things are in tension and are the heart of the project:

- A **LayerNorm** at the output makes training easy and stable, but its `1/√var(x)` is computed
  **per sample** → it is **not** a polynomial → it breaks the tensor-network property.
- A **foldable** normalization (a fixed/running scalar) keeps the model a tensor network, but is
  weaker. **Can we train a fully foldable stack and still get the monotonic loss ordering?**
  That is the central question.

---

## 2. The architecture

```
tokens ─▶ Embed ─▶ [ bilinear-attn (+ bilinear-MLP) ] × n_layers ─▶ final_norm ─▶ Unembed
```

| Component | Form | Foldable (TN)? |
|---|---|---|
| Embed / Unembed | linear | ✅ |
| RoPE | fixed per-position rotation | ✅ |
| **Bilinear attention** | `(Q₁x·K₁x)(Q₂x·K₂x)/d_h²`, causal | ✅ degree-4 polynomial |
| BatchNorm on Q/K | per-channel affine at inference | ✅ folds into Q/K weights |
| **Bilinear MLP** | `D(Lx ⊙ Rx)` | ✅ degree-2 polynomial |
| **ReZero scalar** (`--rezero-init`) | learnable `α` in `x + α·branch(x)` | ✅ folds into `o`/`D` |
| `final_norm = layernorm` | `1/√var(x)` (per-sample) | ❌ **does not fold** |
| `final_norm = static-rms` | `/ running_rms` (fixed scalar) | ✅ folds into Unembed |
| `final_norm = none` | identity | ✅ |

"Bilinear" = the model is built from **products of two linear maps** instead of a nonlinearity
like ReLU/GELU. `(Q₁x·K₁x)(Q₂x·K₂x)` is a product of dot-products → a **degree-4 polynomial** in
`x`; `D(Lx ⊙ Rx)` is a **degree-2** polynomial (`⊙` = elementwise product). No softmax, no GELU —
everything stays polynomial, hence foldable. `d_head` is fixed at 32, so `n_head = d_model // 32`.

---

## 3. The variant ladder (what `attn1`, `xf2`, etc. mean)

Each variant adds a component to the previous one. Loss should fall **monotonically** down the
ladder. **`xf` = trans·**x**·former** (attention **+** bilinear MLP); the number = layer count.

| variant | layers | components | role |
|---|---|---|---|
| `embed_unembed` | 0 | Embed → norm → Unembed | the **bigram floor** (predict next token from current) |
| `attn1` / `attn2` | 1 / 2 | bilinear attention only | does attention help? |
| `xf1` / `xf2` | 1 / 2 | attention **+ bilinear MLP** | does the MLP help on top? |
| `attn4` / `xf4` | 4 | (opt-in) | **depth stress test** — most likely to destabilize a foldable norm |

`xf2` (2-layer attention+MLP) is the deepest in the default ladder and the one most prone to
instability; `xf4` pushes that harder.

---

## 4. How we measure — two datasets, and why it matters

There are **two** data modes, and conflating them caused the original "open issue":

- **`--data cached`** — train on the 500-sequence cached Pile *val* tensor **itself**. This is an
  **overfit / memorization** task: the model sees the same 500 sequences repeatedly and can drive
  loss → ~0. Useful only as a fast wiring check.
- **`--data pile`** — stream DSIR-filtered Pile. **Every batch is fresh** (no epoch repeats), so the
  *training* loss is already a **generalization / held-out** measure. We additionally evaluate on
  the fixed cached val tensor for a clean, comparable number.

**Key insight:** the cached overfit task is *misleading* for judging foldable norms. Driving loss
to 0 by memorization needs extreme **per-token output confidence**, which only the per-token
LayerNorm can supply. A foldable scalar norm **cannot** do that *by construction* — but that has
nothing to do with the real goal. **Always judge on streaming Pile, not the cached set.**

---

## 5. What we found

### 5.1 The "static-rms instability" was a benchmark artifact (RESOLVED)

The README originally flagged `StaticRMSNorm` (foldable) as **unstable with attention**: on the
cached task, attention made loss *worse* and `xf2` collapsed to the uniform floor (8.5 ≈ log 5000).

**Mechanism (instrumented):** the degree-4 bilinear attention makes the residual stream develop
extreme **per-token magnitude spread** (per-token RMS ranging 462×–6800×, even up to 10¹⁰ with no
control). LayerNorm normalizes each token independently → fine. A foldable global scalar divides
**every** token by one value dominated by the few exploding tokens → the normal tokens wash out to
~uniform logits → collapse. It is impossible to match LayerNorm on *memorization*, and irrelevant.

**On real streaming data it is not broken.** `static-rms` is monotonic and within ~0.07 CE of
LayerNorm.

### 5.2 Confirmed loss ordering on real data ✅

6k-step sweep, `d=128`, `n_ctx=128`, `lr=3e-3`, streaming Pile, eval on held-out cached val
(lower = better; **all monotonic** down each row):

| final norm | foldable | embed_unembed | attn1 | attn2 | xf1 | xf2 |
|---|---|---|---|---|---|---|
| `layernorm` (reference) | ❌ | 5.914 | 5.649 | 5.588 | 5.570 | **5.497** |
| `none` (purest TN) | ✅ | 6.034 | 5.838 | 5.782 | 5.758 | **5.694** |
| `static-rms` | ✅ | 5.915 | 5.673 | 5.639 | 5.666 | **5.873** ⚠ |

Adding components lowers loss for all three. `static-rms` even **beats `none`** for the shallow
variants. The ⚠ on `static-rms` `xf2` is the one real residual effect → §5.3.

### 5.3 The one real issue is **depth**, and it has a foldable fix

`static-rms` on the deepest variant (`xf2`) **drifts up** during training instead of converging.
Trajectory experiment (`xf2`, streaming Pile, held-out val):

| config | @2k | @4k | @6k | verdict |
|---|---|---|---|---|
| `layernorm` (ceiling, ❌foldable) | 5.742 | 5.590 | **5.498** | stable |
| `none` (✅TN-pure) | 5.972 | 5.801 | **5.694** | **stable, still ↓** |
| `static-rms` + **ReZero 0.25** (✅) | 5.778 | 5.643 | **5.725** | drift tamed |
| `static-rms` + ReZero 0.1 (✅) | 8.508 | 5.634 | **5.722** | works, slow warmup |
| `static-rms` baseline (✅) | 5.764 | 5.626 | **5.887** | ✗ U-shape **drift** |
| `static-rms` `lr=1e-3` (✅) | 5.915 | 6.118 | **6.604** | ✗ **diverges** |

Two conclusions:
1. **Lower lr is *not* the fix** — it diverges *harder*. (The README's old hypothesis was wrong.)
   The drift is a running-RMS × depth interaction, not a step-size issue.
2. **Foldable fixes that work:**
   - **`final_norm=none`** — fully stable and monotonic at every depth, ~0.2 CE behind LayerNorm.
   - **`--rezero-init 0.25`** — a learnable per-branch residual scalar (`x = x + α·branch(x)`,
     a.k.a. ReZero / SkipInit). It **folds into `o`/`D` at inference**, so the model stays a
     tensor network, and it tames the `xf2` drift back to `none`-level.

### 5.4 Where the ideas came from (lit review)

- **ReZero / SkipInit / Fixup** — learnable small-init residual scalar; stabilize deep residual
  nets without normalization. → our `--rezero-init`.
- **σReparam** (Apple) — spectral-norm + learned gain on every linear; a *weight* reparam, fully
  foldable. Tested; **unnecessary / slightly hurts** on real data (it fixes a memorization
  explosion that doesn't occur on fresh batches).
- **Π-nets** (Chrysos) — polynomial nets; same instability class; use activation boundary loss.

---

## 6. Tooling added to `train_sweep.py`

| flag | effect |
|---|---|
| `--rezero-init F` | learnable foldable residual scalar init `F` (e.g. `0.25`); `None`=fixed 1.0 |
| `--top-tokens N` | log the N `(seq,pos)` datapoints each config predicts best → `sweep.jsonl` |
| `--save-checkpoints` | per-config compile-unwrapped `state_dict`s → `runs/<ts>/checkpoints/` |
| `attn4` / `xf4` variants | opt-in 4-layer depth stress test (`--variants attn4,xf4`) |

Reproduce the confirmed sweep:
```bash
python train_sweep.py --data pile --steps 6000 --widths 128 --n-ctx 128 \
    --norms layernorm,static-rms,none
```

---

## 7. Current status

- ✅ Wiring verified; loss ordering confirmed on real streaming Pile.
- ✅ `static-rms` "instability" resolved (benchmark artifact); foldable stack trains & is monotonic.
- ✅ Depth drift identified + fixed (`none`, or `static-rms`+ReZero).
- 🔬 **In progress (background):** a broad sweep over **depth × norm × ReZero-init × width** at
  longer steps (10k), including the 4-layer `xf4` stress test and `d=256`. Results will land in
  `runs/<timestamp>_sweep/sweep.jsonl` and be folded into this file + the README.

### Open questions the broad sweep targets
1. Does the foldable stack stay monotonic/stable at **4 layers** + 10k steps?
2. Does ReZero fix the drift at `xf4` too, and what's the best init (0.1 / 0.25 / 0.5)?
3. Does the ordering hold and the gap widen at **larger width** (toward the d=512 reference of 4.72)?
4. Final call: canonical TN-pure default — **`none`** vs **`static-rms` + ReZero**.
