# Polynomial-softmax experiments: attention + normalization

Two normalizers from the spherical-loss family, used as drop-ins for (1) the attention
softmax and (2) RMSNorm. Goal: train small GPTs, log cross-entropy vs step, see if either
variant matches or beats the baseline. Everything below is verified in numpy for correctness
of normalization, causal masking, and scaling; torch versions are 1:1.

> **Runnable harness:** `poly_softmax_gpt.py` implements all of this on char-Shakespeare —
> `python poly_softmax_gpt.py --experiment both`. It runs Exp 1 (3 attention variants, RMSNorm
> kept) and Exp 2 (3 norm variants, softmax attention kept), logs train+val CE, and writes
> `exp1_attention.png` / `exp2_norm.png`. One component changes per run; seed/data/init/schedule fixed.

## The two normalizers

For a score/activation vector `z` along the normalized axis:

- Taylor softmax: `num = 1 + z + z**2/2`, then `num / num.sum()`.
  Numerator is the 2nd-order Taylor of `exp`; always > 0 (discriminant 1 - 2 < 0).
  Rational in `z` (numerator polynomial, scalar-inverse denominator), so it stays on the
  "rational" rung: one scalar inverse edge, no `exp`.
- Spherical softmax: `num = z**2`, then `num / num.sum()`.
  Always >= 0 but **loses sign** (`z` and `-z` map to the same weight).

```python
import torch

def poly_softmax(z, dim=-1, kind="taylor", keep=None, eps=1e-6):
    """Spherical-family softmax. keep: optional 0/1 float mask, SAME shape broadcastable
    to z, applied multiplicatively (NOT via -inf, since these are not exp-based)."""
    if kind == "taylor":
        num = 1.0 + z + 0.5 * z * z          # > 0 everywhere
    elif kind == "spherical":
        num = z * z                          # >= 0, sign lost
    else:
        raise ValueError(kind)
    if keep is not None:
        num = num * keep
    denom = num.sum(dim=dim, keepdim=True).clamp_min(eps)
    return num / denom
```

## Experiment 1: replace the attention softmax

In a nanoGPT-style `CausalSelfAttention.forward`, the manual-attention branch is:

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
```

Replace the last two lines with a multiplicative keep-mask (do NOT use `-inf`; these
numerators are non-negative, so masked entries must be zeroed before normalizing):

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
if self.attn_kind == "softmax":
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
else:
    keep = self.bias[:, :, :T, :T]           # 1 lower-tri incl diagonal, 0 future
    att = poly_softmax(att, dim=-1, kind=self.attn_kind, keep=keep)
att = self.attn_dropout(att)
y = att @ v
```

Disable flash attention for the non-softmax variants (`self.flash = False`), since
`scaled_dot_product_attention` hard-codes the exp softmax.

Runs: `attn_kind in {"none", "softmax" (baseline), "taylor", "spherical"}`, RMSNorm kept
everywhere, **single Q·K pair, no QK normalization**.
- `none` = raw masked scores straight into `@v` with **no weight normalization** (`att = att * keep`).
  This is what the bilinear architecture does by default, so it's the reference for "does any
  softmax-like normalization help at all here?"

## Experiment 2: replace RMSNorm

Softmax-as-norm sends the hidden vector to the simplex (sums to 1 over the feature dim),
so raw values are ~1/D. You MUST rescale to restore ~unit RMS or the signal and gradients
vanish (verified: rms ~ 1/D before scaling, ~1 after multiplying by D). This scale is the
make-or-break knob; sweep it.

```python
import torch.nn as nn

class SoftmaxNorm(nn.Module):
    """Drop-in for RMSNorm. Applies a spherical-family softmax across the feature dim,
    then rescales. Output is non-negative and simplex-shaped before the gain."""
    def __init__(self, dim, kind="taylor", scale=None):
        super().__init__()
        self.kind = kind
        self.weight = nn.Parameter(torch.ones(dim))      # per-feature gain, like RMSNorm
        # softmax output ~1/dim; multiply by dim to land near unit RMS. Make it a learnable
        # scalar so the network can retune; init at dim.
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(dim if scale is None else scale))))

    def forward(self, x):
        p = poly_softmax(x, dim=-1, kind=self.kind)       # (..., D), sums to 1 over D
        return self.log_scale.exp() * p * self.weight
```

Swap every `RMSNorm(dim)` (the two per block in pre-norm, plus the final norm before
`lm_head`) for `SoftmaxNorm(dim, kind=norm_kind)`.

Runs: `norm_kind in {"rmsnorm" (baseline), "taylor", "spherical"}`, standard softmax
attention kept.

## Caveats that decide whether this trains (not optional)

- Attention, sign loss. Spherical squares the score, so a strongly negative score (meant to
  suppress a key) becomes a LARGE weight. Demonstrated: scores [2, 0.1, -3] -> spherical
  weights [0.31, 0.001, 0.69], i.e. it attends hardest to the token you wanted masked.
  Expect spherical attention to train poorly; include it mainly as the negative control.
- Attention, Taylor monotonicity. `1 + z + z**2/2` is monotonic only for z > -1; below that
  the quadratic term re-inflates. At z = -3 it returns 2.5, ranking that above z = 0.1.
  Keep scores in roughly [-1, inf): the 1/sqrt(d) scaling mostly does this. If attention
  refuses to sharpen, add a learnable temperature on the scores or clamp the low tail.
- Norm scale. Without the `*D`-ish rescale, activations are ~1/D and the net is dead. If
  unstable, sweep the init scale over {sqrt(D), D, 2D} or freeze `log_scale`.
- Norm sign. Output is non-negative (kills negative activations). In pre-norm this only
  constrains the *block input*; the residual stream stays full-range, so it's survivable.
- Learning rate. Like square-loss-vs-CE (Hui & Belkin), these may want a different LR than
  the softmax/RMSNorm baseline. If a variant looks bad, retune LR before concluding.
- Keep the comparison fair: same seed, data, init, schedule; change only the one component.

## Logging + plot

Record train and val CE at each eval and plot all runs of an experiment on shared axes.

```python
import matplotlib.pyplot as plt

# during training, per eval_interval:
#   history[label]["step"].append(iter_num)
#   history[label]["train"].append(train_ce)   # mean CE over a train batch
#   history[label]["val"].append(val_ce)

def plot_ce(history, title, out_png):
    plt.figure(figsize=(7, 4.5))
    for label, h in history.items():
        plt.plot(h["step"], h["val"], label=f"{label} (val)")
        plt.plot(h["step"], h["train"], ls="--", alpha=0.5, label=f"{label} (train)")
    plt.xlabel("step"); plt.ylabel("cross-entropy loss")
    plt.title(title); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150)

# plot_ce(exp1_history, "Attention softmax variants", "exp1_attention.png")
# plot_ce(exp2_history, "Norm variants", "exp2_norm.png")
```

Note: the reported CE is the standard softmax-CE eval metric for ALL runs (so the curves are
comparable), even though Experiment 1's variants change how attention weights are computed
and Experiment 2's variants change normalization. The training objective stays softmax-CE
throughout; we are only swapping internal components, not the loss.

## Suggested setup for a fast signal

nanoGPT char-level Shakespeare (`data/shakespeare_char`), then a small config:
`n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64, max_iters=5000,
eval_interval=250, lr=1e-3 with warmup+cosine`. Three runs per experiment (~15-20 min each
on a single modern GPU). If signal is promising, scale block/iters up. Add config fields
`attn_kind` (to `CausalSelfAttention`) and `norm_kind` (to wherever norms are built), thread
them from the top-level config, and launch the six runs.
