"""Mechanistic analysis of the 2-layer bilinear models (cycle and grid tasks):
attention-offset profiles, bilinear factor decomposition, key-class statistics,
OV copy circuits in weight space, and causal wire ablations.

Usage: python mech.py   (writes figures/mech_*.png and prints all tables)
"""

import matplotlib.pyplot as plt
import torch
from einops import einsum, rearrange

import analysis
import analysis_geo
from analysis import DIVERGING, SECONDARY
from data import N_CTX as CYC_CTX
from data import eval_sets as cyc_evals
from data import sample_cycles
from geodata import N_CTX as GEO_CTX
from geodata import TAIL
from geodata import eval_sets as geo_evals
from geodata import walk_batch

torch.set_grad_enabled(False)
FIGURES = analysis.FIGURES


def heads(t, layer):
    return rearrange(t, "... (h d) -> ... h d", h=layer.n_head)


def factors(layer, xq, xk):
    """The two bilinear score factors, each scaled by d_head (their product is the pattern pre-mask)."""
    q1 = layer.rotary(heads(layer.q1(xq), layer))
    k1 = layer.rotary(heads(layer.k1(xk), layer))
    q2 = layer.rotary(heads(layer.q2(xq), layer))
    k2 = layer.rotary(heads(layer.k2(xk), layer))
    s1 = einsum(q1, k1, "... q h d, ... k h d -> ... h q k") / layer.d_head
    s2 = einsum(q2, k2, "... q h d, ... k h d -> ... h q k") / layer.d_head
    return s1[:, 0], s2[:, 0]


def pattern_from(layer, xq, xk):
    s1, s2 = factors(layer, xq, xk)
    return s1 * s2 * layer.mask[: xq.size(-2), : xq.size(-2)]


def layer2_out(l2, xq, xk, xv, base):
    p = pattern_from(l2, xq, xk)[:, None]
    v = heads(l2.v(xv), l2)
    z = rearrange(einsum(p, v, "... h q k, ... k h d -> ... q h d"), "... q h d -> ... q (h d)")
    return torch.lerp(base, l2.o(z), 0.5)


def ablated_logits(model, tokens, variant: str):
    """Forward pass with one wire cut. Variants: full, no-L1, no-L2, and
    'L2 {q,k,v} from embed-stream' (that input computed from the pre-L1 stream)."""
    l1, l2 = model.layers
    x = model.embed(tokens)
    x1 = x if variant == "no-L1" else l1(x)
    sources = {cut: (x if variant == f"L2 {cut} from embed-stream" else x1) for cut in "qkv"}
    out = x1 if variant == "no-L2" else layer2_out(l2, sources["q"], sources["k"], sources["v"], x1)
    return model.head(out)


VARIANTS = ("full", "no-L1", "no-L2", "L2 q from embed-stream", "L2 k from embed-stream", "L2 v from embed-stream")


def offset_profile(p, n_ctx, q_min, max_off=25):
    qs = torch.arange(q_min, n_ctx, device=p.device)
    return [p[:, qs, qs - o].mean().item() for o in range(max_off + 1)]


def ov_maps(model):
    """Token -> logit matrices along each path (coefficients from the 0.5-lerp stream)."""
    E, U = model.embed.weight, model.head.weight
    l1, l2 = model.layers
    OV1 = l1.o.weight @ l1.v.weight
    OV2 = l2.o.weight @ l2.v.weight
    return {
        "direct path": 0.25 * U @ E.T,
        "L1 → logits": 0.25 * U @ OV1 @ E.T,
        "L2 copies token": 0.25 * U @ OV2 @ E.T,
        "L2 copies L1-window": 0.125 * U @ OV2 @ OV1 @ E.T,
    }


def fig_ov(model, out: str, title: str):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (name, M) in zip(axes, ov_maps(model).items()):
        sub = M[:30, :30].cpu()
        v = sub.abs().max()
        ax.imshow(sub, cmap=DIVERGING, vmin=-v, vmax=v)
        d = M.diagonal()
        ax.set_title(f"{name}\ndiag {d.mean():+.2f} ± {d.std():.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"{title} — token→logit maps per path (first 30 tokens; red +, blue −; a colored diagonal = (anti-)copying)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)


def cycle_analysis():
    model = analysis.load_model("L2_d64_h1").cuda()
    l1, l2 = model.layers
    L = 7
    tokens = sample_cycles(256, torch.full((256,), L), generator=torch.Generator().manual_seed(5)).cuda()
    x = model.embed(tokens)
    x1 = l1(x)
    p1 = l1.pattern(x)[:, 0]
    p2 = pattern_from(l2, x1, x1)
    s1, s2 = factors(l2, x1, x1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    offs = range(0, 26)
    axes[0].bar(offs, offset_profile(p1, CYC_CTX, 30), color="#2a78d6")
    axes[0].set_title("layer 1 attention by offset t−s", fontsize=9)
    axes[1].bar(offs, offset_profile(p2, CYC_CTX, 30), color="#2a78d6")
    for k in (1, 2, 3):
        axes[1].axvline(k * L, color="#e34948", lw=0.8, ls=(0, (2, 2)))
    axes[1].set_title(f"layer 2 attention by offset (L={L}; red lines at kL)", fontsize=9)
    qs = torch.arange(30, CYC_CTX, device="cuda")
    axes[2].plot(offs, [s1[:, qs, qs - o].mean().item() for o in offs], color="#eda100", lw=1.8, label="factor 1 (envelope)")
    axes[2].plot(offs, [s2[:, qs, qs - o].mean().item() for o in offs], color="#2a78d6", lw=1.8, label="factor 2 (phase detector)")
    axes[2].axhline(0, color="#c3c2b7", lw=0.8)
    axes[2].legend(fontsize=8)
    axes[2].set_title("layer 2 bilinear factors by offset", fontsize=9)
    for ax in axes:
        ax.set_xlabel("offset t−s")
    fig.suptitle("Cycle model (2L·d64) — attention structure", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "mech_cycle.png", bbox_inches="tight")
    plt.close(fig)

    fig_ov(model, "mech_cycle_ov.png", "Cycle model")

    evals = {k: (t.cuda(), m.cuda()) for k, (t, m) in cyc_evals().items()}
    print("\ncycle ablations (accuracy):")
    print(f"{'variant':26s}" + "".join(f"  L={k:2d}" for k in evals))
    for variant in VARIANTS:
        accs = []
        for t, m in evals.values():
            preds = ablated_logits(model, t, variant)[:, :-1].argmax(-1)
            accs.append(((preds == t[:, 1:]) * m).sum().item() / m.sum().item())
        print(f"{variant:26s}" + "".join(f" {a:5.2f}" for a in accs))
    model.cpu()


def grid_analysis():
    model = analysis_geo.load_model("grid_L2_d128_long").cuda()
    l1, l2 = model.layers
    tokens, _, _ = walk_batch(128, (4, 4), "grid", torch.Generator().manual_seed(5))
    tokens = tokens.cuda()
    x = model.embed(tokens)
    x1 = l1(x)
    p1 = l1.pattern(x)[:, 0]
    p2 = pattern_from(l2, x1, x1).cpu()

    t = tokens.cpu()
    self_m = t[:, None, :] == t[:, :, None]
    succ_m = torch.zeros_like(self_m)
    succ_m[:, :, 1:] = t[:, None, :-1] == t[:, :, None]      # x_{s-1}=x_t -> s holds a successor
    pred_m = torch.zeros_like(self_m)
    pred_m[:, :, :-1] = t[:, None, 1:] == t[:, :, None]      # x_{s+1}=x_t -> s holds a predecessor
    base = torch.tril(torch.ones(GEO_CTX, GEO_CTX, dtype=torch.bool), -1)[None] & (torch.arange(GEO_CTX)[None, :, None] >= TAIL)
    classes = {
        "self-match\n(x_s = x_t)": base & self_m,
        "successor slot\n(x_{s−1} = x_t)": base & succ_m,
        "predecessor slot\n(x_{s+1} = x_t)": base & pred_m,
        "unrelated": base & ~self_m & ~succ_m & ~pred_m,
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    offs = range(0, 26)
    axes[0].bar(offs, offset_profile(p1, GEO_CTX, TAIL), color="#2a78d6")
    axes[0].set_title("layer 1 attention by offset t−s", fontsize=9)
    axes[0].set_xlabel("offset t−s")
    means = [p2[sel].mean().item() for sel in classes.values()]
    axes[1].bar(range(len(classes)), means, color=["#104281", "#2a78d6", "#5598e7", "#c3c2b7"])
    axes[1].set_xticks(range(len(classes)), list(classes), fontsize=8)
    axes[1].axhline(0, color="#c3c2b7", lw=0.8)
    axes[1].set_title("layer 2 mean attention by key-token relation", fontsize=9)
    fig.suptitle("Grid model (2L·d128) — attention structure", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "mech_grid.png", bbox_inches="tight")
    plt.close(fig)

    fig_ov(model, "mech_grid_ov.png", "Grid model")

    ev_tokens, legal = [z.cuda() for z in geo_evals("grid", n_seq=64)["in"]]
    print("\ngrid ablations (legal / mass):")
    for variant in VARIANTS:
        logits = ablated_logits(model, ev_tokens, variant)[:, TAIL:-1]
        lg = legal[:, TAIL:-1]
        rate = lg.gather(2, logits.argmax(-1, keepdim=True)).float().mean().item()
        mass = (logits.softmax(-1) * lg).sum(-1).mean().item()
        print(f"{variant:26s} {rate:6.3f} {mass:6.3f}")
    model.cpu()


if __name__ == "__main__":
    cycle_analysis()
    grid_analysis()
