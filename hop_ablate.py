"""Causal head-recruitment check on the k-hop task (user's suggestion): for each hop
category, ablate one attention head at a time and measure the accuracy hit. If a
depth-gated category (e.g. hop-2) loses accuracy when any of MORE THAN TWO heads is
removed, that category genuinely recruits more than the 2-head induction circuit.

Zero-ablation of a head = zero its OV output slice (input to o) before it writes to the
residual stream. Reports per-hop accuracy for baseline and for each single-head ablation.

Usage: python hop_ablate.py attn-mlp-attn-seed0
"""

import json
import sys
from pathlib import Path

import torch

from deep_model import DeepModel
from hop_data import sample_docs, score_by_hop, V, N_CTX, K_MAX

D_MODEL = 128
N_HEAD = 4
D_HEAD = D_MODEL // N_HEAD


def load(name):
    cfg = json.loads(Path(f"runs_hop/{name}/config.json").read_text())
    model = DeepModel(V, D_MODEL, N_HEAD, cfg["spec"], N_CTX, norm=cfg.get("norm", False),
                      attention="bilinear", residual="lerp", mlp_residual="add")
    model.load_state_dict(torch.load(f"runs_hop/{name}/model.pt", map_location="cpu"))
    model.eval()
    return model, cfg


def head_hook(head):
    def hook(module, args):
        (z,) = args
        z = z.clone()
        z[..., head * D_HEAD:(head + 1) * D_HEAD] = 0
        return (z,)
    return hook


def acc_of(model, tok, qa, qk):
    with torch.no_grad():
        logits = model(tok[:, :-1])
    return score_by_hop(logits, qa, qk)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "attn-mlp-attn-seed0"
    model, cfg = load(name)
    attn_layers = [i for i, k in enumerate(cfg["spec"]) if k == "attn"]
    g = torch.Generator().manual_seed(777)
    tok, qa, qk = sample_docs(512, g)

    base = acc_of(model, tok, qa, qk)
    print(f"{name}  spec={cfg['spec']}")
    print(f"baseline acc by hop: {[round(base[k],3) for k in range(K_MAX+1)]}")
    rows = []
    for li in attn_layers:
        for h in range(N_HEAD):
            handle = model.layers[li].o.register_forward_pre_hook(head_hook(h))
            a = acc_of(model, tok, qa, qk)
            handle.remove()
            drop = {k: base[k] - a[k] for k in range(K_MAX + 1)}
            rows.append({"layer": li, "head": h, "acc": a, "drop": drop})
            print(f"  ablate L{li}H{h}: acc {[round(a[k],3) for k in range(K_MAX+1)]}  "
                  f"drop {[round(drop[k],3) for k in range(K_MAX+1)]}")
    # per-hop: how many heads are load-bearing (drop > 0.1)?
    print("\nload-bearing heads per hop (single-ablation drop > 0.10):")
    for k in range(K_MAX + 1):
        lb = [f"L{r['layer']}H{r['head']}" for r in rows if r["drop"][k] > 0.10]
        print(f"  hop {k}: {len(lb)} heads  {lb}")
    Path("runs_hop/ablate_" + name + ".json").write_text(
        json.dumps({"baseline": base, "rows": rows}, indent=1))
