"""Causal tests of copy-as-organizer INSIDE the task family (user's session-4 ideas
2 and 3), against the reliably-anti grid+dring pairing:

  - grid-selfloop (gridSL): every grid node gets a self-edge, so uniform walks
    repeat the current node with prob 1/deg. Repeat-of-current is the burst
    ingredient placed inside the family itself.
  - grid-backtrack-2x (gridBT2): transition matrix puts weight 2 on returning to
    the previous node, weight 1 on other neighbors (dose-increase of reversibility;
    walks still only traverse real edges).

Also trains the missing softmax-add-3L grid+dring BASELINE (bilinear baseline
-0.55/-0.70/-0.72 from earlier sessions).

Pre-registered:
  P6: gridSL+dring kills the anti mode for both archs (org >= +0.2).
  P7: gridBT2+dring moves positive relative to baseline but more weakly than P6
      (possibly still <= 0 given dring's pull).

Measured on STANDARD grid walks (mean_reps protocol, unchanged for comparability);
legal rate reported on standard walks AND on the training variant's own walks.

Usage: python toy_recur.py
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from graphs import N_VOCAB, N_CTX, MAX_NODES, MAX_DEG, TRAIN_POOLS, walk_pool, legal_tokens
from model import CycleModel


def add_self_loops(pool):
    out = []
    for nb, deg in pool:
        nb2, deg2 = nb.clone(), deg.clone()
        n = int((nb[:, 0] >= 0).sum())
        for v in range(n):
            nb2[v, deg[v]] = v
        deg2[:n] += 1
        out.append((nb2, deg2))
    return out


SL_POOL = add_self_loops(TRAIN_POOLS["grid"])


def walk_pool_bt(pool, n_seq, gen, boost=2.0):
    """walk_pool with the previous node's transition weight multiplied by `boost`."""
    pick = torch.randint(0, len(pool), (n_seq,), generator=gen)
    nb = torch.stack([pool[i][0] for i in pick.tolist()])
    deg = torch.stack([pool[i][1] for i in pick.tolist()])
    n_nodes = (nb[:, :, 0] >= 0).sum(1)
    perm = torch.rand(n_seq, N_VOCAB, generator=gen).argsort(1)[:, :MAX_NODES]
    position = torch.randint(0, MAX_NODES, (n_seq,), generator=gen) % n_nodes
    rows = torch.arange(n_seq)
    prev = position.clone()
    trail = [position]
    for step in range(N_CTX - 1):
        w = (torch.arange(MAX_DEG)[None, :] < deg[rows, position][:, None]).float()
        if step > 0:
            w = w * torch.where(nb[rows, position] == prev[:, None], boost, 1.0)
        choice = torch.multinomial(w, 1, generator=gen).squeeze(1)
        prev, position = position, nb[rows, position, choice]
        trail.append(position)
    nodes = torch.stack(trail, 1)
    return perm.gather(1, nodes), nodes, perm, pick


GRID_BATCHES = {
    "grid": lambda n, g: walk_pool(TRAIN_POOLS["grid"], n, g),
    "gridSL": lambda n, g: walk_pool(SL_POOL, n, g),
    "gridBT2": lambda n, g: walk_pool_bt(TRAIN_POOLS["grid"], n, g),
}
LEGAL_POOLS = {"grid": TRAIN_POOLS["grid"], "gridSL": SL_POOL, "gridBT2": TRAIN_POOLS["grid"]}


def legal_rate(model, variant, seed=999):
    g = torch.Generator().manual_seed(seed)
    toks, nodes, perm, pick = GRID_BATCHES[variant](128, g)
    pool = LEGAL_POOLS[variant]
    nb = torch.stack([pool[i][0] for i in pick.tolist()])
    legal = legal_tokens(nodes, perm, nb)
    with torch.no_grad():
        lg = model(toks[:, :-1])
    hit = legal[:, :-1].gather(-1, lg.argmax(-1, keepdim=True)).squeeze(-1)
    return hit[:, 128:].float().mean().item()


if __name__ == "__main__":
    device = "cuda"
    results = {}
    dring_pool = TRAIN_POOLS["dring"]
    CONFIGS = (
        ("softmax-add-3L", dict(attention="softmax", residual="add"), 3, "grid"),     # baseline
        ("softmax-add-3L", dict(attention="softmax", residual="add"), 3, "gridSL"),
        ("bilin-lerp-2L", dict(attention="bilinear", residual="lerp"), 2, "gridSL"),
        ("softmax-add-3L", dict(attention="softmax", residual="add"), 3, "gridBT2"),
        ("bilin-lerp-2L", dict(attention="bilinear", residual="lerp"), 2, "gridBT2"),
    )
    for arch, kwargs, n_layer, variant in CONFIGS:
        seed = 0
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 500)
        name = f"{arch}-{variant}+dring-seed{seed}"
        if (Path("runs_gen") / name / "model.pt").exists():
            print(f"skip {name} (exists)", flush=True)
            continue
        model = CycleModel(N_VOCAB, 128, 1, n_layer, N_CTX, **kwargs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        steps = 24000
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / 100, 0.5 * (1 + math.cos(math.pi * s / steps))))
        for step in range(steps):
            gt = GRID_BATCHES[variant](63, gen)[0]
            dr, _, _, _ = walk_pool(dring_pool, 63, gen)
            tokens = torch.cat([gt, dr]).to(device)
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, N_VOCAB), tokens[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % 6000 == 0:
                print(f"{name} step {step} loss {loss.item():.3f}", flush=True)
        out = Path("runs_gen") / name
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        (out / "history.json").write_text(json.dumps({"config": {
            "d_model": 128, "n_layer": n_layer, "scale": 0.5, "norm": False,
            "variant": variant, **kwargs}}))

        model.cpu().eval()
        from icl_reps import mean_reps, gram_adjacency_corr, adjacency, pc_spectrum_alignment
        torch.set_grad_enabled(True)   # icl_reps disables grads globally at import
        reps = mean_reps(model, "grid")
        A = adjacency("grid")
        c8, c256 = gram_adjacency_corr(reps[8], A), gram_adjacency_corr(reps[256], A)
        var, corr = pc_spectrum_alignment(reps[256], "grid")
        std_rate = legal_rate(model, "grid")
        train_rate = std_rate if variant == "grid" else legal_rate(model, variant)
        print(f"== {name}: grid org ctx8 {c8:+.2f} ctx256 {c256:+.2f}  "
              f"legal(std) {std_rate:.2f} legal(train-dist) {train_rate:.2f}  "
              f"PC12-harmonic corr {corr[0]:.2f}/{corr[1]:.2f}", flush=True)
        results[name] = {"org8": c8, "org256": c256, "legal_std": std_rate,
                         "legal_train": train_rate, "pc_corr": corr[:4], "pc_var": var[:4]}
        model.to(device)
        Path("runs_gen/recur_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
