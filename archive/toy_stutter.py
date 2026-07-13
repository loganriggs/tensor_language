"""Dose-response confirmation of the lag-1 finding (session 4).

The self-loop vs backtrack dissociation showed organization tracks the LAG-1
adjacent-repeat rate (grid 0.00 -> anti; gridSL 0.26 -> positive; gridBT2 0.00 with
51% lag-2 -> anti). This isolates that variable: a pure STUTTER that duplicates the
current node token with probability p WITHOUT touching the graph (no self-edges), on
grid+dring. p=0.0 reproduces the -0.77 anti baseline as an internal control.

Prediction: grid organization rises monotonically with p (the amount of lag-1
adjacent repetition), for the softmax stack that otherwise anti-organizes.

Stutter mechanics: emit a length-256 sequence of node tokens; at each step, with
prob p repeat the previous emitted token (graph position unchanged), else take a real
uniform neighbor step. The GRAPH is the plain grid, so mean_reps measurement on pure
grid walks is unchanged and directly comparable to every earlier run.

Usage: python toy_stutter.py
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from graphs import N_VOCAB, N_CTX, MAX_NODES, TRAIN_POOLS, walk_pool, legal_tokens
from model import CycleModel


def stutter_batch(pool, n_seq, gen, p):
    pick = torch.randint(0, len(pool), (n_seq,), generator=gen)
    nb = torch.stack([pool[i][0] for i in pick.tolist()])
    deg = torch.stack([pool[i][1] for i in pick.tolist()])
    n_nodes = (nb[:, :, 0] >= 0).sum(1)
    perm = torch.rand(n_seq, N_VOCAB, generator=gen).argsort(1)[:, :MAX_NODES]
    position = torch.randint(0, MAX_NODES, (n_seq,), generator=gen) % n_nodes
    rows = torch.arange(n_seq)
    trail = [position]
    for _ in range(N_CTX - 1):
        choice = (torch.rand(n_seq, generator=gen) * deg[rows, position]).long()
        step = nb[rows, position, choice]
        repeat = torch.rand(n_seq, generator=gen) < p
        position = torch.where(repeat, position, step)     # stay (duplicate token) or move
        trail.append(position)
    return perm.gather(1, torch.stack(trail, 1))


if __name__ == "__main__":
    device = "cuda"
    results = {}
    grid_pool = TRAIN_POOLS["grid"]
    dring_pool = TRAIN_POOLS["dring"]
    for p in (0.0, 0.15, 0.30, 0.50):
        arch, kwargs, n_layer = "softmax-add-3L", dict(attention="softmax", residual="add"), 3
        seed = 0
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 900)
        tag = f"p{int(p*100):02d}"
        name = f"{arch}-gridStutter-{tag}+dring-seed{seed}"
        if (Path("runs_gen") / name / "model.pt").exists():
            print(f"skip {name} (exists)", flush=True)
            continue
        model = CycleModel(N_VOCAB, 128, 1, n_layer, N_CTX, **kwargs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        steps = 24000
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / 100, 0.5 * (1 + math.cos(math.pi * s / steps))))
        for step in range(steps):
            gt = stutter_batch(grid_pool, 63, gen, p)
            dr, _, _, _ = walk_pool(dring_pool, 63, gen)
            tokens = torch.cat([gt, dr]).to(device)
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, N_VOCAB), tokens[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % 8000 == 0:
                print(f"{name} step {step} loss {loss.item():.3f}", flush=True)
        out = Path("runs_gen") / name
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        (out / "history.json").write_text(json.dumps({"config": {
            "d_model": 128, "n_layer": n_layer, "scale": 0.5, "norm": False,
            "stutter_p": p, **kwargs}}))

        model.cpu().eval()
        from icl_reps import mean_reps, gram_adjacency_corr, adjacency, pc_spectrum_alignment
        torch.set_grad_enabled(True)
        reps = mean_reps(model, "grid")
        A = adjacency("grid")
        c8, c256 = gram_adjacency_corr(reps[8], A), gram_adjacency_corr(reps[256], A)
        var, corr = pc_spectrum_alignment(reps[256], "grid")
        g2 = torch.Generator().manual_seed(999)
        toks, nodes, perm, pick = walk_pool(grid_pool, 128, g2)
        nb = torch.stack([grid_pool[i][0] for i in pick.tolist()])
        legal = legal_tokens(nodes, perm, nb)
        with torch.no_grad():
            lg = model(toks[:, :-1])
        hit = legal[:, :-1].gather(-1, lg.argmax(-1, keepdim=True)).squeeze(-1)
        legal_rate = hit[:, 128:].float().mean().item()
        print(f"== {name}: grid org ctx8 {c8:+.2f} ctx256 {c256:+.2f}  legal {legal_rate:.2f}  "
              f"PC12-harmonic corr {corr[0]:.2f}/{corr[1]:.2f}", flush=True)
        results[name] = {"stutter_p": p, "org8": c8, "org256": c256, "legal": legal_rate,
                         "pc_corr": corr[:4], "pc_var": var[:4]}
        model.to(device)
        Path("runs_gen/stutter_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
