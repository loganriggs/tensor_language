"""Control for toy_query.py: identical document format, token budget, and training
recipe, but the query answer is a LOCAL (one-hop) property instead of the global
metric. If distance queries flip organization positive but this control does not, the
cause is the METRIC content of the computation, not query formatting or extra
supervision.

Two controls, selectable by CTRL env-style constant below:
  - adj : "are u, v neighbors?" (binary; ANS0 = no, ANS0+1 = yes). One-hop lookup.
  - deg : "degree of u" (2..4 on grids; answer ANS0+deg). Local count.

Both reuse toy_query's block layout ([Q] u v ANS) exactly; only the answer differs.
Trained on grid+dring like the main experiment.

Usage: python toy_query_ctrl.py adj    (or: deg)
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from graphs import N_VOCAB, N_CTX, MAX_NODES, TRAIN_POOLS, walk_pool, legal_tokens
from model import CycleModel
from toy_query import (QDIST, ANS0, N_VOCAB_Q, N_Q, QLEN, WALK_LEN, ANS_POS, QPOOL,
                       query_accuracy)

CTRL = sys.argv[1] if len(sys.argv) > 1 else "adj"
assert CTRL in ("adj", "deg")


def ctrl_walk_batch(n_seq, gen):
    pick = torch.randint(0, len(QPOOL), (n_seq,), generator=gen)
    nb = torch.stack([QPOOL[i][0] for i in pick.tolist()])
    deg = torch.stack([QPOOL[i][1] for i in pick.tolist()])
    n_nodes = (nb[:, :, 0] >= 0).sum(1)
    perm = torch.rand(n_seq, N_VOCAB, generator=gen).argsort(1)[:, :MAX_NODES]
    position = torch.randint(0, MAX_NODES, (n_seq,), generator=gen) % n_nodes
    rows = torch.arange(n_seq)
    trail = [position]
    for _ in range(WALK_LEN - 1):
        choice = (torch.rand(n_seq, generator=gen) * deg[rows, position]).long()
        position = nb[rows, position, choice]
        trail.append(position)
    walk_toks = perm.gather(1, torch.stack(trail, 1))
    u = (torch.rand(n_seq, N_Q, generator=gen) * n_nodes[:, None]).long()
    v = (torch.rand(n_seq, N_Q, generator=gen) * n_nodes[:, None]).long()
    if CTRL == "adj":
        # is v a neighbor of u? compare v against u's neighbor list
        nb_u = nb[rows[:, None], u]                       # (n_seq, N_Q, MAX_DEG)
        ans = (nb_u == v[..., None]).any(-1).long()       # 0/1
    else:  # deg
        ans = deg[rows[:, None], u]                        # true degree of u
    blocks = torch.stack(
        [torch.full_like(u, QDIST), perm.gather(1, u), perm.gather(1, v), ANS0 + ans], -1)
    return torch.cat([walk_toks, blocks.reshape(n_seq, QLEN)], 1)


if __name__ == "__main__":
    device = "cuda"
    results = {}
    grid_pool = TRAIN_POOLS["grid"]
    dring_pool = TRAIN_POOLS["dring"]
    for arch, kwargs, n_layer in (
        ("softmax-add-3L", dict(attention="softmax", residual="add"), 3),
        ("bilin-lerp-2L", dict(attention="bilinear", residual="lerp"), 2),
    ):
        seed = 0
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 800)
        name = f"{arch}-grid+dring-q{CTRL}-seed{seed}"
        if (Path("runs_gen") / name / "model.pt").exists():
            print(f"skip {name} (exists)", flush=True)
            continue
        model = CycleModel(N_VOCAB_Q, 128, 1, n_layer, N_CTX, **kwargs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        steps = 24000
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / 100, 0.5 * (1 + math.cos(math.pi * s / steps))))
        for step in range(steps):
            gq = ctrl_walk_batch(63, gen)
            dr, _, _, _ = walk_pool(dring_pool, 63, gen)
            tokens = torch.cat([gq, dr]).to(device)
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, N_VOCAB_Q), tokens[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % 6000 == 0:
                qacc = query_accuracy(logits[:63].detach().cpu(), tokens[:63].cpu())
                print(f"{name} step {step} loss {loss.item():.3f} qacc {qacc:.2f}", flush=True)
        out = Path("runs_gen") / name
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        (out / "history.json").write_text(json.dumps({"config": {
            "d_model": 128, "n_layer": n_layer, "scale": 0.5, "norm": False,
            "n_vocab": N_VOCAB_Q, "qctrl": CTRL, **kwargs}}))

        model.cpu().eval()
        from icl_reps import mean_reps, gram_adjacency_corr, adjacency, pc_spectrum_alignment
        torch.set_grad_enabled(True)
        reps = mean_reps(model, "grid")
        A = adjacency("grid")
        c8, c256 = gram_adjacency_corr(reps[8], A), gram_adjacency_corr(reps[256], A)
        var, corr = pc_spectrum_alignment(reps[256], "grid")
        g2 = torch.Generator().manual_seed(999)
        qtoks = ctrl_walk_batch(128, g2)
        with torch.no_grad():
            qlg = model(qtoks[:, :-1])
        qacc = query_accuracy(qlg, qtoks)
        toks, nodes, perm, pick = walk_pool(grid_pool, 128, g2)
        nb = torch.stack([grid_pool[i][0] for i in pick.tolist()])
        legal = F.pad(legal_tokens(nodes, perm, nb), (0, N_VOCAB_Q - N_VOCAB))
        with torch.no_grad():
            lg = model(toks[:, :-1])
        hit = legal[:, :-1].gather(-1, lg.argmax(-1, keepdim=True)).squeeze(-1)
        legal_rate = hit[:, 128:].float().mean().item()
        print(f"== {name}: grid org ctx8 {c8:+.2f} ctx256 {c256:+.2f}  legal {legal_rate:.2f}  "
              f"qacc {qacc:.2f}  PC12-harmonic corr {corr[0]:.2f}/{corr[1]:.2f}", flush=True)
        results[name] = {"org8": c8, "org256": c256, "legal": legal_rate, "qacc": qacc,
                         "pc_corr": corr[:4], "pc_var": var[:4]}
        model.to(device)
        Path(f"runs_gen/query_ctrl_{CTRL}_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
