"""Graph-COMPUTATION experiment (user's session-4 idea, tried first): if computing
on the graph is what forces a map, then interleaving distance questions into walk
documents should flip the reliably-anti grid+dring mixture positive.

Document format for grid docs: 216-token walk, then 10 query blocks of 4 tokens
    [QDIST] u v ANS_d
where u, v are node tokens of the document's graph and d = graph distance(u, v)
(BFS; 0..7 on our grids). QDIST and ANS_0..ANS_12 are reserved special tokens
(ids 100..113); node labels stay 0..99. dring docs are plain 256-token walks
(the irreversibility pressure that reliably pins grid+dring anti: bilinear
baseline -0.55/-0.70/-0.72; softmax baseline trained by toy_recur.py this session).

Pre-registered P5: softmax-add-3L on grid+dring WITH distance queries organizes
positive (the anti-map supports next-token elimination, but distance is monotone
in neighbor similarity, so metric queries reward neighbors-nearby coordinates).
Falsifiable alternative: queries answered by a separate lookup circuit, org stays
anti. Query accuracy is reported so "never learned the queries" is identifiable
(chance = 1/13 blind, ~0.25-0.35 if it learns the distance marginal).

Usage: python toy_query.py
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from graphs import N_VOCAB, N_CTX, MAX_NODES, TRAIN_POOLS, walk_pool, legal_tokens
from model import CycleModel

QDIST = N_VOCAB                 # 100
N_ANS = 13
ANS0 = N_VOCAB + 1              # 101..113
N_VOCAB_Q = ANS0 + N_ANS        # 114
N_Q = 10
QLEN = 4 * N_Q                  # 40
WALK_LEN = N_CTX - QLEN         # 216
ANS_POS = torch.tensor([WALK_LEN + 4 * q + 3 for q in range(N_Q)])


def dist_matrix(nb, deg):
    """All-pairs BFS distances, padded to MAX_NODES (pad entries stay 99)."""
    n = int((nb[:, 0] >= 0).sum())
    D = torch.full((MAX_NODES, MAX_NODES), 99, dtype=torch.long)
    for s in range(n):
        D[s, s] = 0
        frontier, d = [s], 0
        while frontier:
            d += 1
            nxt = []
            for v in frontier:
                for u in nb[v, : deg[v]].tolist():
                    if u >= 0 and D[s, u] > d:
                        D[s, u] = d
                        nxt.append(u)
            frontier = nxt
    return D


QPOOL = [(nb, deg, dist_matrix(nb, deg)) for nb, deg in TRAIN_POOLS["grid"]]


def query_walk_batch(n_seq, gen):
    """Grid walk of WALK_LEN tokens + N_Q distance-query blocks."""
    pick = torch.randint(0, len(QPOOL), (n_seq,), generator=gen)
    nb = torch.stack([QPOOL[i][0] for i in pick.tolist()])
    deg = torch.stack([QPOOL[i][1] for i in pick.tolist()])
    Ds = torch.stack([QPOOL[i][2] for i in pick.tolist()])
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
    d = Ds[rows[:, None], u, v]
    blocks = torch.stack(
        [torch.full_like(u, QDIST), perm.gather(1, u), perm.gather(1, v), ANS0 + d], -1)
    return torch.cat([walk_toks, blocks.reshape(n_seq, QLEN)], 1)


def query_accuracy(logits, tokens):
    """Top-1 accuracy at the ANS positions (logits index i predicts token i+1)."""
    pred = logits[:, ANS_POS - 1].argmax(-1)
    return (pred == tokens[:, ANS_POS]).float().mean().item()


if __name__ == "__main__":
    device = "cuda"
    results = {}
    grid_pool = TRAIN_POOLS["grid"]
    dring_pool = TRAIN_POOLS["dring"]
    for seed in (0, 1):
        for arch, kwargs, n_layer in (
            ("softmax-add-3L", dict(attention="softmax", residual="add"), 3),
            ("bilin-lerp-2L", dict(attention="bilinear", residual="lerp"), 2),
        ):
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed + 400)
            name = f"{arch}-grid+dring-qdist-seed{seed}"
            if (Path("runs_gen") / name / "model.pt").exists():
                print(f"skip {name} (exists)", flush=True)
                continue
            model = CycleModel(N_VOCAB_Q, 128, 1, n_layer, N_CTX, **kwargs).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            steps = 24000
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min((s + 1) / 100, 0.5 * (1 + math.cos(math.pi * s / steps))))
            for step in range(steps):
                gq = query_walk_batch(63, gen)
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
                "n_vocab": N_VOCAB_Q, "qdist": True, **kwargs}}))

            model.cpu().eval()
            from icl_reps import mean_reps, gram_adjacency_corr, adjacency, pc_spectrum_alignment
            torch.set_grad_enabled(True)   # icl_reps disables grads globally at import
            reps = mean_reps(model, "grid")
            A = adjacency("grid")
            c8, c256 = gram_adjacency_corr(reps[8], A), gram_adjacency_corr(reps[256], A)
            var, corr = pc_spectrum_alignment(reps[256], "grid")
            # held-out query accuracy
            g2 = torch.Generator().manual_seed(999)
            qtoks = query_walk_batch(128, g2)
            with torch.no_grad():
                qlg = model(qtoks[:, :-1])
            qacc = query_accuracy(qlg, qtoks)
            # legal rate on plain grid walks (argmax over FULL vocab incl. specials)
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
            Path("runs_gen/query_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
