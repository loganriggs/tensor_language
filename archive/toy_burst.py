"""Toy feedback experiment: does natural-text-like RECURRENCE pressure install the
positive map in the toy softmax stack (the way pretraining installed it in LLMs)?

New training family "burst": walks on a complete graph K16 where with p=0.5 the next
token repeats one of the last 3 visited nodes (recency recurrence, all moves legal),
else uniform. This rewards positively copying recent context into predictions --
the pressure the GPT-2 circuit analysis identified as the map-builder's origin.

Pre-registered predictions (from the LLM circuit story):
  P1: softmax-add-3L on grid+burst flips POSITIVE on grid (vs -0.80 on the six-family
      mixture, -0.55..-0.72 on grid+dring, and stochastic-diversity partners were
      supposed to be what flips things -- burst is a K16 blob with NO graph structure).
  P2: bilin-lerp-2L on grid+burst also positive (>= its +0.24/-0.24 cylinder lottery).

Usage: python toy_burst.py
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from graphs import N_VOCAB, N_CTX, TRAIN_POOLS, walk_pool, legal_tokens
from model import CycleModel

BURST_N = 16
P_RECENT = 0.5


def burst_batch(batch, generator):
    perm = torch.stack([torch.randperm(N_VOCAB, generator=generator)[:BURST_N]
                        for _ in range(batch)])
    nodes = torch.empty(batch, N_CTX, dtype=torch.long)
    nodes[:, 0] = torch.randint(BURST_N, (batch,), generator=generator)
    for t in range(1, N_CTX):
        recent = nodes[:, max(0, t - 3):t]
        pick_recent = torch.rand(batch, generator=generator) < P_RECENT
        ridx = torch.randint(recent.size(1), (batch,), generator=generator)
        rchoice = recent.gather(1, ridx[:, None]).squeeze(1)
        uchoice = torch.randint(BURST_N, (batch,), generator=generator)
        nodes[:, t] = torch.where(pick_recent, rchoice, uchoice)
    return perm.gather(1, nodes)


if __name__ == "__main__":
    device = "cuda"
    results = {}
    grid_pool = TRAIN_POOLS["grid"]
    for arch, kwargs, n_layer in (
        ("softmax-add-3L", dict(attention="softmax", residual="add"), 3),
        ("bilin-lerp-2L", dict(attention="bilinear", residual="lerp"), 2),
    ):
        for seed in (0, 1):
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed + 100)
            name = f"{arch}-grid+burst-seed{seed}"
            if (Path("runs_gen") / name / "model.pt").exists():
                print(f"skip {name} (exists)", flush=True)
                continue
            model = CycleModel(N_VOCAB, 128, 1, n_layer, N_CTX, **kwargs).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            steps = 24000
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min((s + 1) / 100, 0.5 * (1 + math.cos(math.pi * s / steps))))
            for step in range(steps):
                gt, _, _, _ = walk_pool(grid_pool, 63, gen)
                bt = burst_batch(63, gen)
                tokens = torch.cat([gt, bt]).to(device)
                logits = model(tokens[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, N_VOCAB), tokens[:, 1:].reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                if step % 6000 == 0:
                    print(f"{name} step {step} loss {loss.item():.3f}", flush=True)
            out = Path("runs_gen") / name
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / "model.pt")
            (out / "history.json").write_text(json.dumps({"config": {
                "d_model": 128, "n_layer": n_layer, "scale": 0.5, "norm": False, **kwargs}}))

            model.cpu().eval()
            from icl_reps import mean_reps, gram_adjacency_corr, adjacency
            torch.set_grad_enabled(True)   # icl_reps disables grads globally at import
            reps = mean_reps(model, "grid")
            A = adjacency("grid")
            c8, c256 = gram_adjacency_corr(reps[8], A), gram_adjacency_corr(reps[256], A)
            # legal rate on grid docs via graphs.py machinery
            g2 = torch.Generator().manual_seed(999)
            toks, nodes, perm, pick = walk_pool(grid_pool, 128, g2)
            nb = torch.stack([grid_pool[i][0] for i in pick.tolist()])
            legal = legal_tokens(nodes, perm, nb)
            with torch.no_grad():
                lg = model(toks[:, :-1])
            # legal[b,t] = neighbor tokens of the CURRENT node at t; logits[t] predicts t+1
            hit = legal[:, :-1].gather(-1, lg.argmax(-1, keepdim=True)).squeeze(-1)
            legal_rate = hit[:, 128:].float().mean().item()
            print(f"== {name}: grid org ctx8 {c8:+.2f} ctx256 {c256:+.2f}  legal {legal_rate:.2f}",
                  flush=True)
            results[name] = {"org8": c8, "org256": c256, "legal": legal_rate}
            model.to(device)
    Path("runs_gen/burst_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
