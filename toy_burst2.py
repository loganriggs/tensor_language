"""Strongest version of the recurrence test: SIX-FAMILY mixture + burst.

The six-family mixture reliably pins bilin-lerp-2L positive (+0.55..+0.66) but
softmax-add-3L stays anti (-0.67/-0.80) on it. If recurrence is the true active
ingredient for softmax (per the LLM circuit story), adding burst documents to the
full mixture should rescue softmax-add-3L into the positive mode too — and may push
bilinear even higher ("more self-organizing").

Usage: python toy_burst2.py
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from graphs import N_VOCAB, N_CTX, TRAIN_POOLS, walk_pool, legal_tokens
from model import CycleModel
from toy_burst import burst_batch

if __name__ == "__main__":
    device = "cuda"
    results = {}
    pools = list(TRAIN_POOLS.values())
    grid_pool = TRAIN_POOLS["grid"]
    for arch, kwargs, n_layer in (
        ("softmax-add-3L", dict(attention="softmax", residual="add"), 3),
        ("bilin-lerp-2L", dict(attention="bilinear", residual="lerp"), 2),
    ):
        seed = 0
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 200)
        name = f"{arch}-mix+burst-seed{seed}"
        if (Path("runs_gen") / name / "model.pt").exists():
            print(f"skip {name}", flush=True)
            continue
        model = CycleModel(N_VOCAB, 128, 1, n_layer, N_CTX, **kwargs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        steps = 24000
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / 100, 0.5 * (1 + math.cos(math.pi * s / steps))))
        for step in range(steps):
            # 108 walk docs spread over the six families + 18 burst docs (1/7 of batch)
            parts = [walk_pool(p, 18, gen)[0] for p in pools]
            parts.append(burst_batch(18, gen))
            tokens = torch.cat(parts).to(device)
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
        torch.set_grad_enabled(True)
        reps = mean_reps(model, "grid")
        A = adjacency("grid")
        c8, c256 = gram_adjacency_corr(reps[8], A), gram_adjacency_corr(reps[256], A)
        g2 = torch.Generator().manual_seed(999)
        toks, nodes, perm, pick = walk_pool(grid_pool, 128, g2)
        nb = torch.stack([grid_pool[i][0] for i in pick.tolist()])
        legal = legal_tokens(nodes, perm, nb)
        with torch.no_grad():
            lg = model(toks[:, :-1])
        hit = legal[:, :-1].gather(-1, lg.argmax(-1, keepdim=True)).squeeze(-1)
        legal_rate = hit[:, 128:].float().mean().item()
        print(f"== {name}: grid org ctx8 {c8:+.2f} ctx256 {c256:+.2f}  legal {legal_rate:.2f}",
              flush=True)
        results[name] = {"org8": c8, "org256": c256, "legal": legal_rate}
        model.to(device)
    Path("runs_gen/burst2_results.json").write_text(json.dumps(results, indent=1))
