"""Seed replication of the session-4 self-loop flip: confirm gridSL+dring flips the
sign for more than one seed (seed 0 gave softmax +0.53, bilinear +0.21).

Usage: python toy_selfloop_seeds.py
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from graphs import N_VOCAB, N_CTX, TRAIN_POOLS, walk_pool
from model import CycleModel
from toy_recur import SL_POOL, GRID_BATCHES, legal_rate

if __name__ == "__main__":
    device = "cuda"
    results = {}
    dring_pool = TRAIN_POOLS["dring"]
    for seed in (1, 2):
        for arch, kwargs, n_layer in (
            ("softmax-add-3L", dict(attention="softmax", residual="add"), 3),
            ("bilin-lerp-2L", dict(attention="bilinear", residual="lerp"), 2),
        ):
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed + 500)
            name = f"{arch}-gridSL+dring-seed{seed}"
            if (Path("runs_gen") / name / "model.pt").exists():
                print(f"skip {name} (exists)", flush=True)
                continue
            model = CycleModel(N_VOCAB, 128, 1, n_layer, N_CTX, **kwargs).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            steps = 24000
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min((s + 1) / 100, 0.5 * (1 + math.cos(math.pi * s / steps))))
            for step in range(steps):
                gt = GRID_BATCHES["gridSL"](63, gen)[0]
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
                "variant": "gridSL", **kwargs}}))

            model.cpu().eval()
            from icl_reps import mean_reps, gram_adjacency_corr, adjacency, pc_spectrum_alignment
            torch.set_grad_enabled(True)
            reps = mean_reps(model, "grid")
            A = adjacency("grid")
            c8, c256 = gram_adjacency_corr(reps[8], A), gram_adjacency_corr(reps[256], A)
            var, corr = pc_spectrum_alignment(reps[256], "grid")
            std = legal_rate(model, "grid")
            print(f"== {name}: grid org ctx8 {c8:+.2f} ctx256 {c256:+.2f}  legal(std) {std:.2f}  "
                  f"PC12-harmonic corr {corr[0]:.2f}/{corr[1]:.2f}", flush=True)
            results[name] = {"org8": c8, "org256": c256, "legal_std": std,
                             "pc_corr": corr[:4], "pc_var": var[:4]}
            model.to(device)
            Path("runs_gen/selfloop_seeds_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
