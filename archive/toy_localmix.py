"""Architecture version of the recurrence experiment: give the toys a built-in
LOCAL VALUE-BLENDING layer (what GPT-2's previous-token heads provide) and train on
the plain six-family mixture (NO burst data).

LocalMixModel = CycleModel with x_t := x_t + 0.5 * mean(x_{t-1..t-3}) applied right
after the embedding (a fixed causal local average = one hard-wired message-passing
step; parameter-free, so the comparison is pure architecture).

Pre-registered predictions:
  P3: softmax-add-3L+localmix organizes positive on the plain mixture
      (baseline -0.80/-0.67) -- if the missing ingredient is a positive local blend,
      hard-wiring it should do what burst data did.
  P4: bilin-lerp-2L+localmix >= +0.66 on the mixture ("more self-organizing"), with a
      cleaner Park spectrum (higher PC1/PC2 harmonic corr).

Usage: python toy_localmix.py
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from graphs import N_VOCAB, N_CTX, TRAIN_POOLS, walk_pool, legal_tokens
from model import CycleModel


class LocalMixModel(CycleModel):
    def _mix(self, x):
        # causal local average of the previous 3 positions
        prev = torch.stack([torch.roll(x, s, dims=-2) for s in (1, 2, 3)]).mean(0)
        prev[..., :3, :] = 0
        return x + 0.5 * prev

    def forward(self, tokens):
        x = self._mix(self.embed(tokens))
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def residuals(self, tokens):
        x, stream = self._mix(self.embed(tokens)), []
        for layer in self.layers:
            x = layer(x)
            stream.append(x)
        return stream


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
        gen = torch.Generator().manual_seed(seed + 300)
        name = f"{arch}-localmix-mix-seed{seed}"
        if (Path("runs_gen") / name / "model.pt").exists():
            print(f"skip {name}", flush=True)
            continue
        model = LocalMixModel(N_VOCAB, 128, 1, n_layer, N_CTX, **kwargs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        steps = 24000
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / 100, 0.5 * (1 + math.cos(math.pi * s / steps))))
        for step in range(steps):
            parts = [walk_pool(p, 21, gen)[0] for p in pools]   # 126 docs, six families only
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
            "d_model": 128, "n_layer": n_layer, "scale": 0.5, "norm": False,
            "localmix": True, **kwargs}}))

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
        results[name] = {"org8": c8, "org256": c256, "legal": legal_rate,
                         "pc_corr": corr[:4], "pc_var": var[:4]}
        model.to(device)
    Path("runs_gen/localmix_results.json").write_text(json.dumps(results, indent=1))
