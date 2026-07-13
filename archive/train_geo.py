"""Train bilinear attention-only models on lattice random walks; sweep sizes per topology.

Usage: python train_geo.py    # writes runs_geo/<topology>_L<layers>_d<dim>/{history.json,model.pt}
"""

import json
import math
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from geodata import N_CTX, N_VOCAB, TAIL, TOPOLOGIES, eval_sets, train_batch
from model import CycleModel

DEVICE = "cuda"
BATCH = 128
STEPS = 8000
WARMUP = 100
LR = 1e-3
EVAL_EVERY = 200


@torch.inference_mode()
def metrics(model, tokens, legal):
    """Legal-move rate and neighbor probability mass on tail positions (graph mostly revealed)."""
    logits = model(tokens)[:, TAIL:-1]
    legal = legal[:, TAIL:-1]
    rate = legal.gather(2, logits.argmax(-1, keepdim=True)).float().mean()
    mass = (logits.softmax(-1) * legal).sum(-1).mean()
    return rate.item(), mass.item()


def train(topology: str, n_layer: int, d_model: int, seed: int = 0, steps: int = STEPS, lr: float = LR, **model_kwargs):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)

    model = CycleModel(N_VOCAB, d_model, 1, n_layer, N_CTX, **model_kwargs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda t: min(1.0, t / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, t / steps))),
    )
    evals = {k: (t.to(DEVICE), m.to(DEVICE)) for k, (t, m) in eval_sets(topology).items()}

    history = {"step": [], "loss": [], **{f"{stat}|{k}": [] for k in evals for stat in ("legal", "mass")}}
    bar = tqdm(range(steps + 1), desc=f"{topology} L{n_layer} d{d_model}")
    for step in bar:
        tokens = train_batch(BATCH, topology, generator).to(DEVICE)
        loss = F.cross_entropy(model(tokens)[:, :-1].transpose(1, 2), tokens[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()

        if step % EVAL_EVERY == 0:
            history["step"].append(step)
            history["loss"].append(loss.item())
            for k, (t, m) in evals.items():
                rate, mass = metrics(model, t, m)
                history[f"legal|{k}"].append(rate)
                history[f"mass|{k}"].append(mass)
            bar.set_postfix(loss=f"{loss.item():.3f}", legal=f"{history['legal|in'][-1]:.2f}")

    return model, history


def main(grid=None, steps: int = STEPS, suffix: str = ""):
    grid = grid or list(product(TOPOLOGIES, (1, 2, 3), (32, 64, 128)))
    for topology, n_layer, d_model in grid:
        name = f"{topology}_L{n_layer}_d{d_model}{suffix}"
        run = Path("runs_geo") / name
        run.mkdir(parents=True, exist_ok=True)

        model, history = train(topology, n_layer, d_model, steps=steps)
        history["config"] = dict(topology=topology, n_layer=n_layer, d_model=d_model, n_params=model.n_params)
        torch.save(model.state_dict(), run / "model.pt")
        (run / "history.json").write_text(json.dumps(history))
        finals = {k: v[-1] for k, v in history.items() if k.startswith("legal")}
        print(name, model.n_params, "params |", " ".join(f"{k.split('|')[1]}={v:.3f}" for k, v in finals.items()), flush=True)


if __name__ == "__main__":
    main()
