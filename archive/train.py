"""Train bilinear attention-only models on token cycles; sweep over sizes.

Usage: python train.py            # runs the full sweep, writes runs/<name>/{history.json,model.pt}
"""

import json
import math
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import N_CTX, N_VOCAB, eval_sets, train_batch
from model import CycleModel

DEVICE = "cuda"
BATCH = 256
STEPS = 4000
WARMUP = 100
LR = 1e-3
EVAL_EVERY = 100


def masked_loss(logits, tokens, mask):
    logits, targets = logits[:, :-1], tokens[:, 1:]
    loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return (loss * mask).sum() / mask.sum()


@torch.inference_mode()
def accuracy(model, tokens, mask):
    preds = model(tokens)[:, :-1].argmax(-1)
    correct = (preds == tokens[:, 1:]) * mask
    return (correct.sum() / mask.sum()).item()


def train(n_layer: int, d_model: int, n_head: int, seed: int = 0, steps: int = STEPS, **model_kwargs):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)

    model = CycleModel(N_VOCAB, d_model, n_head, n_layer, N_CTX, **model_kwargs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda t: min(1.0, t / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, t / steps))),
    )
    evals = {k: (t.to(DEVICE), m.to(DEVICE)) for k, (t, m) in eval_sets().items()}

    history = {"step": [], "loss": [], **{f"acc@{k}": [] for k in evals}}
    bar = tqdm(range(steps + 1), desc=f"L{n_layer} d{d_model} h{n_head}")
    for step in bar:
        tokens, mask = train_batch(BATCH, generator)
        tokens, mask = tokens.to(DEVICE), mask.to(DEVICE)

        loss = masked_loss(model(tokens), tokens, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()

        if step % EVAL_EVERY == 0:
            history["step"].append(step)
            history["loss"].append(loss.item())
            for k, (t, m) in evals.items():
                history[f"acc@{k}"].append(accuracy(model, t, m))
            bar.set_postfix(loss=f"{loss.item():.3f}", acc20=f"{history['acc@20'][-1]:.2f}")

    return model, history


def main(grid=None, steps: int = STEPS, suffix: str = ""):
    grid = grid or list(product((1, 2), (8, 16, 32, 64), (1, 2)))
    for n_layer, d_model, n_head in grid:
        name = f"L{n_layer}_d{d_model}_h{n_head}{suffix}"
        run = Path("runs") / name
        run.mkdir(parents=True, exist_ok=True)

        model, history = train(n_layer, d_model, n_head, steps=steps)
        history["config"] = dict(n_layer=n_layer, d_model=d_model, n_head=n_head, n_params=model.n_params)
        torch.save(model.state_dict(), run / "model.pt")
        (run / "history.json").write_text(json.dumps(history))
        finals = {k: v[-1] for k, v in history.items() if k.startswith("acc")}
        print(name, model.n_params, "params |", " ".join(f"{k}={v:.3f}" for k, v in finals.items()), flush=True)


if __name__ == "__main__":
    main()
