"""Train architectures on the multi-family graph-tracing mixture.

Usage: python train_general.py   # all archs -> runs_gen/<name>/{history.json,model.pt}
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from graphs import N_CTX, N_VOCAB, TAIL, eval_sets, train_batch
from model import CycleModel

DEVICE = "cuda"
BATCH = 126          # divisible by 6 families
STEPS = 24000
WARMUP = 100
LR = 1e-3
EVAL_EVERY = 400

ARCHS = {
    "bilin-lerp-2L": dict(n_layer=2, attention="bilinear", residual="lerp"),
    "bilin-add-2L": dict(n_layer=2, attention="bilinear", residual="add"),
    "bilin-add-3L": dict(n_layer=3, attention="bilinear", residual="add"),
    "softmax-lerp-2L": dict(n_layer=2, attention="softmax", residual="lerp"),
    "softmax-add-3L": dict(n_layer=3, attention="softmax", residual="add"),
}


@torch.inference_mode()
def metrics(model, tokens, legal):
    logits = model(tokens)[:, TAIL:-1]
    legal = legal[:, TAIL:-1]
    rate = legal.gather(2, logits.argmax(-1, keepdim=True)).float().mean().item()
    mass = (logits.softmax(-1) * legal).sum(-1).mean().item()
    return rate, mass


def train(name: str, arch: dict, d_model: int = 128, seed: int = 0, steps: int = STEPS):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    kwargs = {k: v for k, v in arch.items() if k != "n_layer"}
    model = CycleModel(N_VOCAB, d_model, 1, arch["n_layer"], N_CTX, **kwargs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda t: min(1.0, t / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, t / steps))))
    evals = {k: (t.to(DEVICE), m.to(DEVICE)) for k, (t, m) in eval_sets().items()}

    history = {"step": [], "loss": [], **{f"{s}|{k}": [] for k in evals for s in ("legal", "mass")}}
    bar = tqdm(range(steps + 1), desc=name)
    for step in bar:
        tokens = train_batch(BATCH, generator).to(DEVICE)
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
            bar.set_postfix(loss=f"{loss.item():.3f}", grid=f"{history['legal|grid'][-1]:.2f}",
                            torus=f"{history['legal|torus (unseen family)'][-1]:.2f}")
    return model, history


def main(archs=None, steps: int = STEPS, suffix: str = ""):
    for name, arch in (archs or ARCHS).items():
        run = Path("runs_gen") / (name + suffix)
        run.mkdir(parents=True, exist_ok=True)
        model, history = train(name, arch, steps=steps)
        history["config"] = dict(name=name, d_model=128, n_params=model.n_params, **arch)
        torch.save(model.state_dict(), run / "model.pt")
        (run / "history.json").write_text(json.dumps(history))
        finals = {k.split("|")[1]: v[-1] for k, v in history.items() if k.startswith("legal")}
        print(name, "|", " ".join(f"{k}={v:.2f}" for k, v in finals.items()), flush=True)


if __name__ == "__main__":
    main()
