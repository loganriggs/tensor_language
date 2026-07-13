"""Per-token CE of every trained model on the frozen val stream — PLAN.md step 2.

The val stream is chunked into non-overlapping (N_CTX+1)-token windows on a fixed grid
(text_data.val_windows), identical for every model. For each run in runs_lm/ with a final
model.pt, writes runs_lm/<tag>/val_ce.npy: float16 array (n_win, N_CTX) of CE at each
predicted position. This is the model×datapoint matrix that differential.py consumes.

Usage: python lm_eval.py            (all runs missing val_ce.npy)
       python lm_eval.py attn2-seed0 [--ckpt step10000]   (one run / one checkpoint)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deep_model import DeepModel
from text_data import VOCAB, N_CTX, RUNS, val_windows

BATCH = 128


def load_model(run: Path, ckpt: str | None = None, device="cuda"):
    cfg = json.loads((run / "config.json").read_text())
    model = DeepModel(cfg["vocab"], cfg["d_model"], cfg["n_head"], cfg["spec"], cfg["n_ctx"],
                      norm=cfg["norm"], attention=cfg["attention"], residual=cfg["residual"],
                      mlp_residual="add").to(device)
    path = run / (f"ckpt/{ckpt}.pt" if ckpt else "model.pt")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def eval_run(run: Path, ckpt: str | None = None, device="cuda"):
    data, n_win = val_windows()
    model = load_model(run, ckpt, device)
    out = np.empty((n_win, N_CTX), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, n_win, BATCH):
            ws = range(i, min(i + BATCH, n_win))
            buf = np.stack([data[w * N_CTX:w * N_CTX + N_CTX + 1] for w in ws]).astype(np.int64)
            b = torch.from_numpy(buf).to(device)
            logits = model(b[:, :-1])
            ce = F.cross_entropy(logits.transpose(1, 2), b[:, 1:], reduction="none")
            out[i:i + len(ce)] = ce.cpu().numpy().astype(np.float16)
    name = f"val_ce{'-' + ckpt if ckpt else ''}.npy"
    np.save(run / name, out)
    print(f"{run.name}{' @' + ckpt if ckpt else ''}: mean CE {out.mean():.4f} -> {name}", flush=True)
    return out.mean()


if __name__ == "__main__":
    args = sys.argv[1:]
    ckpt = None
    if "--ckpt" in args:
        i = args.index("--ckpt"); ckpt = args[i + 1]; del args[i:i + 2]
    runs = ([RUNS / a for a in args] if args else
            sorted(p.parent for p in RUNS.glob("*/model.pt")
                   if not (p.parent / "val_ce.npy").exists()))
    for run in runs:
        eval_run(run, ckpt)
