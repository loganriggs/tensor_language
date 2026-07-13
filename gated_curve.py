"""CE on a gated token set across a run's checkpoints (phase-change resolution tool).

Usage: python gated_curve.py <run-tag> [gateset=gated_depth2] [n_windows=500]
Writes runs_lm/<tag>/<gateset>_ce_curve.json
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lm_eval import load_model
from text_data import N_CTX, RUNS, val_windows


def curve(tag, gateset="gated_depth2", n_windows=500):
    run = RUNS / tag
    data, _ = val_windows()
    buf = np.stack([data[w * N_CTX:w * N_CTX + N_CTX + 1] for w in range(n_windows)]).astype(np.int64)
    b = torch.from_numpy(buf).cuda()
    gated = np.load(RUNS / f"{gateset}.npy")
    gated = gated[gated < n_windows * N_CTX]
    gm = torch.zeros(n_windows * N_CTX, dtype=torch.bool)
    gm[gated] = True
    gm = gm.view(n_windows, N_CTX).cuda()
    ckpts = sorted(run.glob("ckpt/step*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    out = {}
    with torch.no_grad():
        for ck in ckpts + [None]:
            m = load_model(run, ck.stem if ck else None)
            step = int(re.findall(r"\d+", ck.name)[0]) if ck else \
                json.loads((run / "config.json").read_text())["steps"]
            ces = []
            for i in range(0, n_windows, 100):
                ce = F.cross_entropy(m(b[i:i + 100, :-1]).transpose(1, 2), b[i:i + 100, 1:],
                                     reduction="none")
                ces.append(ce)
            out[step] = round(torch.cat(ces)[gm].mean().item(), 4)
            print(step, out[step], flush=True)
    (run / f"{gateset}_ce_curve.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    curve(a[0], *(a[1:2] or ["gated_depth2"]), n_windows=int(a[2]) if len(a) > 2 else 500)
