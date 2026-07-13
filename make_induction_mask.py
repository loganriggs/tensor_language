"""Cache the induction-pattern mask for the active corpus's val stream:
is_induction[i] = token i's bigram completion appeared earlier in its window.
Writes <RUNS>/is_induction.npy. Usage: python make_induction_mask.py
"""

import numpy as np
import torch

from induction_dynamics import induction_structure
from text_data import CORPUS, N_CTX, RUNS, val_windows

if __name__ == "__main__":
    RUNS.mkdir(exist_ok=True)
    data, n_win = val_windows()
    masks = []
    for i in range(0, n_win, 512):
        buf = np.stack([data[w * N_CTX:w * N_CTX + N_CTX + 1]
                        for w in range(i, min(i + 512, n_win))]).astype(np.int64)
        _, is_ind = induction_structure(torch.from_numpy(buf).cuda())
        masks.append(is_ind.cpu().numpy())
    mask = np.concatenate(masks).reshape(-1)
    np.save(RUNS / "is_induction.npy", mask)
    print(f"{CORPUS}: induction-predictable {mask.mean():.3%} of {mask.size} datapoints")
