"""TICK 234: toy refinement rung 2 — EXPOSURE-MATCHED capacity under Zipf.

Question: when an associative net is trained under the same Zipf(1) exposure it is
scored on (the realistic regime), how deep into the tail does its memory reach per
parameter? Deliverable: rank50(P) — the exposure rank at which recall drops below 50%
— versus parameter count. Extrapolated to the real block-0 MLP (~21M params), this
estimates how many entity-continuation pairs bilin18 plausibly stores, which in turn
bounds what ANY explicit replacement must match.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
VOC, DIN, NPAIR = 1024, 64, 2_000_000

K = torch.randn(NPAIR, DIN, device=DEV)
Vl = torch.randint(0, VOC, (NPAIR,), device=DEV)
pz = 1.0 / torch.arange(1, NPAIR + 1, dtype=torch.float64)
pz = (pz / pz.sum()).float().to(DEV)


class BilinNet(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.L = nn.Linear(DIN, hid, bias=False)
        self.R = nn.Linear(DIN, hid, bias=False)
        self.D = nn.Linear(hid, VOC, bias=False)

    def forward(self, x):
        return self.D(self.L(x) * self.R(x))


out = {}
for hid in (52, 208, 832):
    net = BilinNet(hid).to(DEV)
    P = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for s in range(12000):
        bi = torch.multinomial(pz, 8192, replacement=True)
        loss = F.cross_entropy(net(K[bi]), Vl[bi])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s in (8000, 10500):
            for pg in opt.param_groups:
                pg['lr'] *= 0.3
    with torch.no_grad():
        recall_by_band, rank50 = [], None
        bands = [(0, 1000), (1000, 4000), (4000, 16000), (16000, 64000),
                 (64000, 256000), (256000, 1_000_000), (1_000_000, NPAIR)]
        for lo, hi in bands:
            sel = torch.arange(lo, min(hi, NPAIR), device=DEV)
            if len(sel) > 100000:
                sel = sel[torch.randint(0, len(sel), (100000,), device=DEV)]
            acc = float((net(K[sel]).argmax(1) == Vl[sel]).float().mean())
            recall_by_band.append(round(acc, 3))
            if rank50 is None and acc < 0.5:
                rank50 = lo
        pairs_held = 0.0
        for (lo, hi), acc in zip(bands, recall_by_band):
            pairs_held += acc * (min(hi, NPAIR) - lo)
    out[f'P{P}'] = {'params': P, 'recall_by_band': recall_by_band,
                    'rank50_at_least': rank50, 'pairs_held_est': int(pairs_held)}
    print(f'P={P}: recall by band {recall_by_band} | rank50>={rank50} | '
          f'pairs held ~{int(pairs_held)} ({pairs_held/P:.2f}/param)', flush=True)
    json.dump(out, open(f'{QK}/qk_toy_memory2.json', 'w'), indent=2)
    del net
    torch.cuda.empty_cache()
print('TOY MEMORY 2 DONE', flush=True)
