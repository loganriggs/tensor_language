"""Lesson 6 data: the message on the bond can be SPARSE — a few active symbols from a shared
dictionary — not just low-dimensional. Toy: two modules with a wide dictionary bond (M symbols)
but only k active per input (TopK). Sweep k and show behavior saturates at small k -> sparse
communication. (The 'typed blob' idea is conceptual and shown in the figure.)
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
din, dhid, M, dout, N = 24, 48, 48, 6, 6000
X = torch.randn(N, din, device=DEV)
# same bilinear teacher family as lesson 5 (true underlying rank 4)
Ut = torch.randn(4, din, device=DEV); Vt = torch.randn(4, din, device=DEV)
Y = ((X @ Ut.T) * (X @ Vt.T)) @ torch.randn(4, dout, device=DEV)
Y = (Y - Y.mean(0)) / Y.std(0).clamp_min(1e-6)


def train_topk(k):
    A1 = nn.Linear(din, dhid, bias=False).to(DEV); B1 = nn.Linear(din, dhid, bias=False).to(DEV)
    enc = nn.Linear(dhid, M, bias=False).to(DEV)       # bond code over M symbols
    dec = nn.Linear(M, dout).to(DEV)
    opt = torch.optim.Adam(list(A1.parameters())+list(B1.parameters())+list(enc.parameters())+list(dec.parameters()), lr=3e-3)
    for step in range(3000):
        z = enc(A1(X)*B1(X))
        val, idx = z.topk(k, dim=-1); zk = torch.zeros_like(z).scatter_(-1, idx, F.relu(val))
        loss = F.mse_loss(dec(zk), Y); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = enc(A1(X)*B1(X)); val, idx = z.topk(k, -1); zk = torch.zeros_like(z).scatter_(-1, idx, F.relu(val))
        r2 = 1 - ((dec(zk)-Y)**2).sum().item()/((Y-Y.mean(0))**2).sum().item()
    return round(r2, 3)


ks = [1, 2, 4, 8, 16, 48]
r2 = [train_topk(k) for k in ks]
res = {'M_symbols': M, 'ks_active': ks, 'r2': r2,
       'knee': next((k for k, r in zip(ks, r2) if r >= 0.95), ks[-1])}
json.dump(res, open('tn_sparsecode_demo.json', 'w'), indent=1)
print(f"sparse-code bond over {M} symbols: R2 vs #active {list(zip(ks, r2))} (knee {res['knee']})", flush=True)
print('TN SPARSECODE DEMO DONE', flush=True)
