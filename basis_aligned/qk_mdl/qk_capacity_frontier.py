"""TICK 181 (Logan): per-head capacity frontier for the mechanism ledger — trade off
features-per-token (k) against total atoms (m), per head. "Surely those 7 heads don't
all need 512?"

Part B (ladder, ground truth): for each head and k in {1,2,4,8}, train the Stage-1
triple SAE at ascending m in {32,64,...,4096} and record the sketched-moment residual;
stop a (head,k) ladder at the first GATE pass (<0.05), or abandon it early when the
projected residual at m=4096 (extrapolating the per-doubling decay ratio) exceeds
1.5x the gate. Minimal passing m per (head,k) + a bits proxy
  bits(m,k) = m*384*32 + V*k*(32 + log2 m)
give the per-head Pareto frontier over (k, m).

Part A (pruning, direct joint view from ONE dictionary): train once per head at a big m
(2048; heads 0/4 at 4096, fresh — tick 180 did not save its encoder), rank atoms by p-weighted
usage sum_t p_t |s_ta|, re-encode with only the top-n atoms for n in {16,...,m}, and
measure the moment residual. Shows how many atoms are load-bearing without retraining
(retrained ladder points should dominate; the gap measures how much retraining buys).

Incremental json after every training: qk_capacity_frontier.json. STEPS=9000 (vs 12000
in ticks 172/176/180) to keep ~90 trainings tractable; noted in the json.
"""
import json
import time
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
from tier2_folding import branch_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
STEPS, BATCH, LR, GATE = 9000, 2048, 3e-3, 0.05
KS = (1, 2, 4, 8)
MS = (32, 64, 128, 256, 512, 1024, 2048, 4096)

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
q1, k1 = branch_factors(m, 1)
q2, k2 = branch_factors(m, 2)
K1, K2 = k1.float().to(DEV), k2.float().to(DEV)
with torch.no_grad():
    a0 = m.transformer.h[0].attn
    E = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
    Vv = a0.c_v(E).view(V, NH, HD)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()
QP_CPU = QP.cpu()


def train_triple(Y, m_atoms, k_code, seed=0, steps=STEPS):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = Y[torch.randperm(len(Y), generator=g)[:m_atoms].to(DEV)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = (Y * QP[:, None]).sum(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=LR)
    fired = torch.zeros(m_atoms, device=DEV)
    for step in range(steps):
        kk = max(k_code, int(round(2 * k_code - k_code * min(1.0, 2 * step / steps))))
        bi = torch.multinomial(QP_CPU, BATCH, replacement=True, generator=g).to(DEV)
        y = Y[bi]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = torch.relu((y - b) @ We.T)
        vals, idx = z.topk(kk, dim=1)
        yhat = b + (vals.unsqueeze(-1) * Dn[idx]).sum(1)
        fired.index_add_(0, idx.reshape(-1), (vals > 1e-8).float().reshape(-1))
        loss = ((yhat - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            dead = (fired == 0).nonzero().squeeze(1)
            if len(dead):
                with torch.no_grad():
                    Dn_ = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    z_ = torch.relu((Y - b) @ We.T)
                    v_, i_ = z_.topk(k_code, dim=1)
                    rec = b + (v_.unsqueeze(-1) * Dn_[i_]).sum(1)
                    worst = ((rec - Y) ** 2).sum(1).topk(len(dead)).indices
                    Dm.data[dead] = Y[worst] / Y[worst].norm(dim=1, keepdim=True).clamp(min=1e-8)
                    We.data[dead] = Dm.data[dead]
                    del z_, rec
            fired.zero_()
    with torch.no_grad():
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = torch.relu((Y - b) @ We.T)
        vals, idx = z.topk(k_code, dim=1)
        rec = b + (vals.unsqueeze(-1) * Dn[idx]).sum(1)
    return Dn.detach(), b.detach(), We.detach(), idx, vals.detach(), rec.detach()


@torch.no_grad()
def moment_residual(Y, rec, n_probe=256, seed=3):
    g = torch.Generator(device='cpu').manual_seed(seed)
    num = den = 0.0
    for _ in range(n_probe):
        u, v_, wv = (torch.randn(Y.shape[1], generator=g).to(DEV) for _ in range(3))
        t = (QP * (Y @ u) * (Y @ v_) * (Y @ wv)).sum()
        th = (QP * (rec @ u) * (rec @ v_) * (rec @ wv)).sum()
        num += float((t - th) ** 2)
        den += float(t ** 2)
    return num / max(den, 1e-30)


def bits_proxy(mm, kc):
    return mm * 384 * 32 + V * kc * (32 + math.log2(mm))


def head_rows(h):
    return torch.cat([K1[:, h], K2[:, h], Vv[:, h]], 1)


res = {'steps': STEPS, 'gate': GATE, 'ladder': {}, 'prune': {}}
OUT = f'{QK}/qk_capacity_frontier.json'

# ---------------- Part B: retrained ladder ----------------
HEAD_ORDER = [1, 2, 3, 5, 6, 7, 8, 0, 4]              # easy heads first for early signal
for h in HEAD_ORDER:
    Y = head_rows(h)
    for kc in KS:
        key = f'h{h}_k{kc}'
        pts, prev = [], None
        for mm in MS:
            if mm < 2 * kc:
                continue
            t0 = time.time()
            _, _, _, _, _, rec = train_triple(Y, mm, kc)
            mres = moment_residual(Y, rec)
            pts.append({'m': mm, 'res': round(mres, 4),
                        'Mbit': round(bits_proxy(mm, kc) / 1e6, 2)})
            print(f'{key} m={mm}: res {mres:.4f} ({time.time() - t0:.0f}s)'
                  + (' PASS' if mres < GATE else ''), flush=True)
            res['ladder'][key] = pts
            json.dump(res, open(OUT, 'w'), indent=2)
            torch.cuda.empty_cache()
            if mres < GATE:
                break
            if prev is not None and mm >= 256:
                ratio = mres / max(prev, 1e-9)
                if ratio > 0.98:
                    print(f'{key}: plateau (ratio {ratio:.2f}), abandoning', flush=True)
                    break
                proj = mres * ratio ** math.log2(4096 / mm)
                if proj > 1.5 * GATE:
                    print(f'{key}: projected res@4096 {proj:.3f} > {1.5 * GATE:.3f}, '
                          f'abandoning', flush=True)
                    break
            prev = mres

# ---------------- Part A: usage-ranked pruning from one big dictionary ----------------
for h in range(NH):
    Y = head_rows(h)
    big_m = 4096 if h in (0, 4) else 2048           # h0/h4 only pass the gate at 4096
    kc = 8
    Dn, b, We, _, _, _ = train_triple(Y, big_m, kc)
    src = f'fresh_m{big_m}'
    with torch.no_grad():
        z = torch.relu((Y - b) @ We.T)
        vals, idx = z.topk(kc, dim=1)
        usage = torch.zeros(big_m, device=DEV)
        usage.index_add_(0, idx.reshape(-1), (QP[:, None].expand(-1, kc).reshape(-1)
                                              * vals.reshape(-1)))
        order = usage.argsort(descending=True)
        curve = []
        for n in [n for n in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096) if n <= big_m]:
            keep = order[:n]
            zk = torch.relu((Y - b) @ We[keep].T)
            vk, ik = zk.topk(min(kc, n), dim=1)
            reck = b + (vk.unsqueeze(-1) * Dn[keep][ik]).sum(1)
            mres = moment_residual(Y, reck)
            curve.append({'n': n, 'res': round(mres, 4)})
            del zk, reck
        res['prune'][f'h{h}'] = {'src': src, 'curve': curve}
        print(f'h{h} prune ({src}): ' +
              ' '.join(f'{c["n"]}:{c["res"]:.3f}' for c in curve), flush=True)
        json.dump(res, open(OUT, 'w'), indent=2)
        torch.cuda.empty_cache()
print('CAPACITY FRONTIER DONE', flush=True)
