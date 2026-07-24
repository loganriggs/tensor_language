"""TICK 198: layer-1 capacity frontier — minimal (m, k) per layer-1 head, mirroring the
tick-181 layer-0 sweep. Rows come from the saved layer-1 token tables (qk_l1_tables.pt,
with head 1 on shrinkage tables rebuilt here from the same recipe as tick 196 — the
saved qk_l1_tables.pt predates the shrinkage fix, so ALL heads use shrunk tables built
fresh, keeping the ledger uniform). Ladders m in {32..4096} per k in {1,2,4,8}, gate
sketched moment residual < 0.05, early-abandon by decay projection (with the tick-181
caveat noted: projections can falsely abandon — any '-' at k=8 gets one direct check at
m=4096 before being trusted). Bits proxy as tick 181.
"""
import json
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
STEPS, BATCH, LR, GATE, N_EST, TAU = 9000, 2048, 3e-3, 0.05, 1024, 8.0
KS = (1, 2, 4, 8)
MS = (32, 64, 128, 256, 512, 1024, 2048, 4096)

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()
QP_CPU = QP.cpu()


@torch.no_grad()
def block1_input(idx):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    B, T = idx.shape
    blk = m.transformer.h[0]
    x = blk.lambdas[0] * x + blk.lambdas[1] * x0
    a = blk.attn
    hcur = F.rms_norm(x, (x.size(-1),))
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]

    def qk(lin):
        z = lin(hcur).view(B, T, NH, HD)
        return apply_rot(F.rms_norm(z, (HD,)), cos, sin)

    v = a.c_v(hcur).view(B, T, NH, HD)
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    q, k = qk(a.c_q), qk(a.c_k)
    q2, k2 = qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
    s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
    x = x + a.c_proj(y)
    x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    blk1 = m.transformer.h[1]
    return blk1.lambdas[0] * x + blk1.lambdas[1] * x0


print('estimating shrunk means...', flush=True)
sum_x = torch.zeros(V, D, device=DEV)
cnt = torch.zeros(V, device=DEV)
with torch.no_grad():
    for i in range(0, N_EST, 4):
        b = COOC[i:i + 4].to(DEV)
        idx = b[:, :-1]
        x = block1_input(idx).float().reshape(-1, D)
        ids = idx.reshape(-1)
        sum_x.index_add_(0, ids, x)
        cnt.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
wte = m.transformer.wte.weight.detach().float().to(DEV)
mean_x = torch.where((cnt > 0)[:, None], sum_x / cnt[:, None].clamp_min(1), wte)
shr = (cnt / (cnt + TAU))[:, None] * mean_x + (TAU / (cnt + TAU))[:, None] * wte
del sum_x, mean_x
torch.cuda.empty_cache()
a0 = m.transformer.h[0].attn
a1 = m.transformer.h[1].attn
with torch.no_grad():
    Vv0 = a0.c_v(F.rms_norm(wte, (D,))).view(V, NH, HD).float()
    xn = F.rms_norm(shr, (D,))
    K1 = F.rms_norm(a1.c_k(xn).view(V, NH, HD).float(), (HD,))
    K2 = F.rms_norm(a1.c_k2(xn).view(V, NH, HD).float(), (HD,))
    v_new = a1.c_v(xn).view(V, NH, HD).float()
    Vm = (1 - a1.lamb) * v_new + a1.lamb * Vv0.view_as(v_new)
del shr, xn, v_new
torch.cuda.empty_cache()


def train_triple(Y, m_atoms, k_code, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = Y[torch.randperm(len(Y), generator=g)[:m_atoms].to(DEV)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = (Y * QP[:, None]).sum(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=LR)
    fired = torch.zeros(m_atoms, device=DEV)
    for step in range(STEPS):
        kk = max(k_code, int(round(2 * k_code - k_code * min(1.0, 2 * step / STEPS))))
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
    return rec


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


res = {'steps': STEPS, 'gate': GATE, 'ladder': {}}
OUT = f'{QK}/qk_l1_capacity.json'
for h in range(NH):
    Y = torch.cat([K1[:, h], K2[:, h], Vm[:, h]], 1)
    for kc in KS:
        key = f'h{h}_k{kc}'
        pts, prev = [], None
        for mm in MS:
            if mm < 2 * kc:
                continue
            rec = train_triple(Y, mm, kc)
            mres = moment_residual(Y, rec)
            pts.append({'m': mm, 'res': round(mres, 4),
                        'Mbit': round(bits_proxy(mm, kc) / 1e6, 2)})
            print(f'{key} m={mm}: res {mres:.4f}' + (' PASS' if mres < GATE else ''),
                  flush=True)
            res['ladder'][key] = pts
            json.dump(res, open(OUT, 'w'), indent=2)
            torch.cuda.empty_cache()
            if mres < GATE:
                break
            if prev is not None and mm >= 256:
                ratio = mres / max(prev, 1e-9)
                if ratio > 0.98:
                    print(f'{key}: plateau, abandoning', flush=True)
                    break
                proj = mres * ratio ** math.log2(4096 / mm)
                if proj > 1.5 * GATE:
                    print(f'{key}: projected {proj:.3f}, abandoning', flush=True)
                    break
            prev = mres
# direct checks for any k=8 ladder that never passed (tick-181 false-abandon lesson)
for h in range(NH):
    key = f'h{h}_k8'
    pts = res['ladder'].get(key, [])
    if not any(p['res'] < GATE for p in pts) and (not pts or pts[-1]['m'] < 4096):
        Y = torch.cat([K1[:, h], K2[:, h], Vm[:, h]], 1)
        rec = train_triple(Y, 4096, 8, seed=0)
        mres = moment_residual(Y, rec)
        pts.append({'m': 4096, 'res': round(mres, 4), 'direct_check': True,
                    'Mbit': round(bits_proxy(4096, 8) / 1e6, 2)})
        res['ladder'][key] = pts
        print(f'{key} DIRECT m=4096: res {mres:.4f}' + (' PASS' if mres < GATE else ''),
              flush=True)
        json.dump(res, open(OUT, 'w'), indent=2)
print('L1 CAPACITY DONE', flush=True)
