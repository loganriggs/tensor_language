"""SPEC STAGE 1 on the real head (tick 172): triple SAE in head space.

Rows per head: y_t = [k1_t | k2_t | v_t] in R^384 (folded unit-RMS key factors, both
branches, plus the value head-space vector — spec option 1a). Hardened trainer from tick
171 (k annealed 2k->k), in BOTH code signs (nonneg vs signed — planted gate passed with
nonneg, but real rows may need signed), and BOTH p weightings (unigram vs uniform).
Gate (spec check 4): sketched third-moment residual with 256 random probe triples —
rel_err = E[(T(u,v,w) - That(u,v,w))^2] / E[T(u,v,w)^2], T from raw rows, That from
reconstructions, p-weighted. Per-token reconstruction R^2 reported but NOT the gate.
Saves codes (indices + coefficients) and atoms per head/config -> qk_stage1_triple.pt,
metrics -> qk_stage1_triple.json. Stage 2 (sparse symmetric core) consumes the winner.
"""
import json
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
from tier2_folding import branch_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
M_ATOMS, K_CODE, STEPS, BATCH, LR = 512, 6, 12000, 2048, 3e-3
import numpy as np

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


def train_triple(Y, pw, seed=0, nonneg=True):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = Y[torch.randperm(len(Y), generator=g)[:M_ATOMS].to(DEV)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = (Y * pw[:, None]).sum(0) if pw is not None else Y.mean(0)
    b = b.clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=LR)
    fired = torch.zeros(M_ATOMS, device=DEV)
    pw_cpu = pw.cpu() if pw is not None else None
    for step in range(STEPS):
        kk = max(K_CODE, int(round(2 * K_CODE - K_CODE * min(1.0, 2 * step / STEPS))))
        if pw_cpu is not None:
            bi = torch.multinomial(pw_cpu, BATCH, replacement=True, generator=g).to(DEV)
        else:
            bi = torch.randint(0, len(Y), (BATCH,), generator=g).to(DEV)
        y = Y[bi]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (y - b) @ We.T
        if nonneg:
            z = torch.relu(z)
        vals, idx = z.abs().topk(kk, dim=1)
        coeff = torch.gather(z, 1, idx)
        yhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
        fired.index_add_(0, idx.reshape(-1), (coeff.abs() > 1e-8).float().reshape(-1))
        loss = ((yhat - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            dead = (fired == 0).nonzero().squeeze(1)
            if len(dead):
                with torch.no_grad():
                    Dn_ = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    z_ = (Y - b) @ We.T
                    if nonneg:
                        z_ = torch.relu(z_)
                    v_, i_ = z_.abs().topk(K_CODE, dim=1)
                    rec = b + (torch.gather(z_, 1, i_).unsqueeze(-1) * Dn_[i_]).sum(1)
                    worst = ((rec - Y) ** 2).sum(1).topk(len(dead)).indices
                    Dm.data[dead] = Y[worst] / Y[worst].norm(dim=1, keepdim=True).clamp(min=1e-8)
                    We.data[dead] = Dm.data[dead]
            fired.zero_()
    with torch.no_grad():
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (Y - b) @ We.T
        if nonneg:
            z = torch.relu(z)
        vals, idx = z.abs().topk(K_CODE, dim=1)
        coeff = torch.gather(z, 1, idx)
        rec = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
    return Dn.detach(), b.detach(), We.detach(), idx, coeff.detach(), rec.detach()


@torch.no_grad()
def moment_residual(Y, rec, pw, n_probe=256, seed=3):
    """Spec check 4: sketched third-moment relative error, p-weighted."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    num = den = 0.0
    w = pw if pw is not None else torch.full((len(Y),), 1.0 / len(Y), device=DEV)
    for _ in range(n_probe):
        u, v_, wv = (torch.randn(Y.shape[1], generator=g).to(DEV) for _ in range(3))
        t = (w * (Y @ u) * (Y @ v_) * (Y @ wv)).sum()
        th = (w * (rec @ u) * (rec @ v_) * (rec @ wv)).sum()
        num += float((t - th) ** 2)
        den += float(t ** 2)
    return num / max(den, 1e-30)


results = {}
blob = {}
for h in range(NH):
    Y = torch.cat([K1[:, h], K2[:, h], Vv[:, h]], 1)          # (V, 384)
    for pname, pw in (('unigram', QP), ('uniform', None)):
        for sname, nn in (('nonneg', True), ('signed', False)):
            key = f'h{h}_{pname}_{sname}'
            Dn, b, We, idx, coeff, rec = train_triple(Y, pw, seed=0, nonneg=nn)
            w = pw if pw is not None else torch.full((V,), 1.0 / V, device=DEV)
            r2 = 1 - float((w[:, None] * (rec - Y) ** 2).sum()
                           / (w[:, None] * (Y - (w[:, None] * Y).sum(0)) ** 2).sum())
            mres = moment_residual(Y, rec, pw)
            results[key] = {'recon_r2': round(r2, 4), 'moment_rel_err': round(mres, 4)}
            blob[f'{key}_Dn'] = Dn.cpu()
            blob[f'{key}_b'] = b.cpu()
            blob[f'{key}_idx'] = idx.to(torch.int16).cpu()
            blob[f'{key}_coeff'] = coeff.cpu()
            print(f'{key}: R2 {r2:.4f} moment-rel-err {mres:.4f}', flush=True)
    torch.save(blob, f'{QK}/qk_stage1_triple.pt')
    json.dump(results, open(f'{QK}/qk_stage1_triple.json', 'w'), indent=2)
print('STAGE1 DONE', flush=True)
