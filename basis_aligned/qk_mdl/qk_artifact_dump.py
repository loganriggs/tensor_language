"""TICK 186a: full archetype inventory dump for the model-features artifact.

Per gated head (1,2,3,5,6,7,8): m=512 codes from qk_stage1_triple.pt (unigram+nonneg),
symmetric CP R=32 seed 0 (identical fits to tick 184); per archetype: lambda, top-10
tokens by code loading, plus each token's unigram frequency rank (to distinguish
scaffold classes from rare-token features).
Heads 0/4: asymmetric mode-separated form (SAEs retrained deterministically, seed 0,
configs (2048,4)/(1024,4) per mode — identical to ticks 183/185), asymmetric CP R=64
seed 0; per archetype: lambda, top-8 tokens per mode (branch-1 keys, branch-2 keys,
values). Factors and SAE codes saved to qk_artifact_dump.pt for reuse.
Output: qk_artifact_dump.json.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
from tier2_folding import branch_factors
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
STEPS, BATCH, LR = 12000, 2048, 3e-3
tok = AutoTokenizer.from_pretrained('gpt2')

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
FREQ_RANK = torch.empty(V, dtype=torch.long)
FREQ_RANK[QP.argsort(descending=True).cpu()] = torch.arange(V)


def train_sae(Y, m_atoms, k_code, seed=0):
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
    return idx, vals.detach()


@torch.no_grad()
def build_core(idxs, coeffs, mm):
    ka, kb, kc_ = (ii.shape[1] for ii in idxs)
    keys_all, vals_all = [], []
    for s in range(0, V, 4096):
        i1, i2, i3 = (ii[s:s + 4096].long() for ii in idxs)
        c1, c2, c3 = (cc[s:s + 4096] for cc in coeffs)
        w = QP[s:s + 4096, None] * c1
        a = i1[:, :, None, None].expand(-1, ka, kb, kc_)
        b_ = i2[:, None, :, None].expand(-1, ka, kb, kc_)
        c_ = i3[:, None, None, :].expand(-1, ka, kb, kc_)
        v = w[:, :, None, None] * c2[:, None, :, None] * c3[:, None, None, :]
        keys_all.append(((a * mm + b_) * mm + c_).reshape(-1))
        vals_all.append(v.reshape(-1))
    uk, inv = torch.unique(torch.cat(keys_all), return_inverse=True)
    cv = torch.zeros(len(uk), device=DEV)
    cv.scatter_add_(0, inv, torch.cat(vals_all))
    return (torch.div(uk, mm * mm, rounding_mode='floor'),
            torch.div(uk, mm, rounding_mode='floor') % mm, uk % mm, cv)


class AsymCore:
    def __init__(self, ai, bi, ci, vals, mm):
        self.mm = mm
        self.ai, self.bi, self.ci = ai, bi, ci
        self.vals = vals / vals.norm().clamp_min(1e-30)

    def mode_mat(self, mode, x, y):
        out = torch.zeros(self.mm, device=DEV)
        if mode == 0:
            out.scatter_add_(0, self.ai, self.vals * x[self.bi] * y[self.ci])
        elif mode == 1:
            out.scatter_add_(0, self.bi, self.vals * x[self.ai] * y[self.ci])
        else:
            out.scatter_add_(0, self.ci, self.vals * x[self.ai] * y[self.bi])
        return out

    def triple(self, a, b, c):
        return float((self.vals * a[self.ai] * b[self.bi] * c[self.ci]).sum())


def cp_fit(sp, R, seed, iters=40, n_starts=6, symmetric=False):
    mm = sp.mm
    gg = torch.Generator().manual_seed(seed)
    A = torch.zeros(mm, 0, device=DEV)
    B = torch.zeros(mm, 0, device=DEV)
    C = torch.zeros(mm, 0, device=DEV)
    lv = torch.zeros(0, device=DEV)

    def res_mode(mode, x, y):
        out = sp.mode_mat(mode, x, y)
        if lv.numel():
            if mode == 0:
                out -= A @ (lv * (B.T @ x) * (C.T @ y))
            elif mode == 1:
                out -= B @ (lv * (A.T @ x) * (C.T @ y))
            else:
                out -= C @ (lv * (A.T @ x) * (B.T @ y))
        return out

    lams = []
    for r in range(R):
        best = None
        for s in range(n_starts):
            if symmetric:
                u = torch.rand(mm, generator=gg).to(DEV)
                u = u / u.norm()
                for _ in range(60):
                    u = res_mode(0, u, u).clamp_min(0)
                    n = float(u.norm())
                    if n < 1e-20:
                        break
                    u = u / n
                a = b = c = u
            else:
                a, b, c = (torch.rand(mm, generator=gg).to(DEV) for _ in range(3))
                a, b, c = a / a.norm(), b / b.norm(), c / c.norm()
                for _ in range(iters):
                    a = res_mode(0, b, c).clamp_min(0)
                    a = a / a.norm().clamp_min(1e-20)
                    b = res_mode(1, a, c).clamp_min(0)
                    b = b / b.norm().clamp_min(1e-20)
                    c = res_mode(2, a, b).clamp_min(0)
                    nc = float(c.norm())
                    if nc < 1e-20:
                        break
                    c = c / nc
            lam = sp.triple(a, b, c) - (float((lv * (A.T @ a) * (B.T @ b) * (C.T @ c)).sum())
                                        if lv.numel() else 0.0)
            if best is None or lam > best[0]:
                best = (lam, a, b, c)
        if best[0] <= 0:
            break
        lams.append(best[0])
        A = torch.cat([A, best[1][:, None]], 1)
        B = torch.cat([B, best[2][:, None]], 1)
        C = torch.cat([C, best[3][:, None]], 1)
        lv = torch.tensor(lams, device=DEV)
    return A, B, C, lv


def tokens_of(load, n=10):
    top = load.argsort(descending=True)[:n]
    return [{'t': tok.decode([x]).replace('\n', '\\n'),
             'fr': int(FREQ_RANK[x])} for x in top.tolist()]


out = {}
blob512 = torch.load(f'{QK}/qk_stage1_triple.pt', map_location=DEV)
save = {}
for h in (1, 2, 3, 5, 6, 7, 8):
    key = f'h{h}_unigram_nonneg'
    idx = blob512[f'{key}_idx'].long().to(DEV)
    coeff = blob512[f'{key}_coeff'].to(DEV)
    mm = 512
    sp = AsymCore(*build_core([idx] * 3, [coeff] * 3, mm), mm)
    U, _, _, lv = cp_fit(sp, 32, 0, symmetric=True)
    S = torch.zeros(V, mm, device=DEV)
    S.scatter_(1, idx, coeff)
    archs = []
    for r in range(U.shape[1]):
        archs.append({'lam': round(float(lv[r]), 4), 'tokens': tokens_of(S @ U[:, r])})
    out[f'h{h}'] = {'form': 'symmetric', 'm': mm, 'k': 6, 'archetypes': archs}
    save[f'h{h}_U'] = U.cpu()
    save[f'h{h}_lam'] = lv.cpu()
    print(f'h{h}: {U.shape[1]} archetypes dumped', flush=True)
    json.dump(out, open(f'{QK}/qk_artifact_dump.json', 'w'), indent=1)
    del sp, S
    torch.cuda.empty_cache()

for h, (mm, kc) in ((0, (2048, 4)), (4, (1024, 4))):
    Ys = [K1[:, h], K2[:, h], Vv[:, h]]
    parts = [train_sae(Y, mm, kc, seed=0) for Y in Ys]
    idxs = [p[0] for p in parts]
    coeffs = [p[1] for p in parts]
    sp = AsymCore(*build_core(idxs, coeffs, mm), mm)
    A, B, C, lv = cp_fit(sp, 64, 0)
    Ss = []
    for t in range(3):
        Sd = torch.zeros(V, mm, device=DEV)
        Sd.scatter_(1, idxs[t], coeffs[t])
        Ss.append(Sd)
    archs = []
    for r in range(A.shape[1]):
        archs.append({'lam': round(float(lv[r]), 4),
                      'branch1': tokens_of(Ss[0] @ A[:, r], 8),
                      'branch2': tokens_of(Ss[1] @ B[:, r], 8),
                      'values': tokens_of(Ss[2] @ C[:, r], 8)})
    out[f'h{h}'] = {'form': 'asymmetric', 'm_per_mode': mm, 'k': kc, 'archetypes': archs}
    for t, nm in ((0, 'A'), (1, 'B'), (2, 'C')):
        save[f'h{h}_{nm}'] = (A, B, C)[t].cpu()
    save[f'h{h}_lam'] = lv.cpu()
    save[f'h{h}_idxs'] = [i.cpu() for i in idxs]
    save[f'h{h}_coeffs'] = [c.cpu() for c in coeffs]
    print(f'h{h}: {A.shape[1]} asymmetric archetypes dumped', flush=True)
    json.dump(out, open(f'{QK}/qk_artifact_dump.json', 'w'), indent=1)
    torch.save(save, f'{QK}/qk_artifact_dump.pt')
    del sp, Ss
    torch.cuda.empty_cache()
torch.save(save, f'{QK}/qk_artifact_dump.pt')
print('ARTIFACT DUMP DONE', flush=True)
