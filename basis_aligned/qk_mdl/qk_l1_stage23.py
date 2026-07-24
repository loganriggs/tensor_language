"""TICK 196: layer-1 Stages 2-3 — sparse cores, CP archetypes, corrected-null
validation, stability, and token dumps for all nine layer-1 heads.

Inputs: tick-194 Stage-1 codes for the eight gated heads (qk_l1_stage1.pt); head 1
rebuilt here with SHRINKAGE tables (tick-195 cure: tau=8 toward the embedding prior) at
m=1024, k=8, and its codes saved back into the ledger. Per head: sparse symmetric core
(concatenated [k1|k2|v] codes, unigram-weighted), symmetric CP R=32 (power+deflation),
3-seed stability, CORRECTED null (factors fit on the column-permuted core, transplanted
onto the real core with nonneg lambda refit — the only valid statistic, tick 183), and
top-token dumps for the top-5 archetypes. Output: qk_l1_stage23.json (+ factors in
qk_l1_stage23.pt) — the layer-1 mechanism ledger.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
STEPS, BATCH, LR, GATE, N_EST, TAU = 12000, 2048, 3e-3, 0.05, 1024, 8.0
R_CP = 32
tok = AutoTokenizer.from_pretrained('gpt2')

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


# ---- shrunk tables for h1 rebuild ----
print('estimating means for h1 shrinkage rebuild...', flush=True)
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
shrunk = (cnt / (cnt + TAU))[:, None] * mean_x + (TAU / (cnt + TAU))[:, None] * wte
del sum_x, mean_x
torch.cuda.empty_cache()
a0 = m.transformer.h[0].attn
a1 = m.transformer.h[1].attn
with torch.no_grad():
    Vv0 = a0.c_v(F.rms_norm(wte, (D,))).view(V, NH, HD).float()
    xn = F.rms_norm(shrunk, (D,))
    k1s = F.rms_norm(a1.c_k(xn).view(V, NH, HD).float(), (HD,))
    k2s = F.rms_norm(a1.c_k2(xn).view(V, NH, HD).float(), (HD,))
    v_new = a1.c_v(xn).view(V, NH, HD).float()
    vs = (1 - a1.lamb) * v_new + a1.lamb * Vv0.view_as(v_new)
del shrunk, xn, v_new
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


@torch.no_grad()
def build_core(idx, coeff, mm):
    k = idx.shape[1]
    keys_all, vals_all = [], []
    for s in range(0, V, 4096):
        ii = idx[s:s + 4096].long()
        cc = coeff[s:s + 4096]
        w = QP[s:s + 4096, None] * cc
        aa = ii[:, :, None, None].expand(-1, k, k, k)
        bb = ii[:, None, :, None].expand(-1, k, k, k)
        cc_ = ii[:, None, None, :].expand(-1, k, k, k)
        v = w[:, :, None, None] * cc[:, None, :, None] * cc[:, None, None, :]
        keys_all.append(((aa * mm + bb) * mm + cc_).reshape(-1))
        vals_all.append(v.reshape(-1))
    uk, inv = torch.unique(torch.cat(keys_all), return_inverse=True)
    cv = torch.zeros(len(uk), device=DEV)
    cv.scatter_add_(0, inv, torch.cat(vals_all))
    return (torch.div(uk, mm * mm, rounding_mode='floor'),
            torch.div(uk, mm, rounding_mode='floor') % mm, uk % mm, cv)


class Core:
    def __init__(self, ai, bi, ci, vals, mm):
        self.mm = mm
        self.ai, self.bi, self.ci = ai, bi, ci
        self.vals = vals / vals.norm().clamp_min(1e-30)

    def mat(self, u, Um=None, lv=None):
        out = torch.zeros(self.mm, device=DEV)
        out.scatter_add_(0, self.ai, self.vals * u[self.bi] * u[self.ci])
        if Um is not None and Um.shape[1]:
            out -= Um @ (lv * (Um.T @ u) ** 2)
        return out

    def triple(self, u):
        return float((self.vals * u[self.ai] * u[self.bi] * u[self.ci]).sum())


def cp_fit(sp, R, seed, n_starts=6):
    mm = sp.mm
    gg = torch.Generator().manual_seed(seed)
    Us, lams = [], []
    Um = torch.zeros(mm, 0, device=DEV)
    lv = torch.zeros(0, device=DEV)
    for r in range(R):
        best = None
        for s in range(n_starts):
            u = torch.rand(mm, generator=gg).to(DEV)
            u = u / u.norm()
            for _ in range(60):
                u = sp.mat(u, Um, lv).clamp_min(0)
                n = float(u.norm())
                if n < 1e-20:
                    break
                u = u / n
            lam = sp.triple(u) - (float((lv * (Um.T @ u) ** 3).sum()) if lv.numel() else 0.0)
            if best is None or lam > best[0]:
                best = (lam, u)
        if best[0] <= 0:
            break
        Us.append(best[1])
        lams.append(best[0])
        Um = torch.stack(Us, 1)
        lv = torch.tensor(lams, device=DEV)
    return Um, lv


def eval_on(sp, U):
    R = U.shape[1]
    h_ = torch.tensor([sp.triple(U[:, r]) for r in range(R)], device=DEV)
    G = (U.T @ U) ** 3
    lam = torch.clamp(torch.linalg.solve(G + 1e-8 * torch.eye(R, device=DEV), h_), min=0)
    L = float(torch.linalg.eigvalsh(G)[-1].clamp_min(1e-12))
    for _ in range(300):
        lam = torch.clamp(lam - (G @ lam - h_) / L, min=0)
    res2 = 1.0 - 2.0 * float(lam @ h_) + float(lam @ G @ lam)
    return max(res2, 0.0) ** 0.5


blob = torch.load(f'{QK}/qk_l1_stage1.pt', map_location=DEV)
s1_js = json.load(open(f'{QK}/qk_l1_stage1.json'))

# rebuild h1 with shrunk tables
Y1 = torch.cat([k1s[:, 1], k2s[:, 1], vs[:, 1]], 1)
Dn, b_, We, idx1, coeff1, rec1 = train_triple(Y1, 1024, 8)
mres1 = moment_residual(Y1, rec1)
print(f'l1 h1 (shrunk) m=1024 k=8: gate {mres1:.4f} {"PASS" if mres1 < GATE else "FAIL"}',
      flush=True)
blob['h1_Dn'] = Dn.cpu()
blob['h1_b'] = b_.cpu()
blob['h1_We'] = We.cpu()
blob['h1_idx'] = idx1.to(torch.int16).cpu()
blob['h1_coeff'] = coeff1.cpu()
torch.save(blob, f'{QK}/qk_l1_stage1.pt')
s1_js['h1'] = {'m': 1024, 'k': 8, 'moment_rel_err': round(mres1, 4),
               'gate_pass': bool(mres1 < GATE), 'note': 'shrunk tau=8 tables'}
json.dump(s1_js, open(f'{QK}/qk_l1_stage1.json', 'w'), indent=2)

out = {}
save = {}
for h in range(NH):
    idx = blob[f'h{h}_idx'].long().to(DEV)
    coeff = blob[f'h{h}_coeff'].to(DEV)
    mm = int(s1_js[f'h{h}']['m'])
    sp = Core(*build_core(idx, coeff, mm), mm)
    Us3, rels = [], []
    for seed in range(3):
        U, lv = cp_fit(sp, R_CP, seed)
        Us3.append(U)
        rels.append(eval_on(sp, U))
    stab = []
    for i in range(3):
        for j in range(i + 1, 3):
            C = (Us3[i].T @ Us3[j]).abs()
            stab.append(float(C.max(1).values.mean()))
    U, lv = Us3[int(np.argmin(rels))], None
    U0, lv0 = cp_fit(sp, R_CP, int(np.argmin(rels)))
    # corrected null
    S = torch.zeros(V, mm, device=DEV)
    S.scatter_(1, idx, coeff)
    Sp = S.clone()
    gp = torch.Generator().manual_seed(7)
    for f in range(mm):
        Sp[:, f] = Sp[torch.randperm(V, generator=gp).to(DEV), f]
    vn, in_ = Sp.topk(min(12, mm), dim=1)
    del Sp
    sp_n = Core(*build_core(in_, vn, mm), mm)
    Un, _ = cp_fit(sp_n, R_CP, 0)
    null_on_real = eval_on(sp, Un)
    arch = []
    for r in lv0.argsort(descending=True)[:5].tolist():
        load = S @ U0[:, r]
        top = load.argsort(descending=True)[:8]
        arch.append([tok.decode([t]).replace('\n', '\\n') for t in top.tolist()])
    out[f'h{h}'] = {'m': mm, 'R32_relerr': round(min(rels), 4),
                    'stability': round(sum(stab) / len(stab), 3),
                    'null_on_real': round(null_on_real, 4),
                    'top_archetypes': arch}
    save[f'h{h}_U'] = U0.cpu()
    save[f'h{h}_lam'] = lv0.cpu()
    print(f'l1 h{h}: R32 {min(rels):.4f} stab {sum(stab)/len(stab):.3f} '
          f'null-on-real {null_on_real:.4f} | top arch: {arch[0][:5]}', flush=True)
    json.dump(out, open(f'{QK}/qk_l1_stage23.json', 'w'), indent=2)
    torch.save(save, f'{QK}/qk_l1_stage23.pt')
    del sp, sp_n, S
    torch.cuda.empty_cache()
print('L1 STAGE23 DONE', flush=True)
