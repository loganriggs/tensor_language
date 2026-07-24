"""TICK 194: layer-1 mechanism ledger, Stage 1. The tick-193 port test showed layer 1's
pattern is ~99% token-identity-driven, so the validated pipeline transfers with
mean-residual tables in place of embeddings.

(A) Build and SAVE the layer-1 token tables: q1/k1/q2/k2 (block-1 projections of the
token-conditional mean residual, per-head rms-normed — identical recipe to tick 193)
plus the VALUE table with the model's exact lamb mixing:
    v_l1[t] = (1 - a1.lamb) * a1.c_v(rms(mean_x[t])) + a1.lamb * v1[t]
where v1[t] is block-0's value table (embedding-exact, = the layer-0 Vv rows).
-> qk_l1_tables.pt.

(B) Stage-1 triple SAE per layer-1 head on y_t = [k1|k2|v_l1] (384-dim), unigram +
nonneg, hardened trainer (annealed 2k->k), sketched third-moment gate < 0.05.
Standard config m=512, k=6 first; heads that fail auto-ladder to (1024, 8) and
(2048, 8) — layer-0 precedent says capacity spreads widely across heads.
Codes and dictionaries saved -> qk_l1_stage1.pt for Stages 2-3.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
from tier2_folding import branch_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
STEPS, BATCH, LR, GATE, N_EST = 12000, 2048, 3e-3, 0.05, 1024

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
    nh, hd = a.n_head, a.head_dim
    cos, sin = rope_tables(T, hd, idx.device, dt, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]

    def qk(lin):
        z = lin(hcur).view(B, T, nh, hd)
        return apply_rot(F.rms_norm(z, (hd,)), cos, sin)

    v = a.c_v(hcur).view(B, T, nh, hd)
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    q, k = qk(a.c_q), qk(a.c_k)
    q2, k2 = qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / hd
    s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / hd
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
    x = x + a.c_proj(y)
    x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    blk1 = m.transformer.h[1]
    return blk1.lambdas[0] * x + blk1.lambdas[1] * x0


print('estimating mean residuals...', flush=True)
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
seen = cnt > 0
mean_x = torch.where(seen[:, None], sum_x / cnt[:, None].clamp_min(1), 0.0)
wte = m.transformer.wte.weight.detach().float().to(DEV)
mean_x[~seen] = wte[~seen]
del sum_x, cnt
torch.cuda.empty_cache()

a0 = m.transformer.h[0].attn
a1 = m.transformer.h[1].attn
with torch.no_grad():
    Vv0 = a0.c_v(F.rms_norm(wte, (D,))).view(V, NH, HD).float()
    xn = F.rms_norm(mean_x, (D,))
    T1 = {}
    for name, lin in (('q1', a1.c_q), ('k1', a1.c_k), ('q2', a1.c_q2), ('k2', a1.c_k2)):
        z = lin(xn).view(V, NH, HD).float()
        T1[name] = F.rms_norm(z, (HD,)).contiguous()
    v_new = a1.c_v(xn).view(V, NH, HD).float()
    lamb = a1.lamb
    T1['v'] = ((1 - lamb) * v_new + lamb * Vv0.view_as(v_new)).contiguous()
torch.save({**{k: v.cpu() for k, v in T1.items()}, 'seen': seen.cpu()},
           f'{QK}/qk_l1_tables.pt')
print(f'tables saved ({int(seen.sum())}/{V} tokens seen)', flush=True)
del mean_x, xn, v_new
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


LADDER = [(512, 6), (1024, 8), (2048, 8)]
results = {}
blob = {}
for h in range(NH):
    Y = torch.cat([T1['k1'][:, h], T1['k2'][:, h], T1['v'][:, h]], 1)
    for mm, kc in LADDER:
        Dn, b, We, idx, coeff, rec = train_triple(Y, mm, kc, seed=0)
        mres = moment_residual(Y, rec)
        print(f'l1 h{h} m={mm} k={kc}: moment-rel-err {mres:.4f} '
              f'{"PASS" if mres < GATE else "FAIL"}', flush=True)
        if mres < GATE or (mm, kc) == LADDER[-1]:
            results[f'h{h}'] = {'m': mm, 'k': kc, 'moment_rel_err': round(mres, 4),
                                'gate_pass': bool(mres < GATE)}
            blob[f'h{h}_Dn'] = Dn.cpu()
            blob[f'h{h}_b'] = b.cpu()
            blob[f'h{h}_We'] = We.cpu()
            blob[f'h{h}_idx'] = idx.to(torch.int16).cpu()
            blob[f'h{h}_coeff'] = coeff.cpu()
            break
    json.dump(results, open(f'{QK}/qk_l1_stage1.json', 'w'), indent=2)
    torch.save(blob, f'{QK}/qk_l1_stage1.pt')
    torch.cuda.empty_cache()
print('L1 STAGE1 DONE', flush=True)
