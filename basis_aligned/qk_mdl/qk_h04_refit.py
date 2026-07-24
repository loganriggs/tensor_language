"""H0/H4 CAPACITY REFIT + ARCHETYPE-ANCHOR OVERLAP (tick 175): triple SAE in head space.

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



M_ATOMS, K_CODE = 1024, 8          # doubled capacity for the two gate-failing heads

def build_core(idx, coeff, m):
    """Dense symmetric core (m^3 flat) from sparse codes, p-weighted."""
    k = idx.shape[1]
    core = torch.zeros(m * m * m, device=DEV)
    w = QP[:, None] * coeff                                    # (V, k) p_t * s
    for i in range(k):
        for j in range(k):
            keys = (idx[:, i].long() * m + idx[:, j].long()) * m
            vals = w[:, i] * coeff[:, j]
            for l in range(k):
                core.scatter_add_(0, keys + idx[:, l].long(), vals * coeff[:, l])
    return core.view(m, m, m)


def cp_fit(core_raw, R, seed, n_starts=8, iters=60):
    """Symmetric nonneg CP via tensor power iteration + deflation (passed the planted
    known-answer test at 0.9998 matched-cos; ALS and gradient variants all failed it)."""
    m = core_raw.shape[0]
    gg = torch.Generator().manual_seed(seed)
    scale = core_raw.norm().clamp_min(1e-30)
    res = (core_raw / scale).clone()
    nrm2 = float((res ** 2).sum())
    Us, lams = [], []
    for r in range(R):
        M1 = res.reshape(m, m * m)
        best_u, best_lam = None, -1.0
        for s in range(n_starts):
            u = torch.rand(m, generator=gg).to(DEV)
            u = u / u.norm()
            for _ in range(iters):
                u = (M1 @ (u[:, None] * u[None, :]).reshape(-1)).clamp_min(0)
                n = float(u.norm())
                if n < 1e-20:
                    break
                u = u / n
            lam = float(torch.einsum('abc,a,b,c->', res, u, u, u))
            if lam > best_lam:
                best_lam, best_u = lam, u
        if best_lam <= 0:
            break
        Us.append(best_u)
        lams.append(best_lam)
        res = res - best_lam * torch.einsum('a,b,c->abc', best_u, best_u, best_u)
    U = torch.stack(Us, 1)
    rel = float(res.norm()) / max(nrm2, 1e-30) ** 0.5
    return U, torch.tensor(lams, device=DEV), rel



import json
results = json.load(open(f'{QK}/qk_stage23.json'))
out2 = {}
for h in (0, 4):
    Y = torch.cat([K1[:, h], K2[:, h], Vv[:, h]], 1)
    Dn, b, We, idx, coeff, rec = train_triple(Y, QP, seed=0, nonneg=True)
    mres = moment_residual(Y, rec, QP)
    w = QP
    r2 = 1 - float((w[:, None] * (rec - Y) ** 2).sum()
                   / (w[:, None] * (Y - (w[:, None] * Y).sum(0)) ** 2).sum())
    out2[f'h{h}'] = {'recon_r2': round(r2, 4), 'moment_rel_err': round(mres, 4),
                     'gate': 'PASS' if mres < 0.05 else 'FAIL'}
    print(f'h{h} refit m=1024 k=8: R2 {r2:.4f} moment-rel-err {mres:.4f} '
          f'{out2[f"h{h}"]["gate"]}', flush=True)
    if mres < 0.05:
        core = build_core(idx, coeff, M_ATOMS)
        U, lam, rel = cp_fit(core, 32, 0)
        S_dense = torch.zeros(V, M_ATOMS, device=DEV)
        S_dense.scatter_(1, idx, coeff)
        arch = []
        for r in lam.argsort(descending=True)[:5].tolist():
            load = S_dense @ U[:, r]
            top = load.argsort(descending=True)[:8]
            arch.append([tok.decode([t]).replace(chr(10), '\\n') for t in top.tolist()])
        out2[f'h{h}']['R32_relerr'] = round(rel, 4)
        out2[f'h{h}']['top_archetype_tokens'] = arch
        print(f'  R32 rel-err {rel:.4f}; top archetype: {arch[0]}', flush=True)
        del core, S_dense
        torch.cuda.empty_cache()
json.dump(out2, open(f'{QK}/qk_h04_refit.json', 'w'), indent=2)

# archetype-vs-anchor overlap for the 7 stage-23 heads
err = torch.load(f'{QK}/qk_err_explore.pt', map_location=DEV, weights_only=True)
anchors = set((err['q_err'] + err['k_scat'] + err['k_coh']).argsort(descending=True)[:256].tolist())
blob1 = torch.load(f'{QK}/qk_stage1_triple.pt', map_location=DEV)
overlaps = {}
gtok = torch.Generator().manual_seed(0)
for h in (1, 2, 3, 5, 6, 7, 8):
    key = f'h{h}_unigram_nonneg'
    idx_h = blob1[f'{key}_idx'].long().to(DEV)
    coeff_h = blob1[f'{key}_coeff'].to(DEV)
    core = build_core(idx_h, coeff_h, 512)
    U, lam, _ = cp_fit(core, 32, 0)
    S_dense = torch.zeros(V, 512, device=DEV)
    S_dense.scatter_(1, idx_h, coeff_h)
    hit = tot = 0
    for r in lam.argsort(descending=True)[:5].tolist():
        top = (S_dense @ U[:, r]).argsort(descending=True)[:32]
        hit += sum(1 for t in top.tolist() if t in anchors)
        tot += 32
    overlaps[f'h{h}'] = round(hit / tot, 3)
    print(f'h{h} archetype-anchor overlap (top-5 archetypes x top-32 tokens): '
          f'{hit}/{tot} = {hit/tot:.3f} (random baseline 0.005)', flush=True)
    del core, S_dense
    torch.cuda.empty_cache()
json.dump(overlaps, open(f'{QK}/qk_arch_anchor_overlap.json', 'w'), indent=2)
print('TICK175 DONE', flush=True)
