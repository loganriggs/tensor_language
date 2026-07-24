"""JOINT TRAINING (tick 177, spec section 6, the endgame): triple SAE in head space.

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
R_CP, GAMMA_MAX, JSTEPS, N_PROBE = 32, 0.05, 4000, 8
out = {}
for h in (2, 8, 1):                                    # best / mid / worst of the gated heads
    Y = torch.cat([K1[:, h], K2[:, h], Vv[:, h]], 1)
    # stagewise warm start (deterministic -> identical to tick 172)
    Dn0, b0, We0, idx0, coeff0, rec0 = train_triple(Y, QP, seed=0, nonneg=True)
    core0 = build_core(idx0, coeff0, M_ATOMS)
    _, _, rel0 = cp_fit(core0, R_CP, 0)
    mres0 = moment_residual(Y, rec0, QP)
    w = QP
    r2_0 = 1 - float((w[:, None] * (rec0 - Y) ** 2).sum()
                     / (w[:, None] * (Y - (w[:, None] * Y).sum(0)) ** 2).sum())
    del core0
    torch.cuda.empty_cache()
    # CP warm start on stagewise codes in CODE space
    S0 = torch.zeros(V, M_ATOMS, device=DEV)
    S0.scatter_(1, idx0, coeff0)
    # joint phase
    g = torch.Generator(device='cpu').manual_seed(11 + h)
    Dm = Dn0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    We = We0.clone().requires_grad_(True)
    B = (S0[torch.multinomial(QP_CPU, R_CP, generator=g).to(DEV)].T.clone()
         .clamp_min(0) + 0.01).requires_grad_(True)     # (m, R) archetype warm-ish init
    opt = torch.optim.Adam([Dm, b, We, B], lr=1e-3)
    ema_m3 = 1.0
    for step in range(JSTEPS):
        gamma = GAMMA_MAX * max(0.0, (step - 500) / (JSTEPS - 500))
        bi = torch.multinomial(QP_CPU, 4096, replacement=True, generator=g).to(DEV)
        y = Y[bi]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = torch.relu((y - b) @ We.T)
        with torch.no_grad():
            tidx = z.topk(K_CODE, dim=1).indices
        s = torch.zeros_like(z).scatter_(1, tidx, torch.gather(z, 1, tidx))
        yhat = b + s @ Dn
        loss = ((yhat - y) ** 2).mean()
        if gamma > 0:
            mm = 0.0
            for _ in range(N_PROBE):
                u, v_, wv = (torch.randn(M_ATOMS, generator=g).to(DEV) for _ in range(3))
                m3 = ((s @ u) * (s @ v_) * (s @ wv)).mean()
                cp3 = ((B.clamp_min(0).T @ u) * (B.clamp_min(0).T @ v_)
                       * (B.clamp_min(0).T @ wv)).sum() / V
                mm = mm + (m3 - cp3) ** 2
            mm = mm / N_PROBE
            ema_m3 = 0.99 * ema_m3 + 0.01 * float(mm)
            loss = loss + gamma * mm / max(ema_m3, 1e-20)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
        z = torch.relu((Y - b) @ We.T)
        tidx = z.topk(K_CODE, dim=1).indices
        coeffJ = torch.gather(z, 1, tidx)
        recJ = b + (coeffJ.unsqueeze(-1) * Dn[tidx]).sum(1)
        r2_J = 1 - float((w[:, None] * (recJ - Y) ** 2).sum()
                         / (w[:, None] * (Y - (w[:, None] * Y).sum(0)) ** 2).sum())
    mresJ = moment_residual(Y, recJ, QP)
    coreJ = build_core(tidx, coeffJ, M_ATOMS)
    _, _, relJ = cp_fit(coreJ, R_CP, 0)
    del coreJ
    torch.cuda.empty_cache()
    out[f'h{h}'] = {'stagewise': {'r2': round(r2_0, 4), 'mres': round(mres0, 4),
                                  'cp_relerr_R32': round(rel0, 4)},
                    'joint': {'r2': round(r2_J, 4), 'mres': round(mresJ, 4),
                              'cp_relerr_R32': round(relJ, 4)}}
    print(f'h{h}: stagewise r2 {r2_0:.4f} mres {mres0:.4f} cpR32 {rel0:.4f} | '
          f'joint r2 {r2_J:.4f} mres {mresJ:.4f} cpR32 {relJ:.4f}', flush=True)
    json.dump(out, open(f'{QK}/qk_joint_g005.json', 'w'), indent=2)
print('JOINT DONE', flush=True)
