"""TICK 186b: joint training RETRY with the TRUE deflation warm start (closes the
tick-177 caveat). Single-variable change vs qk_joint.py gamma=0.05 arm: the archetype
matrix B is initialized from the stagewise deflation CP solution
    B_r = (V * scale0 * lam_r)^(1/3) u_r
(so that sum_r B_r^(x3) / V equals the stagewise p-weighted code core exactly at step 0),
instead of tick-177's random rows of the code matrix. Same heads (2, 8, 1), same gamma
ramp to 0.05 after step 500, same steps/lr/seeds.

Extra metrics beyond tick 177: (a) archetype drift — matched cosine between the refit
archetypes on the joint core and the stagewise archetypes U0; (b) corrected-statistic
transfer — rel-err of U0 (lambda refit) on the JOINT core, i.e. does joint training
preserve the stagewise mechanism even if refit finds different components.
Verdict rule: joint wins only if moment residual stays gated AND cp rel-err does not
degrade materially vs stagewise (tick-177 failure mode was 0.03->0.45 etc.).
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
from tier2_folding import branch_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
M_ATOMS, K_CODE, STEPS, BATCH, LR = 512, 6, 12000, 2048, 3e-3
R_CP, GAMMA_MAX, JSTEPS, N_PROBE = 32, 0.05, 4000, 8

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


def train_triple(Y, pw, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = Y[torch.randperm(len(Y), generator=g)[:M_ATOMS].to(DEV)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = (Y * pw[:, None]).sum(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=LR)
    fired = torch.zeros(M_ATOMS, device=DEV)
    for step in range(STEPS):
        kk = max(K_CODE, int(round(2 * K_CODE - K_CODE * min(1.0, 2 * step / STEPS))))
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
                    v_, i_ = z_.topk(K_CODE, dim=1)
                    rec = b + (v_.unsqueeze(-1) * Dn_[i_]).sum(1)
                    worst = ((rec - Y) ** 2).sum(1).topk(len(dead)).indices
                    Dm.data[dead] = Y[worst] / Y[worst].norm(dim=1, keepdim=True).clamp(min=1e-8)
                    We.data[dead] = Dm.data[dead]
                    del z_, rec
            fired.zero_()
    with torch.no_grad():
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = torch.relu((Y - b) @ We.T)
        vals, idx = z.topk(K_CODE, dim=1)
        rec = b + (vals.unsqueeze(-1) * Dn[idx]).sum(1)
    return Dn.detach(), b.detach(), We.detach(), idx, vals.detach(), rec.detach()


@torch.no_grad()
def moment_residual(Y, rec, pw, n_probe=256, seed=3):
    g = torch.Generator(device='cpu').manual_seed(seed)
    num = den = 0.0
    for _ in range(n_probe):
        u, v_, wv = (torch.randn(Y.shape[1], generator=g).to(DEV) for _ in range(3))
        t = (pw * (Y @ u) * (Y @ v_) * (Y @ wv)).sum()
        th = (pw * (rec @ u) * (rec @ v_) * (rec @ wv)).sum()
        num += float((t - th) ** 2)
        den += float(t ** 2)
    return num / max(den, 1e-30)


def build_core(idx, coeff, mm):
    k = idx.shape[1]
    core = torch.zeros(mm * mm * mm, device=DEV)
    w = QP[:, None] * coeff
    for i in range(k):
        for j in range(k):
            keys = (idx[:, i].long() * mm + idx[:, j].long()) * mm
            vals = w[:, i] * coeff[:, j]
            for l in range(k):
                core.scatter_add_(0, keys + idx[:, l].long(), vals * coeff[:, l])
    return core.view(mm, mm, mm)


def cp_fit(core_raw, R, seed, n_starts=8, iters=60):
    mm = core_raw.shape[0]
    gg = torch.Generator().manual_seed(seed)
    scale = core_raw.norm().clamp_min(1e-30)
    res = (core_raw / scale).clone()
    Us, lams = [], []
    for r in range(R):
        M1 = res.reshape(mm, mm * mm)
        best_u, best_lam = None, -1.0
        for s in range(n_starts):
            u = torch.rand(mm, generator=gg).to(DEV)
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
    rel = float(res.norm())
    return U, torch.tensor(lams, device=DEV), rel, float(scale)


def eval_transfer(core_raw, U):
    """Corrected statistic: rel-err on core_raw of fixed archetypes U, nonneg lambda refit."""
    scale = core_raw.norm().clamp_min(1e-30)
    cn = core_raw / scale
    R = U.shape[1]
    h = torch.einsum('abc,ar,br,cr->r', cn, U, U, U)
    G = (U.T @ U) ** 3
    lam = torch.clamp(torch.linalg.solve(G + 1e-8 * torch.eye(R, device=DEV), h), min=0)
    L = float(torch.linalg.eigvalsh(G)[-1].clamp_min(1e-12))
    for _ in range(300):
        lam = torch.clamp(lam - (G @ lam - h) / L, min=0)
    res2 = 1.0 - 2.0 * float(lam @ h) + float(lam @ G @ lam)
    return max(res2, 0.0) ** 0.5


out = {}
for h in (2, 8, 1):
    Y = torch.cat([K1[:, h], K2[:, h], Vv[:, h]], 1)
    Dn0, b0, We0, idx0, coeff0, rec0 = train_triple(Y, QP, seed=0)
    core0 = build_core(idx0, coeff0, M_ATOMS)
    U0, lam0, rel0, scale0 = cp_fit(core0, R_CP, 0)
    mres0 = moment_residual(Y, rec0, QP)
    w = QP
    r2_0 = 1 - float((w[:, None] * (rec0 - Y) ** 2).sum()
                     / (w[:, None] * (Y - (w[:, None] * Y).sum(0)) ** 2).sum())
    del core0
    torch.cuda.empty_cache()
    # joint phase — TRUE deflation warm start for B
    g = torch.Generator(device='cpu').manual_seed(11 + h)
    Dm = Dn0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    We = We0.clone().requires_grad_(True)
    B = (U0 * (V * scale0 * lam0.clamp_min(1e-12)) ** (1.0 / 3.0)).clone().requires_grad_(True)
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
            mm3 = 0.0
            for _ in range(N_PROBE):
                u, v_, wv = (torch.randn(M_ATOMS, generator=g).to(DEV) for _ in range(3))
                m3 = ((s @ u) * (s @ v_) * (s @ wv)).mean()
                cp3 = ((B.clamp_min(0).T @ u) * (B.clamp_min(0).T @ v_)
                       * (B.clamp_min(0).T @ wv)).sum() / V
                mm3 = mm3 + (m3 - cp3) ** 2
            mm3 = mm3 / N_PROBE
            ema_m3 = 0.99 * ema_m3 + 0.01 * float(mm3)
            loss = loss + gamma * mm3 / max(ema_m3, 1e-20)
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
    UJ, lamJ, relJ, _ = cp_fit(coreJ, R_CP, 0)
    drift = float((UJ.T @ U0).abs().max(1).values.mean())
    transfer0_on_J = eval_transfer(coreJ, U0)
    with torch.no_grad():
        Bn = B.clamp_min(0)
        Bn = Bn / Bn.norm(dim=0, keepdim=True).clamp_min(1e-12)
    transferB_on_J = eval_transfer(coreJ, Bn.detach())
    del coreJ
    torch.cuda.empty_cache()
    out[f'h{h}'] = {'stagewise': {'r2': round(r2_0, 4), 'mres': round(mres0, 4),
                                  'cp_relerr_R32': round(rel0, 4)},
                    'joint_warm': {'r2': round(r2_J, 4), 'mres': round(mresJ, 4),
                                   'cp_relerr_R32': round(relJ, 4),
                                   'archetype_drift_cos': round(drift, 4),
                                   'U0_transfer_on_jointcore': round(transfer0_on_J, 4),
                                   'B_transfer_on_jointcore': round(transferB_on_J, 4)},
                    'tick177_joint_cp_relerr': {'h2': 0.4517, 'h8': 0.5397, 'h1': 0.6477}[f'h{h}']}
    print(f'h{h}: stage r2 {r2_0:.4f} mres {mres0:.4f} cp {rel0:.4f} | warm-joint r2 {r2_J:.4f} '
          f'mres {mresJ:.4f} cp {relJ:.4f} (t177: {out[f"h{h}"]["tick177_joint_cp_relerr"]}) | '
          f'drift-cos {drift:.3f} U0-transfer {transfer0_on_J:.4f} B-transfer {transferB_on_J:.4f}',
          flush=True)
    json.dump(out, open(f'{QK}/qk_joint_warm.json', 'w'), indent=2)
print('JOINT WARM DONE', flush=True)
