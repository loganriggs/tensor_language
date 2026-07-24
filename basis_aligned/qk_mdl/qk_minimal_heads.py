"""TICK 188 (Logan): (A) rebuild the seven scaffold heads at their BITS-OPTIMAL minimal
configurations from the tick-181 capacity frontier, warm-started-polish them, and
re-validate with the corrected null statistic — these become the displayed inventories.
(B) Direct-path "copy score" for every archetype of every head: decode the archetype's
write vector through the value dictionary -> W_o -> unembedding (logit-lens direct path,
final-rms approximation noted), and compare the boosted-token profile against the
archetype's attended key class. copy_cos > 0 means the archetype boosts the very tokens
it attends to (copy-like); < 0 anti-copy; ~0 means it routes a signal consumed elsewhere
(layers 1-17, out of scope). (C) For heads 0/4 (polished, from tick 187/187b blobs):
per-archetype branch agreement cos(S1 A_r, S2 B_r) in token space, for the artifact's
symmetric-vs-asymmetric badges, plus their copy scores via the value-mode dictionary.

Minimal configs (bits-optimal): h1 (512,2) h2 (32,1) h3 (512,2) h5 (256,2) h6 (256,1)
h7 (512,4) h8 (128,1). Polish gamma_max=0.025 (gate-aware, tick-187b lesson).
CP rank min(32, m//2). Outputs: qk_minimal_heads.json (+ .pt with everything).
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
GAMMA_MAX, JSTEPS, N_PROBE = 0.025, 4000, 8
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
    Wo = a0.c_proj.weight.detach().float().view(D, NH, HD)
    LM = (m.lm_head.weight if hasattr(m, 'lm_head') else m.transformer.wte.weight).detach().float().to(DEV)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()
QP_CPU = QP.cpu()
FR = torch.empty(V, dtype=torch.long)
FR[QP.argsort(descending=True).cpu()] = torch.arange(V)


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
    return Dm.detach(), b.detach(), We.detach()


def encode(Y, Dm, b, We, kc):
    with torch.no_grad():
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = torch.relu((Y - b) @ We.T)
        vals, idx = z.topk(kc, dim=1)
        rec = b + (vals.unsqueeze(-1) * Dn[idx]).sum(1)
    return idx, vals, rec


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
        self.scale = float(vals.norm().clamp_min(1e-30))
        self.ai, self.bi, self.ci = ai, bi, ci
        self.vals = vals / self.scale

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


def cp_fit_sym(sp, R, seed, n_starts=6):
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
                out = sp.mode_mat(0, u, u)
                if lv.numel():
                    out -= Um @ (lv * (Um.T @ u) ** 2)
                u = out.clamp_min(0)
                n = float(u.norm())
                if n < 1e-20:
                    break
                u = u / n
            lam = sp.triple(u, u, u) - (float((lv * (Um.T @ u) ** 3).sum()) if lv.numel() else 0.0)
            if best is None or lam > best[0]:
                best = (lam, u)
        if best[0] <= 0:
            break
        Us.append(best[1])
        lams.append(best[0])
        Um = torch.stack(Us, 1)
        lv = torch.tensor(lams, device=DEV)
    return Um, lv


def eval_on_core(sp, A, B, C):
    R = A.shape[1]
    h = torch.tensor([sp.triple(A[:, r], B[:, r], C[:, r]) for r in range(R)], device=DEV)
    G = (A.T @ A) * (B.T @ B) * (C.T @ C)
    lam = torch.clamp(torch.linalg.solve(G + 1e-8 * torch.eye(R, device=DEV), h), min=0)
    L = float(torch.linalg.eigvalsh(G)[-1].clamp_min(1e-12))
    for _ in range(300):
        lam = torch.clamp(lam - (G @ lam - h) / L, min=0)
    res2 = 1.0 - 2.0 * float(lam @ h) + float(lam @ G @ lam)
    return max(res2, 0.0) ** 0.5


def toks(load, n):
    top = load.argsort(descending=True)[:n]
    mx = float(load[top[0]].clamp_min(1e-12))
    return [[tok.decode([t]).replace('\n', '\\n'), int(FR[t]), round(float(load[t]) / mx, 3)]
            for t in top.tolist()]


def copy_info(h, w_head, attend_load):
    """Direct path: head-space write vector -> W_o -> unembedding. Returns top boosted
    tokens and cosine between the boosted-logit profile and the attended-class loading."""
    gl = LM @ (Wo[:, h] @ w_head)                     # (V,) direct-path logit deltas
    a = attend_load.clamp_min(0)
    ccos = float(F.cosine_similarity(gl, a, dim=0))
    top = gl.argsort(descending=True)[:6]
    return {'copy_cos': round(ccos, 3),
            'boosts': [tok.decode([t]).replace('\n', '\\n') for t in top.tolist()]}


CONFIGS = {1: (512, 2), 2: (32, 1), 3: (512, 2), 5: (256, 2), 6: (256, 1), 7: (512, 4), 8: (128, 1)}
out = {}
save = {}
for h, (mm, kc) in CONFIGS.items():
    Y = torch.cat([K1[:, h], K2[:, h], Vv[:, h]], 1)
    Dm0, b0, We0 = train_sae(Y, mm, kc, seed=0)
    idx0, coeff0, rec0 = encode(Y, Dm0, b0, We0, kc)
    mres0 = moment_residual(Y, rec0)
    sp0 = AsymCore(*build_core([idx0] * 3, [coeff0] * 3, mm), mm)
    R = min(32, mm // 2)
    U0, lam0 = cp_fit_sym(sp0, R, 0)
    rel0 = eval_on_core(sp0, U0, U0, U0)
    scale0 = sp0.scale
    del sp0
    torch.cuda.empty_cache()
    # polish (gamma 0.025, gate-aware)
    g = torch.Generator(device='cpu').manual_seed(11 + h)
    Dm = Dm0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    We = We0.clone().requires_grad_(True)
    B = (U0 * (V * scale0 * lam0.clamp_min(1e-12)) ** (1.0 / 3.0)).clone().requires_grad_(True)
    opt = torch.optim.Adam([Dm, b, We, B], lr=1e-3)
    ema = 1.0
    for step in range(JSTEPS):
        gamma = GAMMA_MAX * max(0.0, (step - 500) / (JSTEPS - 500))
        bi = torch.multinomial(QP_CPU, 4096, replacement=True, generator=g).to(DEV)
        y = Y[bi]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = torch.relu((y - b) @ We.T)
        with torch.no_grad():
            tidx = z.topk(kc, dim=1).indices
        s = torch.zeros_like(z).scatter_(1, tidx, torch.gather(z, 1, tidx))
        loss = ((b + s @ Dn - y) ** 2).mean()
        if gamma > 0:
            mm3 = 0.0
            for _ in range(N_PROBE):
                u, v_, wv = (torch.randn(mm, generator=g).to(DEV) for _ in range(3))
                m3 = ((s @ u) * (s @ v_) * (s @ wv)).mean()
                cp3 = ((B.clamp_min(0).T @ u) * (B.clamp_min(0).T @ v_)
                       * (B.clamp_min(0).T @ wv)).sum() / V
                mm3 = mm3 + (m3 - cp3) ** 2
            mm3 = mm3 / N_PROBE
            ema = 0.99 * ema + 0.01 * float(mm3)
            loss = loss + gamma * mm3 / max(ema, 1e-20)
        opt.zero_grad()
        loss.backward()
        opt.step()
    Dm, b, We = Dm.detach(), b.detach(), We.detach()
    idxJ, coeffJ, recJ = encode(Y, Dm, b, We, kc)
    mresJ = moment_residual(Y, recJ)
    spJ = AsymCore(*build_core([idxJ] * 3, [coeffJ] * 3, mm), mm)
    UJ, lamJ = cp_fit_sym(spJ, R, 0)
    relJ = eval_on_core(spJ, UJ, UJ, UJ)
    # corrected null on the polished inventory
    S = torch.zeros(V, mm, device=DEV)
    S.scatter_(1, idxJ, coeffJ)
    gp = torch.Generator().manual_seed(7)
    Sp = S.clone()
    for f in range(mm):
        Sp[:, f] = Sp[torch.randperm(V, generator=gp).to(DEV), f]
    vn, in_ = Sp.topk(min(12, mm), dim=1)
    del Sp
    sp_n = AsymCore(*build_core([in_] * 3, [vn] * 3, mm), mm)
    Un, _ = cp_fit_sym(sp_n, R, 0)
    null_on_real = eval_on_core(spJ, Un, Un, Un)
    del sp_n, spJ
    torch.cuda.empty_cache()
    Dnn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    archs = []
    for r in range(UJ.shape[1]):
        load = S @ UJ[:, r]
        w_head = Dnn[:, 2 * HD:].T @ UJ[:, r]
        archs.append({'lam': round(float(lamJ[r]), 4), 'tok': toks(load, 8),
                      'copy': copy_info(h, w_head, load)})
    out[f'h{h}'] = {'form': 'sym', 'm': mm, 'k': kc,
                    'stage': {'mres': round(mres0, 4), 'cp': round(rel0, 4)},
                    'polish': {'mres': round(mresJ, 4), 'cp': round(relJ, 4),
                               'null_on_real': round(null_on_real, 4)},
                    'arch': archs}
    save[f'h{h}'] = {'Dm': Dm.cpu(), 'b': b.cpu(), 'We': We.cpu(),
                     'U': UJ.cpu(), 'lam': lamJ.cpu()}
    print(f'h{h} (m={mm},k={kc}): stage mres {mres0:.4f} cp {rel0:.4f} | polish mres {mresJ:.4f} '
          f'cp {relJ:.4f} null {null_on_real:.4f} | copy_cos top-arch '
          f'{archs[0]["copy"]["copy_cos"]}', flush=True)
    json.dump(out, open(f'{QK}/qk_minimal_heads.json', 'w'), indent=1)
    torch.save(save, f'{QK}/qk_minimal_heads.pt')
    del S
    torch.cuda.empty_cache()

# ---- heads 0/4: branch agreement + copy scores from polished blobs ----
for h, blobf, mm in ((0, f'{QK}/qk_h0_polish_g025.pt', 2048), (4, f'{QK}/qk_h04_polish.pt', 1024)):
    bb = torch.load(blobf, map_location=DEV)
    idxs, coeffs = bb[f'h{h}_idxsJ'], bb[f'h{h}_coeffsJ']
    A, B, C, lam = bb[f'h{h}_AJ'], bb[f'h{h}_BJ'], bb[f'h{h}_CJ'], bb[f'h{h}_lamJ']
    A, B, C, lam = A.to(DEV), B.to(DEV), C.to(DEV), lam.to(DEV)
    Ss = []
    for t in range(3):
        Sd = torch.zeros(V, mm, device=DEV)
        Sd.scatter_(1, idxs[t].to(DEV), coeffs[t].to(DEV))
        Ss.append(Sd)
    Dv = bb[f'h{h}_v_Dm'].to(DEV)
    Dv = Dv / Dv.norm(dim=1, keepdim=True).clamp(min=1e-8)
    rows = []
    for r in range(A.shape[1]):
        l1, l2 = Ss[0] @ A[:, r], Ss[1] @ B[:, r]
        agree = float(F.cosine_similarity(l1, l2, dim=0))
        w_head = Dv.T @ C[:, r]
        rows.append({'lam': round(float(lam[r]), 4),
                     'branch_agree': round(agree, 3),
                     'copy': copy_info(h, w_head, (l1.clamp_min(0) * l2.clamp_min(0)).sqrt())})
    out[f'h{h}_meta'] = rows
    print(f'h{h}: branch-agree mean {np.mean([r["branch_agree"] for r in rows[:32]]):.3f} '
          f'copy_cos mean {np.mean([r["copy"]["copy_cos"] for r in rows[:32]]):.3f}', flush=True)
    json.dump(out, open(f'{QK}/qk_minimal_heads.json', 'w'), indent=1)
    del Ss
    torch.cuda.empty_cache()
print('MINIMAL HEADS DONE', flush=True)
