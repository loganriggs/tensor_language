"""TICK 185: corpus-component decomposition of the mechanism cores — which kinds of
data is each head's archetype structure built from?

Corpus components: the 6000 disjoint co-occurrence documents (data_fineweb_cooc_tokens,
never used in audits) are featurized as histograms over the 256 saved token clusters
(qk_cooc_lift.pt 'assign'), then k-means'd into C=12 document components. Each component
c yields a unigram p^(c); components are named by their most over-represented tokens
(highest p^(c)/p_global among tokens appearing >= 20 times in the component).

Per head: rebuild the third-moment core under each component's p^(c) (codes fixed —
the same token codes, reweighted), then profile each fitted archetype r by its inner
product with each component core: profile[r, c] = <M_c, comp_r> / sum_c' <M_c', comp_r>.
Summary per archetype: effective number of components exp(entropy) — scaffold-syntax
archetypes should be near-uniform (mechanism is corpus-general), topical structure
component-concentrated. Also: pairwise component-core cosine per head (generalizes the
tick-176 two-slice stability to 12 slices).

Heads: the seven gated heads (symmetric, m=512 codes from qk_stage1_triple.pt, CP R=32
seed 0 — identical to tick 184) + h0/h4 (asymmetric, SAEs retrained deterministically,
CP R=32 seed 0 — identical to tick 183).
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
STEPS, BATCH, LR, C_COMP = 12000, 2048, 3e-3, 12
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

# ---------------- corpus components ----------------
DOCS = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
assign = torch.load(f'{QK}/qk_cooc_lift.pt', map_location='cpu')['assign'].long()
ND = DOCS.shape[0]
H = torch.zeros(ND, 256)
for i in range(ND):
    H[i] = torch.bincount(assign[DOCS[i]], minlength=256).float()
H = H / H.sum(1, keepdim=True).clamp_min(1e-9)
H = H.to(DEV)
g = torch.Generator().manual_seed(11)
cent = H[torch.randperm(ND, generator=g)[:C_COMP].to(DEV)].clone()
for _ in range(60):
    d = torch.cdist(H, cent)
    lab = d.argmin(1)
    for c in range(C_COMP):
        sel = lab == c
        if sel.any():
            cent[c] = H[sel].mean(0)
sizes = torch.bincount(lab, minlength=C_COMP)
PCs, names = [], []
for c in range(C_COMP):
    toks = DOCS[(lab == c).cpu()].flatten()
    cnt = torch.bincount(toks, minlength=V).float().to(DEV)
    pc = (cnt + 0.5)
    pc = pc / pc.sum()
    PCs.append(pc)
    ratio = torch.where(cnt >= 20, pc / QP, torch.zeros_like(pc))
    top = ratio.argsort(descending=True)[:8]
    names.append([tok.decode([t]).replace('\n', '\\n') for t in top.tolist()])
    print(f'component {c} ({int(sizes[c])} docs): {names[c]}', flush=True)


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
def build_core(idxs, coeffs, mm, pw):
    ka, kb, kc_ = (ii.shape[1] for ii in idxs)
    keys_all, vals_all = [], []
    for s in range(0, V, 4096):
        i1, i2, i3 = (ii[s:s + 4096].long() for ii in idxs)
        c1, c2, c3 = (cc[s:s + 4096] for cc in coeffs)
        w = pw[s:s + 4096, None] * c1
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
    def __init__(self, ai, bi, ci, vals, mm, normalize=True):
        self.mm = mm
        self.ai, self.bi, self.ci = ai, bi, ci
        self.vals = vals / vals.norm().clamp_min(1e-30) if normalize else vals

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


results = {'component_names': names, 'component_sizes': sizes.tolist()}
blob512 = torch.load(f'{QK}/qk_stage1_triple.pt', map_location=DEV)
HEADS = [(h, 'sym') for h in (1, 2, 3, 5, 6, 7, 8)] + [(0, 'asym'), (4, 'asym')]
for h, form in HEADS:
    if form == 'sym':
        key = f'h{h}_unigram_nonneg'
        idxs = [blob512[f'{key}_idx'].long().to(DEV)] * 3
        coeffs = [blob512[f'{key}_coeff'].to(DEV)] * 3
        mm = 512
    else:
        mm, kc = (2048, 4) if h == 0 else (1024, 4)
        Ys = [K1[:, h], K2[:, h], Vv[:, h]]
        parts = [train_sae(Y, mm, kc, seed=0) for Y in Ys]
        idxs = [p[0] for p in parts]
        coeffs = [p[1] for p in parts]
    sp_glob = AsymCore(*build_core(idxs, coeffs, mm, QP), mm)
    A, B, C, lv = cp_fit(sp_glob, 32, 0, symmetric=(form == 'sym'))
    comp_cores = [AsymCore(*build_core(idxs, coeffs, mm, pc), mm, normalize=False)
                  for pc in PCs]
    # pairwise component-core cosine (flattened sparse dot via shared key space)
    cos_mat = torch.zeros(C_COMP, C_COMP)
    flat = []
    for cc_ in comp_cores:
        keys = (cc_.ai * mm + cc_.bi) * mm + cc_.ci
        flat.append((keys, cc_.vals))
    for i in range(C_COMP):
        for j in range(i, C_COMP):
            ki, vi = flat[i]
            kj, vj = flat[j]
            # cores share the same support (same codes), keys align after coalesce
            dot = float((vi * vj).sum()) if len(ki) == len(kj) else float('nan')
            cos_mat[i, j] = cos_mat[j, i] = dot / max(float(vi.norm() * vj.norm()), 1e-30)
    row = {'form': form, 'm': mm,
           'min_pairwise_core_cos': round(float(cos_mat.min()), 4),
           'mean_pairwise_core_cos': round(float(cos_mat.mean()), 4)}
    # archetype x component profile
    profiles, effn = [], []
    for r in range(min(32, A.shape[1])):
        inner = torch.tensor([max(cc_.triple(A[:, r], B[:, r], C[:, r]), 0.0)
                              for cc_ in comp_cores])
        tot = float(inner.sum())
        p = inner / max(tot, 1e-30)
        profiles.append([round(float(x), 4) for x in p])
        ent = -float((p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum())
        effn.append(float(np.exp(ent)))
    row['archetype_eff_components'] = [round(x, 2) for x in effn]
    row['mean_eff_components'] = round(float(np.mean(effn)), 2)
    row['most_concentrated_archetype'] = int(np.argmin(effn))
    row['profiles_top5'] = profiles[:5]
    results[f'h{h}'] = row
    print(f'h{h} ({form}): mean pairwise core cos {row["mean_pairwise_core_cos"]} '
          f'(min {row["min_pairwise_core_cos"]}) | mean effective components '
          f'{row["mean_eff_components"]}/12 | most concentrated archetype '
          f'{row["most_concentrated_archetype"]} effN {min(effn):.2f}', flush=True)
    json.dump(results, open(f'{QK}/qk_corpus_components.json', 'w'), indent=2)
    del sp_glob, comp_cores, flat
    torch.cuda.empty_cache()
print('CORPUS COMPONENTS DONE', flush=True)
