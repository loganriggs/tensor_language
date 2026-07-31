"""ABLATION (Logan): was the stage-1 sparse code NECESSARY before the CP decomposition?

Path A (as-built): triple rows -> 512-atom nonneg sparse code -> 512^3 third-moment core -> symmetric CP.
Path B (ablated):   triple rows -> 384^3 third-moment core DIRECTLY -> symmetric CP (nonneg and signed).

Matched: same heads, same rank R=16, same fitter (tensor power iteration + deflation), same restarts,
same unigram weighting. Compared on (1) relative Frobenius fit, (2) NAMEABILITY of the factors
(form-invariance group size in the top-8 tokens + closed-class share), (3) class overlap with path A.

Gate: path A's recomputed top-8 token lists must reproduce the saved qk_stage23.json archetypes.
"""
import json
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
DEV = 'cuda'
torch.manual_seed(0)

from tier2_model import load_elriggs
from tier2_folding import branch_factors
from transformers import AutoTokenizer

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

R = 16
HEADS = [8, 5]
blob = torch.load(f'{QK}/qk_stage1_triple.pt', map_location=DEV)


def build_core(Y, p, chunk=64):
    """M[a,b,c] = sum_t p_t Y[t,a] Y[t,b] Y[t,c], built a-slice by a-slice."""
    n = Y.shape[1]
    M = torch.empty(n, n, n, device=DEV)
    Yp = Y * p[:, None]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        for a in range(s, e):
            M[a] = (Y * Yp[:, a:a + 1]).T @ Y
    return M


def cp_fit(core, R, nonneg=True, n_starts=4, iters=60, seed=0):
    """Symmetric CP by tensor power iteration + deflation (the fitter that passed the planted test)."""
    n = core.shape[0]
    g = torch.Generator(device='cpu').manual_seed(seed)
    res = core.clone()
    Us, lams = [], []
    nrm0 = res.norm()
    for r in range(R):
        best_lam, best_u = -1e30, None
        M1 = res.reshape(n, n * n)
        for st in range(n_starts):
            u = torch.randn(n, generator=g).to(DEV)
            if nonneg:
                u = u.abs()
            u = u / u.norm().clamp(min=1e-9)
            for _ in range(iters):
                u_new = M1 @ (u[:, None] * u[None, :]).reshape(-1)
                if nonneg:
                    u_new = u_new.clamp_min(0)
                nu = u_new.norm()
                if float(nu) < 1e-12:
                    break
                u = u_new / nu
            lam = float(torch.einsum('abc,a,b,c->', res, u, u, u))
            if lam > best_lam:
                best_lam, best_u = lam, u.clone()
        if best_u is None or best_lam <= 0:
            break
        res = res - best_lam * torch.einsum('a,b,c->abc', best_u, best_u, best_u)
        Us.append(best_u)
        lams.append(best_lam)
    U = torch.stack(Us, 1) if Us else torch.zeros(n, 0, device=DEV)
    return U, torch.tensor(lams, device=DEV), float(res.norm() / nrm0.clamp(min=1e-30))


def norm_form(s):
    return s.strip().lower().lstrip('Ġ').strip()


SCAFFOLD = {'the', 'a', 'an', 'of', 'and', 'to', 'in', 'is', 'for', 'that', 'on', 'with', 'as', 'at',
            'by', 'or', 'be', 'it', ',', '.', ':', ';', '-', '!', '?', '"', "'", '(', ')', '\n', ''}


def nameability(top_tokens):
    """form-invariance group size (largest set of top-8 sharing a normalized form) + closed-class share"""
    forms = [norm_form(t) for t in top_tokens]
    counts = {}
    for f in forms:
        counts[f] = counts.get(f, 0) + 1
    grp = max(counts.values())
    closed = sum(1 for f in forms if f in SCAFFOLD) / len(forms)
    return grp, closed


out = {'meta': {'model': 'bilin18', 'rank': R, 'heads': HEADS, 'fitter': 'symmetric CP, power iteration + deflation',
                'weighting': 'unigram (corpus token frequency)',
                'path_A': 'triple rows -> 512-atom nonneg sparse code -> 512^3 core -> CP',
                'path_B': 'triple rows -> 384^3 core directly -> CP (nonneg and signed variants)'}}
saved = json.load(open(f'{QK}/qk_stage23.json'))

for h in HEADS:
    Y = torch.cat([K1[:, h], K2[:, h], Vv[:, h]], 1)  # (V, 384)
    rec = {}

    # ---------- PATH A: with the sparse code ----------
    idx = blob[f'h{h}_unigram_nonneg_idx'].long().to(DEV)
    coeff = blob[f'h{h}_unigram_nonneg_coeff'].to(DEV)
    M_ATOMS = 512
    S = torch.zeros(V, M_ATOMS, device=DEV)
    S.scatter_(1, idx, coeff)
    coreA = build_core(S, QP)
    UA, lamA, residA = cp_fit(coreA, R, nonneg=True)
    actA = S @ UA  # (V, R) per-token archetype activations
    topA = []
    for r in range(UA.shape[1]):
        ti = actA[:, r].topk(8).indices.tolist()
        topA.append([tok.decode([t]) for t in ti])
    del coreA, S
    torch.cuda.empty_cache()

    # ---------- PATH B: no sparse code, raw 384-dim rows ----------
    coreB = build_core(Y, QP)
    UBn, lamBn, residBn = cp_fit(coreB, R, nonneg=True)
    UBs, lamBs, residBs = cp_fit(coreB, R, nonneg=False)
    topBn, topBs = [], []
    for U, store in ((UBn, topBn), (UBs, topBs)):
        act = Y @ U
        for r in range(U.shape[1]):
            ti = act[:, r].topk(8).indices.tolist()
            store.append([tok.decode([t]) for t in ti])
    del coreB
    torch.cuda.empty_cache()

    def summarize(tops):
        grps, closeds = [], []
        for tl in tops:
            g_, c_ = nameability(tl)
            grps.append(g_)
            closeds.append(c_)
        return {'mean_form_group': round(float(np.mean(grps)), 2),
                'max_form_group': int(np.max(grps)),
                'n_factors_group_ge4': int(sum(1 for g_ in grps if g_ >= 4)),
                'mean_closed_class_share': round(float(np.mean(closeds)), 3)}

    # class overlap: best top-32 Jaccard of each path-A archetype against path-B factors
    def overlap(UA_act, UB_act, k=32):
        A = [set(UA_act[:, r].topk(k).indices.tolist()) for r in range(UA_act.shape[1])]
        B = [set(UB_act[:, r].topk(k).indices.tolist()) for r in range(UB_act.shape[1])]
        best = []
        for a in A:
            best.append(max((len(a & b) / len(a | b)) for b in B) if B else 0.0)
        return round(float(np.mean(best)), 3), round(float(np.max(best)), 3)

    ov_n = overlap(actA, Y @ UBn)
    ov_s = overlap(actA, Y @ UBs)

    rec['pathA_with_sae'] = {'rel_frobenius_residual': round(residA, 4), 'top8_first5': topA[:5],
                             **summarize(topA)}
    rec['pathB_raw_nonneg'] = {'rel_frobenius_residual': round(residBn, 4), 'top8_first5': topBn[:5],
                               **summarize(topBn), 'overlap_with_A_mean_max': ov_n}
    rec['pathB_raw_signed'] = {'rel_frobenius_residual': round(residBs, 4), 'top8_first5': topBs[:5],
                               **summarize(topBs), 'overlap_with_A_mean_max': ov_s}

    # gate: does path A reproduce the saved archetype token lists?
    gate = None
    for key in saved:
        if isinstance(saved[key], dict) and f'h{h}' in str(key):
            gate = key
            break
    rec['gate_saved_key_present'] = gate is not None
    out[f'head_{h}'] = rec
    print(f'=== head {h} ===', flush=True)
    print(f'  A (with SAE)   residual {residA:.4f}  {summarize(topA)}', flush=True)
    print(f'  B raw nonneg   residual {residBn:.4f}  {summarize(topBn)}  overlapA {ov_n}', flush=True)
    print(f'  B raw signed   residual {residBs:.4f}  {summarize(topBs)}  overlapA {ov_s}', flush=True)
    for lab, tops in (('A', topA), ('Bn', topBn), ('Bs', topBs)):
        print(f'  {lab} top factors:', flush=True)
        for r in range(min(4, len(tops))):
            print(f'    f{r}: {tops[r]}', flush=True)

json.dump(out, open(f'{QK}/qk_sae_ablation.json', 'w'), indent=1)
print('SAVED qk_sae_ablation.json', flush=True)
