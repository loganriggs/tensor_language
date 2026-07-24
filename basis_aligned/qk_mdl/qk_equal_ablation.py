"""TICK 199 (Logan): are the layer-0 head-3 archetype directions PRIVILEGED, or would
any equal-sized ablation do the same? Hypothesis: the archetypes align with the sparse
interaction structure (with itself, the embedding, and the bilinear read-out), so
removing them should (i) cost more per unit of pattern actually removed and (ii)
concentrate its damage on few positions, versus energy-matched generic controls.

Arms (all on layer-0 head 3, displayed minimal inventory m=512 k=4):
  arch1   : top-1 archetype channel projected out of both key tables
  arch10  : top-10 archetype span projected out
  pca10   : top-10 PCA directions of the p-weighted key tables projected out
            (biggest variance directions, NOT interaction-fitted)
  rand10  : random 10-dim subspace projected out
  shrink1 : uniform score shrink beta on both branches, calibrated so the removed
            pattern energy E_p x p[(dP)^2] matches arch1
  shrink10: same, matched to arch10
Matching metric: pattern energy removed, E_{i,t ~ p x p}[(P_ablated - P)^2], sampled
over 8 x 4096^2 token pairs. Evaluation: per-position CE on 128 held-out documents
(65k predictions): mean dCE, dCE per unit removed energy, and concentration (share of
total positive damage carried by the top 0.1% / 1% of positions; fraction of positions
with |dCE| > 0.01).
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
H, N_SEQ = 3, 128

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
TAB = {}
for br, (qn, kn) in ((1, ('q1', 'k1')), (2, ('q2', 'k2'))):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQS = FINEWEB[:N_SEQ]
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()

mh_pt = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
P3 = mh_pt[f'h{H}']
Dn3 = P3['Dm'].to(DEV)
Dn3 = Dn3 / Dn3.norm(dim=1, keepdim=True).clamp_min(1e-8)
U3 = P3['U'].to(DEV)


def arch_dirs(rs):
    g1s, g2s = [], []
    for r in rs:
        g1 = Dn3[:, :HD].T @ U3[:, r]
        g2 = Dn3[:, HD:2 * HD].T @ U3[:, r]
        g1s.append(g1 / g1.norm().clamp_min(1e-12))
        g2s.append(g2 / g2.norm().clamp_min(1e-12))
    return (torch.linalg.qr(torch.stack(g1s, 1)).Q, torch.linalg.qr(torch.stack(g2s, 1)).Q)


def project_tabs(Q1, Q2):
    tabs = {k: v.clone() for k, v in TAB.items()}
    tabs['k1'][:, H] -= (tabs['k1'][:, H] @ Q1) @ Q1.T
    tabs['k2'][:, H] -= (tabs['k2'][:, H] @ Q2) @ Q2.T
    return tabs


def shrink_tabs(beta):
    tabs = {k: v.clone() for k, v in TAB.items()}
    tabs['k1'][:, H] *= beta
    tabs['k2'][:, H] *= beta
    return tabs


@torch.no_grad()
def pattern_energy(tabs, n_batch=8, n=4096, seed=0):
    """E_{i,t~pxp}[(P_abl - P)^2] with static (non-rotary) scores — the energy meter."""
    g = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(n_batch):
        si = torch.multinomial(QP.cpu(), n, replacement=True, generator=g).to(DEV)
        ti = torch.multinomial(QP.cpu(), n, replacement=True, generator=g).to(DEV)
        s1 = TAB['q1'][si, H] @ TAB['k1'][ti, H].T / HD
        s2 = TAB['q2'][si, H] @ TAB['k2'][ti, H].T / HD
        a1 = TAB['q1'][si, H] @ tabs['k1'][ti, H].T / HD
        a2 = TAB['q2'][si, H] @ tabs['k2'][ti, H].T / HD
        tot += float(((s1 * s2 - a1 * a2) ** 2).mean())
    return tot / n_batch


@torch.no_grad()
def per_pos_loss(tabs, batch=4):
    outs = []
    for i in range(0, N_SEQ, batch):
        b = SEQS[i:i + batch].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16',
                                   score_patch=None if tabs is None else patch).float()
        ls = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1), reduction='none')
        outs.append(ls.view(b.shape[0], -1).cpu())
    return torch.cat(outs, 0)


base = per_pos_loss(None)
print(f'baseline mean CE {float(base.mean()):.5f}', flush=True)

# build arms
Q1a, Q2a = arch_dirs([0])
Q1b, Q2b = arch_dirs(list(range(10)))
Wp = QP.sqrt()[:, None]
pca1 = torch.linalg.svd(Wp * TAB['k1'][:, H], full_matrices=False).Vh[:10].T
pca2 = torch.linalg.svd(Wp * TAB['k2'][:, H], full_matrices=False).Vh[:10].T
gr = torch.Generator().manual_seed(0)
R1 = torch.linalg.qr(torch.randn(HD, 10, generator=gr).to(DEV)).Q
R2 = torch.linalg.qr(torch.randn(HD, 10, generator=gr).to(DEV)).Q

arms = {'arch1': project_tabs(Q1a, Q2a), 'arch10': project_tabs(Q1b, Q2b),
        'pca10': project_tabs(pca1, pca2), 'rand10': project_tabs(R1, R2)}
E = {name: pattern_energy(t) for name, t in arms.items()}
EP2 = pattern_energy(shrink_tabs(0.0))                        # = E[P^2]
for name, target in (('shrink1', E['arch1']), ('shrink10', E['arch10'])):
    # dP = (1 - beta^2) P  =>  energy = (1-beta^2)^2 E[P^2]
    frac = min(1.0, (target / max(EP2, 1e-30)) ** 0.5)
    beta = (1 - frac) ** 0.5
    arms[name] = shrink_tabs(beta)
    E[name] = pattern_energy(arms[name])
    print(f'{name}: beta {beta:.4f} energy {E[name]:.3e} (target {target:.3e})', flush=True)

out = {'base_ce': round(float(base.mean()), 5), 'E_P2': EP2}
for name, tabs in arms.items():
    la = per_pos_loss(tabs)
    delta = (la - base).flatten()
    mean_dce = float(delta.mean())
    pos = delta.clamp_min(0)
    tot = float(pos.sum()).__abs__() or 1e-9
    srt = pos.sort(descending=True).values
    n = len(srt)
    top01 = float(srt[:max(1, n // 1000)].sum()) / tot
    top1 = float(srt[:max(1, n // 100)].sum()) / tot
    frac_hit = float((delta.abs() > 0.01).float().mean())
    row = {'energy_removed': round(E[name], 6), 'mean_dce': round(mean_dce, 5),
           'dce_per_energy': round(mean_dce / max(E[name], 1e-12), 2),
           'top0.1pct_share': round(top01, 3), 'top1pct_share': round(top1, 3),
           'frac_positions_hit': round(frac_hit, 4)}
    out[name] = row
    print(f'{name}: E {E[name]:.3e} | mean dCE {mean_dce:+.5f} | per-energy '
          f'{row["dce_per_energy"]} | top0.1% {top01:.2f} top1% {top1:.2f} | '
          f'hit-frac {frac_hit:.4f}', flush=True)
    json.dump(out, open(f'{QK}/qk_equal_ablation.json', 'w'), indent=2)
    del la
    torch.cuda.empty_cache()
print('EQUAL ABLATION DONE', flush=True)
