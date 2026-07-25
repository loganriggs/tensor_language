"""TICK 219b: the layer-0 double dissociation (contrast to the layer-1 negative).
Same four-condition protocol on l0 head 3's determiner channel (its top archetypes
{a/an} and {The} from the minimal inventory): T0 exact, T1 channel removed, T2 channel
only (h3 keys rank-reduced to the det detectors), T3 head zeroed. Split metric: dCE on
positions whose current token is in the archetypes' top-token set vs others. At layer
0, head 3 is near-load-bearing, so the prediction is a TRUE dissociation.

Original tick-199 header:
"""
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


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
P3 = mh_pt['h3']
Dn3 = P3['Dm'].to(DEV)
Dn3 = Dn3 / Dn3.norm(dim=1, keepdim=True).clamp_min(1e-8)
U3 = P3['U'].to(DEV)
DET_RS = [0, 2]
G1 = torch.linalg.qr(torch.stack([Dn3[:, :HD].T @ U3[:, r] for r in DET_RS], 1)).Q
G2 = torch.linalg.qr(torch.stack([Dn3[:, HD:2 * HD].T @ U3[:, r] for r in DET_RS], 1)).Q
kc3 = 2
z3 = torch.relu((torch.cat([TAB['k1'][:, H], TAB['k2'][:, H],
                            torch.zeros(V, HD, device=DEV)], 1) - 0) @ torch.zeros(1, 1, device=DEV).expand(384, Dn3.shape[0]).T) if False else None
S3 = torch.zeros(V, Dn3.shape[0], device=DEV)
mh_js = __import__('json').load(open(f'{QK}/qk_minimal_heads.json'))
DET_TOKENS = set()
for a in [mh_js['h3']['arch'][r] for r in DET_RS]:
    for trow in a['tok'][:8]:
        pass
# token ids via encoding: use loadings from saved codes instead — rebuild codes quickly
import torch.nn.functional as F2
Yh3 = torch.cat([TAB['k1'][:, H], TAB['k2'][:, H]], 1)
# DET tokens from archetype dumps (strings) -> ids via tokenizer round trip
det_strs = set()
for r in DET_RS:
    for trow in mh_js['h3']['arch'][r]['tok'][:8]:
        det_strs.add(trow[0])
DET_MASK_V = torch.zeros(V, dtype=torch.bool)
n_found = 0
for t in range(V):
    if tok.decode([t]).replace('\n', '\\n') in det_strs:
        DET_MASK_V[t] = True
        n_found += 1
print(f'DET token variants found: {n_found}; classes: {sorted(det_strs)}', flush=True)


def tabs_variant(kind):
    tabs = {k: v.clone() for k, v in TAB.items()}
    if kind == 'T0':
        return tabs
    for name, Gd in (('k1', G1), ('k2', G2)):
        col = tabs[name][:, H]
        proj = (col @ Gd) @ Gd.T
        if kind == 'T1':
            tabs[name][:, H] = col - proj
        elif kind == 'T2':
            tabs[name][:, H] = proj
        elif kind == 'T3':
            tabs[name][:, H] = 0
    return tabs


base = per_pos_loss(None)
CUR = SEQS[:, :-1].reshape(-1)
DETPOS = DET_MASK_V[CUR]
out = {'n_det_positions': int(DETPOS.sum())}
print(f'DET positions: {int(DETPOS.sum())} of {len(CUR)}', flush=True)
for kind in ('T1', 'T2', 'T3'):
    la = per_pos_loss(tabs_variant(kind))
    d = (la - base).flatten()
    row = {'overall': round(float(d.mean()), 5),
           'det_pos': round(float(d[DETPOS].mean()), 5),
           'other_pos': round(float(d[~DETPOS].mean()), 5)}
    out[kind] = row
    print(f'{kind}: overall {row["overall"]:+.5f} | det {row["det_pos"]:+.5f} | '
          f'other {row["other_pos"]:+.5f}', flush=True)
    __import__('json').dump(out, open(f'{QK}/qk_dissociation_l0.json', 'w'), indent=2)
print('L0 DISSOCIATION DONE', flush=True)
