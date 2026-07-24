"""TICK 191 (follow-up to Logan's ablation direction): GROUP ablations — is the
near-zero single-channel damage on heads 1, 2, 5, 0, 4 true redundancy (group ablation
also small) or distributed-but-critical structure (group >> sum of singles)?

Per head, on the same 64 held-out documents as tick 190:
  singles_sum : sum of the ten individual mean dCEs (tick 190, recomputed here from json)
  group       : project the SPAN of all ten archetype key channels (QR-orthonormalized)
                out of both key tables at once
  random      : control — project out a random 10-dimensional subspace per branch
                (matched dimension; seed 0)
  whole_head  : zero the head's pattern entirely (k1[:,h] = k2[:,h] = 0)
Interpretation: group ~ singles_sum -> channels independent; group >> singles_sum ->
distributed structure with internal backup (any one channel removable, the ensemble
not); group ~ random -> the specific directions don't matter, only dimension does;
whole_head calibrates how much the head matters at all.
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
N_SEQ, TOP_R = 64, 10

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
TAB = {}
for br, (qn, kn) in ((1, ('q1', 'k1')), (2, ('q2', 'k2'))):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQS = FINEWEB[:N_SEQ]


@torch.no_grad()
def mean_ce(tabs, batch=4):
    tot, n = 0.0, 0
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
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


base = mean_ce(None)
print(f'baseline CE {base:.5f}', flush=True)

mh_pt = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
polish = {0: torch.load(f'{QK}/qk_h0_polish_g025.pt', map_location=DEV),
          4: torch.load(f'{QK}/qk_h04_polish.pt', map_location=DEV)}
singles = json.load(open(f'{QK}/qk_arch_ablation.json'))


def detector_mats(h):
    g1s, g2s = [], []
    for r in range(TOP_R):
        if h in (0, 4):
            bb = polish[h]
            D1 = bb[f'h{h}_k1_Dm'].to(DEV)
            D2 = bb[f'h{h}_k2_Dm'].to(DEV)
            D1 = D1 / D1.norm(dim=1, keepdim=True).clamp_min(1e-8)
            D2 = D2 / D2.norm(dim=1, keepdim=True).clamp_min(1e-8)
            g1 = D1.T @ bb[f'h{h}_AJ'][:, r].to(DEV)
            g2 = D2.T @ bb[f'h{h}_BJ'][:, r].to(DEV)
        else:
            P = mh_pt[f'h{h}']
            Dn = P['Dm'].to(DEV)
            Dn = Dn / Dn.norm(dim=1, keepdim=True).clamp_min(1e-8)
            U = P['U'].to(DEV)
            if r >= U.shape[1]:
                break
            g1 = Dn[:, :HD].T @ U[:, r]
            g2 = Dn[:, HD:2 * HD].T @ U[:, r]
        g1s.append(g1 / g1.norm().clamp_min(1e-12))
        g2s.append(g2 / g2.norm().clamp_min(1e-12))
    Q1 = torch.linalg.qr(torch.stack(g1s, 1)).Q
    Q2 = torch.linalg.qr(torch.stack(g2s, 1)).Q
    return Q1, Q2


def project_out(tabs, h, Q1, Q2):
    tabs['k1'][:, h] -= (tabs['k1'][:, h] @ Q1) @ Q1.T
    tabs['k2'][:, h] -= (tabs['k2'][:, h] @ Q2) @ Q2.T


out = {'base_ce': round(base, 5)}
gr = torch.Generator().manual_seed(0)
for h in [1, 2, 3, 5, 6, 7, 8, 0, 4]:
    Q1, Q2 = detector_mats(h)
    dim = Q1.shape[1]
    tabs = {k: v.clone() for k, v in TAB.items()}
    project_out(tabs, h, Q1, Q2)
    grp = mean_ce(tabs) - base
    del tabs
    R1 = torch.linalg.qr(torch.randn(HD, dim, generator=gr).to(DEV)).Q
    R2 = torch.linalg.qr(torch.randn(HD, dim, generator=gr).to(DEV)).Q
    tabs = {k: v.clone() for k, v in TAB.items()}
    project_out(tabs, h, R1, R2)
    rnd = mean_ce(tabs) - base
    del tabs
    tabs = {k: v.clone() for k, v in TAB.items()}
    tabs['k1'][:, h] = 0
    tabs['k2'][:, h] = 0
    whole = mean_ce(tabs) - base
    del tabs
    ss = sum(r['mean_dce'] for r in singles[f'h{h}'])
    out[f'h{h}'] = {'dim': dim, 'singles_sum': round(ss, 5), 'group': round(grp, 5),
                    'random_ctrl': round(rnd, 5), 'whole_head': round(whole, 5)}
    print(f'h{h}: singles-sum {ss:+.5f} | group({dim}d) {grp:+.5f} | random {rnd:+.5f} | '
          f'whole-head {whole:+.5f}', flush=True)
    json.dump(out, open(f'{QK}/qk_group_ablation.json', 'w'), indent=2)
    torch.cuda.empty_cache()
print('GROUP ABLATION DONE', flush=True)
