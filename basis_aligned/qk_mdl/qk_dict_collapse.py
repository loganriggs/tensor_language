"""Composed arm (tick 158): dictionary for the 7 content-using heads + position-only collapse for
the content-free heads 2 and 5 (tick 156). Bits = 14/18 of the dictionary bits + 4*256 floats for
the collapsed mean rows. FineWeb audit. Writes qk_dict_collapse.json."""
import json
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from tier2_folding import branch_factors, scores_from_factors
from mdl_accounting import dl_sparse_dict
from qk_sae_lib import encode_token

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
COLLAPSE = (2, 5)

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
NHB, ROW = NH * 2, 2 * HD
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
HB = [(h, qn, kn) for h in range(NH) for (qn, kn) in BRANCHES]
blob = torch.load(f'{QK}/qk_dict_l0_seed0.pt', map_location=DEV)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


out = {n: TAB[n].clone() for n in NAMES}
for bi, (h, qn, kn) in enumerate(HB):
    if h in COLLAPSE:
        for n in (qn, kn):
            out[n][:, h] = TAB[n][:, h].mean(0, keepdim=True)
    else:
        X = torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)
        rec = encode_token(X, blob[f'Dn{bi}'], blob[f'b{bi}'], blob[f'We{bi}'], 8)
        out[qn][:, h] = rec[:, :HD]
        out[kn][:, h] = rec[:, HD:]
TABS = {n: unit_rms(out[n]) for n in NAMES}

n_dict_branches = NHB - 2 * len(COLLAPSE)
bits = n_dict_branches * dl_sparse_dict(1024, ROW, V * 8) + 2 * len(COLLAPSE) * 32 * ROW


@torch.no_grad()
def audit_fw(tabs, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16', score_patch=None if tabs is None else patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


CE0 = audit_fw(None)
d = audit_fw(TABS) - CE0
res = {'baseline_ce_fw': round(CE0, 4), 'dce_fw': round(d, 4), 'Mbits': round(bits / 1e6, 1),
       'pct_raw': round(100 * bits / (32 * NHB * V * ROW), 2), 'collapsed_heads': list(COLLAPSE)}
json.dump(res, open(f'{QK}/qk_dict_collapse.json', 'w'), indent=2)
print(f'dict(14 branches) + collapse heads {COLLAPSE}: dCE fw {d:+.4f}  '
      f'{bits/1e6:.1f} Mbit ({res["pct_raw"]}% raw)', flush=True)
