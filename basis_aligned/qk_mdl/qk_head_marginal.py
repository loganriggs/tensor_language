"""PER-HEAD MARGINAL COLLAPSE on FineWeb (Logan Q, tick 156): is any head's query/key
content-free? For each head, replace BOTH branches' factor rows by their vocabulary mean
(K=1 merge: that head's pattern becomes position-only via rotary, token-independent) with the
other 8 heads exact, and audit held-out FineWeb delta-CE. Then the JOINT arm: collapse every head
that was individually cheap (< 0.003), because this program's standing lesson is that marginals
do not compose. Bits for a collapsed head ~ 0. Writes qk_head_marginal.json.
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
OUT = f'{QK}/qk_head_marginal.json'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


def collapse(heads):
    out = {n: TAB[n].clone() for n in NAMES}
    for h in heads:
        for n in NAMES:
            out[n][:, h] = out[n][:, h].mean(0, keepdim=True)
    return {n: unit_rms(out[n]) for n in NAMES}


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
res = {'baseline_ce_fw': round(CE0, 4), 'marginal': {}, 'joint': {}}
print(f'baseline CE fineweb {CE0:.4f}', flush=True)

cheap = []
for h in range(NH):
    d = audit_fw(collapse([h])) - CE0
    res['marginal'][h] = round(d, 4)
    if d < 0.003:
        cheap.append(h)
    print(f'collapse head {h} alone: dCE fw {d:+.4f}', flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)

res['cheap_heads'] = cheap
if len(cheap) >= 2:
    d = audit_fw(collapse(cheap)) - CE0
    res['joint'][f'collapse {cheap}'] = round(d, 4)
    print(f'JOINT collapse {cheap}: dCE fw {d:+.4f}', flush=True)
d = audit_fw(collapse(list(range(NH)))) - CE0
res['joint']['collapse all 9'] = round(d, 4)
print(f'JOINT collapse all 9: dCE fw {d:+.4f}', flush=True)
json.dump(res, open(OUT, 'w'), indent=2)
print(f'wrote {OUT}', flush=True)
