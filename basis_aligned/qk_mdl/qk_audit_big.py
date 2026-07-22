"""BIG AUDIT (Logan 2026-07-22): re-measure key layer-0 arms on much larger held-out sets.

Datasets: PILE_BIG = 512 fresh Pile sequences (262,144 predictions; disjoint from the 16-seq
original audit ALL[4:20] and the 128-seq wide audit ALL[20:148]) and FINEWEB = 600 sequences
(307,200 predictions) from data_fineweb_tokens.npy — the model's training distribution.

Arms (all deterministic re-fits, seed 0, same recipes as qk_sae_robust.py):
exact fold (gate) | svd r16/32/64/128 | merge K=2048 per-head-branch | dict n=1024 k=8
(token-linear + token-OMP/LS) | two-stage merge2048 -> OMP dict n=512 k=8.

Also SAVES the seed-0 dictionary fits to qk_dict_l0_seed0.pt for the feature-inspection pass.
Writes qk_audit_big.json.
"""
import json
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, build_eval_tokens, reference_forward
from tier2_folding import branch_factors, scores_from_factors
from qk_sae_lib import train_dict, encode_token, encode_omp, kmeans, arm_svd, fvu

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_audit_big.json'
DICT_PT = f'{QK}/qk_dict_l0_seed0.pt'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
NHB, ROW = NH * 2, 2 * HD

ALL = build_eval_tokens(n_chunks=660, seq_len=513)
PILE_BIG = ALL[148:660]
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
print(f'pile_big {tuple(PILE_BIG.shape)}  fineweb {tuple(FINEWEB.shape)}', flush=True)

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
HB = [(h, qn, kn) for h in range(NH) for (qn, kn) in BRANCHES]


def rows(h, qn, kn):
    return torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


def tables_from(recs, renorm=True):
    out = {n: TAB[n].clone() for n in NAMES}
    for (h, qn, kn), rec in zip(HB, recs):
        out[qn][:, h] = rec[:, :HD]
        out[kn][:, h] = rec[:, HD:]
    return {n: unit_rms(out[n]) for n in NAMES} if renorm else out


@torch.no_grad()
def audit_ce(tabs, tokens, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(tokens), batch):
        b = tokens[i:i + batch].to(DEV)
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


import os
prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
res = {'n_pile_preds': int(PILE_BIG.shape[0] * 512), 'n_fw_preds': int(FINEWEB.shape[0] * 512),
       'arms': dict(prev.get('arms', {}))}
if 'baseline_ce_pile' in prev:
    CE0_P, CE0_F = prev['baseline_ce_pile'], prev['baseline_ce_fw']
else:
    CE0_P = audit_ce(None, PILE_BIG)
    CE0_F = audit_ce(None, FINEWEB)
res['baseline_ce_pile'] = round(CE0_P, 4)
res['baseline_ce_fw'] = round(CE0_F, 4)
print(f'baseline CE pile_big {CE0_P:.4f} | fineweb {CE0_F:.4f}', flush=True)


def report(name, recs, mean_fvu=None):
    if name in res['arms']:
        print(f'{name:46s} (cached)', flush=True)
        recs.clear(); torch.cuda.empty_cache(); return
    tabs = tables_from(recs)
    recs.clear(); torch.cuda.empty_cache()
    dp = audit_ce(tabs, PILE_BIG) - CE0_P
    df = audit_ce(tabs, FINEWEB) - CE0_F
    row = {'dce_pile': round(dp, 4), 'dce_fw': round(df, 4)}
    if mean_fvu is not None:
        row['fvu'] = round(mean_fvu, 5)
    res['arms'][name] = row
    print(f'{name:46s} dCE pile {dp:+.4f} | fineweb {df:+.4f}', flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)


report('exact fold', [rows(*hb) for hb in HB])

for r in (16, 32, 64, 128):
    recs = [arm_svd(rows(*hb), r) for hb in HB]
    report(f'svd rank {r}', recs, sum(fvu(x, rows(*hb)) for x, hb in zip(recs, HB)) / NHB)

recs = []
for bi, hb in enumerate(HB):
    X = rows(*hb)
    assign, C = kmeans(X, 2048, seed=bi)
    recs.append(C[assign])
report('merge K=2048 per-head-branch', recs)

# dictionary seed 0 — fit, SAVE, audit both encoders
if os.path.exists(DICT_PT):
    blob = torch.load(DICT_PT, map_location=DEV)
    fits = [(blob[f'Dn{i}'], blob[f'b{i}'], blob[f'We{i}']) for i in range(NHB)]
    print('loaded cached dictionary fits', flush=True)
else:
    fits = []
    for bi, hb in enumerate(HB):
        fits.append(train_dict(rows(*hb), 1024, 8, seed=0))
        print(f'  dict fit {bi + 1}/{NHB}', flush=True)
    torch.save({**{f'Dn{i}': f[0].cpu() for i, f in enumerate(fits)},
                **{f'b{i}': f[1].cpu() for i, f in enumerate(fits)},
                **{f'We{i}': f[2].cpu() for i, f in enumerate(fits)}}, DICT_PT)
    print(f'saved {DICT_PT}', flush=True)
fits = [(a.to(DEV), b.to(DEV), c.to(DEV)) for (a, b, c) in fits]

recs = [encode_token(rows(*hb), *f, 8) for f, hb in zip(fits, HB)]
report('dict n=1024 k=8 token-linear', recs,
       sum(fvu(encode_token(rows(*hb), *f, 8), rows(*hb)) for f, hb in zip(fits, HB)) / NHB)
recs = [encode_omp(rows(*hb), f[0], f[1], 8) for f, hb in zip(fits, HB)]
report('dict n=1024 k=8 token-OMP/LS', recs,
       sum(fvu(encode_omp(rows(*hb), f[0], f[1], 8), rows(*hb)) for f, hb in zip(fits, HB)) / NHB)

recs = []
for bi, hb in enumerate(HB):
    X = rows(*hb)
    assign, C = kmeans(X, 2048, seed=bi)
    Dn, b, We = train_dict(C, 512, 8, seed=0, steps=2000)
    recs.append(encode_omp(C, Dn, b, 8)[assign])
report('two-stage merge2048 -> OMP dict n=512 k=8', recs)

json.dump(res, open(OUT, 'w'), indent=2)
print(f'\nwrote {OUT}', flush=True)
