"""FINEWEB FRONTIER FILL (tick 155): audit the remaining layer-0 arms on the training
distribution (FineWeb, 307,200 predictions) so the corrected frontier figure is complete.
Headline audits moved to FineWeb after tick 154 (Pile is off-distribution for this model and
shows a real coarsening-helps effect that confounds the frontier).

Arms added here (seed-0 recipes from qk_sae_lib): svd r8 | per-head-branch merge K=256, K=8192 |
GLOBAL merge K=2048 (stage-1 recipe: PCA-256 space, one partition, renorm) | dict n=1024 k=8
batch-topk + matryoshka | dict n=4096 k=8 token-OMP/LS. Writes qk_fw_fill.json.
"""
import json
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from tier2_folding import branch_factors, scores_from_factors
from mdl_accounting import dl_svd, dl_sparse_dict, dl_bits
from qk_sae_lib import (train_dict, encode_token, encode_batch, encode_omp,
                        kmeans, arm_svd, fvu)

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_fw_fill.json'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))

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


def rows(h, qn, kn):
    return torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


def tables_from(recs):
    out = {n: TAB[n].clone() for n in NAMES}
    for (h, qn, kn), rec in zip(HB, recs):
        out[qn][:, h] = rec[:, :HD]
        out[kn][:, h] = rec[:, HD:]
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


import os
prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
res = {'arms': dict(prev.get('arms', {}))}
CE0 = prev.get('baseline_ce_fw') or audit_fw(None)
res['baseline_ce_fw'] = round(CE0, 4)
print(f'baseline CE fineweb {CE0:.4f}', flush=True)


def report(name, recs, Mbits, mean_fvu=None):
    if name in res['arms']:
        print(f'{name:46s} (cached)', flush=True)
        recs.clear(); torch.cuda.empty_cache(); return
    tabs = tables_from(recs)
    recs.clear(); torch.cuda.empty_cache()
    d = audit_fw(tabs) - CE0
    row = {'dce_fw': round(d, 4), 'Mbits': round(Mbits, 1)}
    if mean_fvu is not None:
        row['fvu'] = round(mean_fvu, 5)
    res['arms'][name] = row
    print(f'{name:46s} dCE fw {d:+.4f}  {Mbits:8.1f} Mbit', flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)


# svd rank 8
recs = [arm_svd(rows(*hb), 8) for hb in HB]
report('svd rank 8', recs, NHB * dl_svd(8, V, ROW) / 1e6,
       sum(fvu(arm_svd(rows(*hb), 8), rows(*hb)) for hb in HB) / NHB)

# per-head-branch merges
for K in (256, 8192):
    recs, fv = [], []
    for bi, hb in enumerate(HB):
        X = rows(*hb)
        assign, C = kmeans(X, K, seed=bi)
        recs.append(C[assign])
        fv.append(fvu(C[assign], X))
    report(f'merge K={K} per-head-branch', recs,
           (32 * NHB * K * ROW + NHB * V * math.log2(K)) / 1e6, sum(fv) / len(fv))

# global merge K=2048 (stage-1 recipe: PCA-256 space, one shared partition)
Xall = torch.cat([rows(*hb) for hb in HB], 1)
mu = Xall.mean(0)
Xc = Xall - mu
cov = torch.zeros(Xc.shape[1], Xc.shape[1], dtype=torch.float64, device=DEV)
for i in range(0, V, 8192):
    xx = Xc[i:i + 8192].double()
    cov += xx.T @ xx
evals, evecs = torch.linalg.eigh(cov)
Xp = (Xc.double() @ evecs[:, -256:].flip(-1)).float()
assign, _ = kmeans(Xp, 2048, seed=0)
del Xall, Xc, cov, evals, evecs, Xp
torch.cuda.empty_cache()
recs, fv = [], []
for hb in HB:
    X = rows(*hb)
    C = torch.zeros(2048, ROW, device=DEV)
    c2 = torch.zeros(2048, device=DEV)
    C.index_add_(0, assign, X)
    c2.index_add_(0, assign, torch.ones(V, device=DEV))
    C[c2 > 0] /= c2[c2 > 0][:, None]
    recs.append(C[assign])
    fv.append(fvu(C[assign], X))
report('merge K=2048 GLOBAL', recs, (32 * NHB * 2048 * ROW + V * math.log2(2048)) / 1e6,
       sum(fv) / len(fv))

# dict n=1024 k=8: batch-topk and matryoshka (refit, seed 0)
bits_1024 = NHB * dl_sparse_dict(1024, ROW, V * 8) / 1e6
recs, fv, nnz_tot = [], [], 0
for bi, hb in enumerate(HB):
    X = rows(*hb)
    Dn, b, We = train_dict(X, 1024, 8, mode='batch', seed=0)
    xh, nnz = encode_batch(X, Dn, b, We, 8)
    recs.append(xh); fv.append(fvu(xh, X)); nnz_tot += nnz
    print(f'  batch fit {bi + 1}/{NHB}', flush=True)
report('dict n=1024 k=8 batch-topk', recs,
       (NHB * dl_bits(n_floats=1024 * ROW + ROW) + nnz_tot * (32 + math.log2(1024))) / 1e6,
       sum(fv) / len(fv))

recs, fv = [], []
for bi, hb in enumerate(HB):
    X = rows(*hb)
    Dn, b, We = train_dict(X, 1024, 8, seed=0, nested=[128, 512, 1024])
    xh = encode_token(X, Dn, b, We, 8)
    recs.append(xh); fv.append(fvu(xh, X))
    print(f'  matryoshka fit {bi + 1}/{NHB}', flush=True)
report('dict n=1024 k=8 matryoshka', recs, bits_1024, sum(fv) / len(fv))

# dict n=4096 k=8 OMP/LS (refit, seed 0)
recs, fv = [], []
for bi, hb in enumerate(HB):
    X = rows(*hb)
    Dn, b, We = train_dict(X, 4096, 8, seed=0)
    xh = encode_omp(X, Dn, b, 8)
    recs.append(xh); fv.append(fvu(xh, X))
    print(f'  n4096 fit {bi + 1}/{NHB}', flush=True)
report('dict n=4096 k=8 token-OMP/LS', recs, NHB * dl_sparse_dict(4096, ROW, V * 8) / 1e6,
       sum(fv) / len(fv))

json.dump(res, open(OUT, 'w'), indent=2)
print(f'\nwrote {OUT}', flush=True)
