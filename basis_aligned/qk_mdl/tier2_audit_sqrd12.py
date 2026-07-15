"""Same per-head + joint dCE audit for sqrd12 (162M, single QK branch,
pattern = (q.k/D)^2 row-NORMALIZED -> per-query scale gauge exists here).
T=512, same conventions as bilin18."""

import json
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, build_eval_tokens
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/tier2_audit_sqrd12.json'

m, cfg = load_elriggs('sqrd12')
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
V = cfg['vocab_size']
TOK = build_eval_tokens(n_chunks=20, seq_len=513)
FIT = TOK[:4].to(DEV)
AUDIT = TOK[4:]
qh_all, kh_all = branch_factors(m, 1, dtype=torch.float32)
S_ORIG = scores_from_factors(qh_all, kh_all, FIT, HD)


@torch.no_grad()
def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 8192):
            x = X[i:i + 8192]
            assign[i:i + 8192] = ((x ** 2).sum(1, keepdim=True) - 2 * x @ C.T
                                  + (C ** 2).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C)
        cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
        C = torch.where((cnt == 0)[:, None], C, Cn / cnt.clamp(min=1)[:, None])
    return C, assign


@torch.no_grad()
def ce(patch_fn=None, batch=8):
    tot, n = 0.0, 0
    st = {'tokens': None}

    def wrap(li, s, s2):
        return patch_fn(li, s, st['tokens']), s2
    for i in range(0, len(AUDIT), batch):
        b = AUDIT[i:i + batch].to(DEV)
        st['tokens'] = b[:, :-1]
        logits = reference_forward(m, b[:, :-1],
                                   score_patch=wrap if patch_fn else None).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


CE0 = ce()
print(f'baseline {CE0:.4f}')
FULL_DL = 32 * 2 * V * HD
results = {'model': 'sqrd12', 'baseline_ce': CE0, 'T': 512,
           'full_dl_bits': FULL_DL, 'rows': [], 'joint': {}}

VQ = {}
for hh in range(NH):
    qh = qh_all[:, hh].contiguous()
    kh = kh_all[:, hh].contiguous()
    row = {'head': hh, 'cands': []}
    cands = []
    for r in [1, 4, 16, 64]:
        def trunc(Xm, r=r):
            U, S, Vt = torch.linalg.svd(Xm, full_matrices=False)
            return U[:, :r] @ torch.diag(S[:r]) @ Vt[:r]
        cands.append((f'svd{r}', 2 * 32 * r * (V + HD + 1), trunc(qh), trunc(kh)))
    for k in [16, 256, 4096]:
        C, assign = kmeans(torch.cat([qh, kh], 1), k)
        cq, ck = C[assign][:, :HD].contiguous(), C[assign][:, HD:].contiguous()
        VQ[(k, hh)] = (cq, ck)
        cands.append((f'vq{k}', 32 * k * 2 * HD + V * math.log2(k), cq, ck))
    cands.append(('zero', 0, torch.zeros_like(qh), torch.zeros_like(kh)))
    for name, dl, qc, kc in cands:
        s_hat = scores_from_factors(qc[:, None], kc[:, None], FIT, HD)[:, 0]
        mse = float(((s_hat - S_ORIG[:, hh]) ** 2).sum() / (S_ORIG[:, hh] ** 2).sum())

        def patch(li, s, tokens, hh=hh, qc=qc, kc=kc):
            if li != 0:
                return s
            s = s.clone()
            s[:, hh] = scores_from_factors(qc[:, None], kc[:, None], tokens, HD)[:, 0]
            return s
        dce = ce(patch) - CE0
        row['cands'].append({'name': name, 'dl_bits': dl, 'pattern_mse': mse,
                             'dce': dce})
    results['rows'].append(row)
    print(f"L0H{hh}: " + '  '.join(f"{c['name']} {c['dce']:+.4f}"
                                   for c in row['cands']), flush=True)
    with open(OUT, 'w') as fh:
        json.dump(results, fh, indent=2)

for k in (16, 256):
    def joint(li, s, tokens, k=k):
        if li != 0:
            return s
        s = s.clone()
        for hh in range(NH):
            qc, kc = VQ[(k, hh)]
            s[:, hh] = scores_from_factors(qc[:, None], kc[:, None], tokens, HD)[:, 0]
        return s
    d = ce(joint) - CE0
    dl = NH * (32 * k * 2 * HD + V * math.log2(k))
    results['joint'][f'all vq{k}'] = {'dce': d, 'dl_bits': dl,
                                      'ratio': dl / (FULL_DL * NH)}
    print(f'joint all vq{k}: dCE {d:+.4f}  ratio {dl / (FULL_DL * NH):.2e}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(results, fh, indent=2)
print('sqrd12 audit done')
