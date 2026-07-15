"""Tier-2 full-grid dCE audit for bilin18 layer-0: every candidate, every
(head, branch) — the DL-vs-dCE curves the MDL table is read from. The
pattern-MSE screen proved useless here (dCE ~ 0 at pattern-MSE 0.04), so the
binding metric (dCE, per Logan) is audited on the full grid directly.
Self-contained (tier2_mdl.py executes at import)."""

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
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/tier2_audit_bilin18.json'

m, cfg = load_elriggs('bilin18')
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
V = cfg['vocab_size']
TOK = build_eval_tokens(n_chunks=20, seq_len=513)
FIT = TOK[:4].to(DEV)
AUDIT = TOK[4:]
FACT = {br: branch_factors(m, br, dtype=torch.float32) for br in (1, 2)}
S_ORIG = {br: scores_from_factors(FACT[br][0], FACT[br][1], FIT, HD) for br in (1, 2)}


@torch.no_grad()
def ce_with_patch(patch=None, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), batch):
        b = AUDIT[i:i + batch].to(DEV)
        if patch is not None:
            patch['tokens'] = b[:, :-1]
        logits = reference_forward(
            m, b[:, :-1], score_patch=patch['fn'] if patch else None).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


@torch.no_grad()
def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 8192):
            x = X[i:i + 8192]
            d2 = (x ** 2).sum(1, keepdim=True) - 2 * x @ C.T + (C ** 2).sum(1)[None]
            assign[i:i + 8192] = d2.argmin(1)
        Cn = torch.zeros_like(C)
        cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
        C = torch.where((cnt == 0)[:, None], C, Cn / cnt.clamp(min=1)[:, None])
    return C, assign


def candidates(hh, br):
    qh = FACT[br][0][:, hh].contiguous()
    kh = FACT[br][1][:, hh].contiguous()
    d = HD // 2
    out = []
    for r in [1, 2, 4, 8, 16, 32, 64]:
        def trunc(Xm, r=r):
            U, S, Vt = torch.linalg.svd(Xm, full_matrices=False)
            return U[:, :r] @ torch.diag(S[:r]) @ Vt[:r]
        out.append((f'svd{r}', 2 * 32 * r * (V + HD + 1), trunc(qh), trunc(kh)))
    for k in [1, 16, 256, 4096]:
        if k == 1:
            cq = qh.mean(0, keepdim=True).expand_as(qh)
            ck = kh.mean(0, keepdim=True).expand_as(kh)
            dl = 32 * 2 * HD
        else:
            C, assign = kmeans(torch.cat([qh, kh], 1), k)
            cq, ck = C[assign][:, :HD].contiguous(), C[assign][:, HD:].contiguous()
            dl = 32 * k * 2 * HD + V * math.log2(k)
        out.append((f'vq{k}', dl, cq, ck))
    qa, qb, ka, kb = qh[:, :d], qh[:, d:], kh[:, :d], kh[:, d:]
    mass = ((qa ** 2).sum(0) + (qb ** 2).sum(0)) * ((ka ** 2).sum(0) + (kb ** 2).sum(0))
    order = mass.argsort(descending=True)
    for mm in [4, 16, 32]:
        keep = torch.zeros(d, dtype=torch.bool, device=DEV)
        keep[order[:mm]] = True
        mask = torch.cat([keep, keep]).float()
        out.append((f'band{mm}', 32 * 2 * V * 2 * mm + 64, qh * mask, kh * mask))
    out.append(('zero', 0, torch.zeros_like(qh), torch.zeros_like(kh)))  # ablation ref
    return out


def make_patch(hh, br, qc, kc):
    state = {'fn': None, 'tokens': None}

    def fn(li, s1, s2):
        if li != 0:
            return s1, s2
        sc = scores_from_factors(qc[:, None], kc[:, None], state['tokens'], HD)
        if br == 1:
            s1 = s1.clone(); s1[:, hh] = sc[:, 0]
        else:
            s2 = s2.clone(); s2[:, hh] = sc[:, 0]
        return s1, s2
    state['fn'] = fn
    return state


FULL_DL = 32 * 2 * V * HD
print('baseline CE (T=512)...')
CE0 = ce_with_patch()
print(f'baseline {CE0:.4f}')
results = {'model': 'bilin18', 'baseline_ce': CE0, 'T': 512,
           'full_dl_bits': FULL_DL, 'rows': []}
for hh in range(NH):
    for br in (1, 2):
        s_ref = S_ORIG[br][:, hh]
        row = {'head': hh, 'branch': br, 'cands': []}
        for name, dl, qc, kc in candidates(hh, br):
            s_hat = scores_from_factors(qc[:, None], kc[:, None], FIT, HD)[:, 0]
            mse = float(((s_hat - s_ref) ** 2).sum() / (s_ref ** 2).sum())
            dce = ce_with_patch(make_patch(hh, br, qc, kc)) - CE0
            row['cands'].append({'name': name, 'dl_bits': dl,
                                 'pattern_mse': mse, 'dce': dce})
        results['rows'].append(row)
        cheap = [c for c in row['cands'] if c['dce'] <= 0.01]
        best = min(cheap, key=lambda c: c['dl_bits']) if cheap else None
        print(f"L0H{hh} b{br}: min-DL @ dCE<=0.01: "
              f"{best['name'] if best else 'NONE'} "
              f"({best['dl_bits'] / FULL_DL:.4f} of full, mse {best['pattern_mse']:.3f})"
              if best else f'L0H{hh} b{br}: NONE under 0.01', flush=True)
        with open(OUT, 'w') as fh:
            json.dump(results, fh, indent=2)
print('audit done')
