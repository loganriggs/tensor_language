"""Tier-2 MDL table for bilin18 layer-0 heads (9 heads x 2 branches).

Folded object per (head, branch): factor pair (qhat, khat) in (V x 128)^2 —
exact (tier2_fold_gate PASS). Codebooks act on factors (never V x V):

  svd-r    rank-r truncation of qhat and khat.  DL = 2 * r(V + 128 + 1) floats
  vq-k     k-means over tokens on [qhat|khat]; token -> centroid factors.
           DL = k*256 floats + V*log2(k) bits   (k=1 == pure positional head)
  band-m   keep m of 64 RoPE planes by mass.    DL = 2*V*2m floats + 64 bits

Search loop metric: relative pattern MSE of branch scores on FIT chunks
(frozen convention). Binding audit: dCE with the compressed (head, branch)
patched into the full model, AUDIT chunks, T=512 (the model's competent
regime — CE explodes beyond ~pos 512; see LOG). Eval tables use bf16 rotary
tables (deployed semantics).
"""

import json
import math
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, build_eval_tokens
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/tier2_mdl_bilin18.json'

m, cfg = load_elriggs('bilin18')
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
V = cfg['vocab_size']

TOK = build_eval_tokens(n_chunks=20, seq_len=513)
FIT = TOK[:4].to(DEV)
AUDIT = TOK[4:]

import torch.nn.functional as F


@torch.no_grad()
def ce_with_patch(patch=None, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), batch):
        b = AUDIT[i:i + batch].to(DEV)
        if patch is not None:
            patch['tokens'] = b[:, :-1]
        logits = reference_forward(m, b[:, :-1], score_patch=patch['fn'] if patch else None).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


print('baseline CE (T=512)...')
CE0 = ce_with_patch()
print(f'baseline {CE0:.4f}')

FACT = {br: branch_factors(m, br, dtype=torch.float32) for br in (1, 2)}
S_ORIG = {br: scores_from_factors(FACT[br][0], FACT[br][1], FIT, HD)
          for br in (1, 2)}


def rel_mse(s_hat, s_ref):
    return float(((s_hat - s_ref) ** 2).sum() / (s_ref ** 2).sum())


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
    """Yield (name, DL_bits, qh_c, kh_c) compressed factor pairs for one head."""
    qh = FACT[br][0][:, hh].contiguous()      # (V, 128)
    kh = FACT[br][1][:, hh].contiguous()
    d = HD // 2
    out = []
    # svd-r
    for r in [1, 2, 4, 8, 16, 32, 64, 128]:
        def trunc(Xm, r=r):
            U, S, Vt = torch.linalg.svd(Xm, full_matrices=False)
            return U[:, :r] @ torch.diag(S[:r]) @ Vt[:r]
        out.append((f'svd{r}', 2 * 32 * r * (V + HD + 1), trunc(qh), trunc(kh)))
    # vq-k
    for k in [1, 16, 256, 4096]:
        if k == 1:
            cq, ck = qh.mean(0, keepdim=True).expand_as(qh), kh.mean(0, keepdim=True).expand_as(kh)
            dl = 32 * 2 * HD
        else:
            C, assign = kmeans(torch.cat([qh, kh], 1), k)
            cq, ck = C[assign][:, :HD], C[assign][:, HD:]
            dl = 32 * k * 2 * HD + V * math.log2(k)
        out.append((f'vq{k}', dl, cq.contiguous(), ck.contiguous()))
    # band-m (top planes by mass product)
    qa, qb, ka, kb = qh[:, :d], qh[:, d:], kh[:, :d], kh[:, d:]
    mass = ((qa ** 2).sum(0) + (qb ** 2).sum(0)) * ((ka ** 2).sum(0) + (kb ** 2).sum(0))
    order = mass.argsort(descending=True)
    for mm in [1, 2, 4, 8, 16, 32]:
        keep = torch.zeros(d, dtype=torch.bool, device=DEV)
        keep[order[:mm]] = True
        mask = torch.cat([keep, keep]).float()
        out.append((f'band{mm}', 32 * 2 * V * 2 * mm + 64, qh * mask, kh * mask))
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


FULL_DL = 32 * 2 * V * HD   # per (head, branch) raw factor storage
results = {'model': 'bilin18', 'baseline_ce': CE0, 'T': 512, 'full_dl_bits': FULL_DL,
           'rows': []}
for hh in range(NH):
    for br in (1, 2):
        s_ref = S_ORIG[br][:, hh]
        cands = candidates(hh, br)
        scored = []
        for name, dl, qc, kc in cands:
            s_hat = scores_from_factors(qc[:, None], kc[:, None], FIT, HD)[:, 0]
            scored.append((name, dl, rel_mse(s_hat, s_ref), qc, kc))
        # operating points: cheapest candidate under each pattern-MSE level
        chosen = {}
        for lvl in [1e-3, 1e-2, 5e-2]:
            ok = [c for c in scored if c[2] <= lvl]
            if ok:
                best = min(ok, key=lambda c: c[1])
                chosen[best[0]] = best
        audits = {}
        for name, dl, mse, qc, kc in chosen.values():
            patch = make_patch(hh, br, qc, kc)
            dce = ce_with_patch(patch) - CE0
            audits[name] = {'dl_bits': dl, 'pattern_mse': mse, 'dce': dce,
                            'dl_ratio_vs_full': dl / FULL_DL}
        row = {'head': hh, 'branch': br,
               'sweep': [{'name': n, 'dl_bits': d, 'pattern_mse': ms}
                         for n, d, ms, *_ in scored],
               'audited': audits}
        results['rows'].append(row)
        aud = '  '.join(f"{k}: dl {v['dl_bits'] / 1e6:.2f}Mb mse {v['pattern_mse']:.4f} "
                        f"dCE {v['dce']:+.4f}" for k, v in audits.items())
        print(f'L0H{hh} b{br}: {aud}', flush=True)
        with open(OUT, 'w') as fh:
            json.dump(results, fh, indent=2)
print('tier2_mdl done')
