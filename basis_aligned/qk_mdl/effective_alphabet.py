"""Effective alphabet per (head, branch): the minimal k (powers of 2) such that
replacing this head-branch's factors with vq-k costs dCE <= 0.01 — the head's
SUFFICIENT PARTITION of the vocabulary at the behavioral tier. Also reports the
weight-side alphabet (minimal k with mean relative factor error <= 25%) for the
tier-1-flavored contrast. bilin18 layer-0, all 18 head-branches."""

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, build_eval_tokens
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/effective_alphabet.json'
EPS = 0.01

m, cfg = load_elriggs('bilin18')
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:]
FACT = {br: branch_factors(m, br, dtype=torch.float32) for br in (1, 2)}


@torch.no_grad()
def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 8192):
            xx = X[i:i + 8192]
            assign[i:i + 8192] = ((xx ** 2).sum(1, keepdim=True) - 2 * xx @ C.T
                                  + (C ** 2).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C)
        cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
        C = torch.where((cnt == 0)[:, None], C, Cn / cnt.clamp(min=1)[:, None])
    return C, assign


@torch.no_grad()
def ce(patch=None, batch=4):
    tot, n = 0.0, 0
    st = {'tokens': None}

    def wrap(li, s1, s2):
        return patch(li, s1, s2, st['tokens'])
    for i in range(0, len(AUDIT), batch):
        b = AUDIT[i:i + batch].to(DEV)
        st['tokens'] = b[:, :-1]
        logits = reference_forward(m, b[:, :-1],
                                   score_patch=wrap if patch else None).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


CE0 = ce()
print(f'baseline {CE0:.4f}')
results = {'baseline_ce': CE0, 'eps': EPS, 'alphabet': {}}

for hh in range(NH):
    for br in (1, 2):
        q, k_ = FACT[br][0][:, hh].contiguous(), FACT[br][1][:, hh].contiguous()
        X = torch.cat([q, k_], 1)
        norm = X.norm(dim=1).mean()
        entry = {}
        # weight-side alphabet: minimal k with mean rel factor error <= 25%
        kw = None
        for kk in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            C, assign = kmeans(X, kk)
            rel = float((X - C[assign]).norm(dim=1).mean() / norm)
            if rel <= 0.25:
                kw = kk
                break
        entry['weight_alphabet_rel25'] = kw
        # behavioral alphabet
        kb = None
        for kk in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            C, assign = kmeans(X, kk)
            qc = C[assign][:, :HD].contiguous()
            kc = C[assign][:, HD:].contiguous()

            def patch(li, s1, s2, tokens, hh=hh, br=br, qc=qc, kc=kc):
                if li != 0:
                    return s1, s2
                sc = scores_from_factors(qc[:, None], kc[:, None], tokens, HD)
                if br == 1:
                    s1 = s1.clone(); s1[:, hh] = sc[:, 0].to(s1.dtype)
                else:
                    s2 = s2.clone(); s2[:, hh] = sc[:, 0].to(s2.dtype)
                return s1, s2
            d = ce(patch) - CE0
            if d <= EPS:
                kb = kk
                entry['dce_at_kb'] = d
                break
        entry['behavioral_alphabet'] = kb
        results['alphabet'][f'L0H{hh}b{br}'] = entry
        print(f"L0H{hh}b{br}: behavioral alphabet {kb}  "
              f"(weight-side rel25: {kw})", flush=True)
        with open(OUT, 'w') as fh:
            json.dump(results, fh, indent=2)
print('effective alphabet done')
