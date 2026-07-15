"""Fill in the missing JOINT arms on bilin18 layer-0 QK so every codebook
family is measured on the same object: joint svd-r, joint band-m, joint
positional (per-Δ mean), plus zero — completing the unified methods graph
(vq raw / CE-trained / KL points already exist)."""

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, build_eval_tokens
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/tier2_joint_families.json'

m, cfg = load_elriggs('bilin18')
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:]
FACT = {br: branch_factors(m, br, dtype=torch.float32) for br in (1, 2)}
FULL = 32 * 2 * 2 * V * HD * NH


@torch.no_grad()
def ce(patch_fn=None, batch=4):
    tot, n = 0.0, 0
    st = {'tokens': None}

    def wrap(li, s1, s2):
        return patch_fn(li, s1, s2, st['tokens'])
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
results = {'baseline_ce': CE0, 'arms': {}}


def factor_patch(make_qk):
    """Patch layer-0 scores from transformed factor tensors (all heads)."""
    tabs = {}
    for br in (1, 2):
        qh, kh = FACT[br]
        tabs[br] = (make_qk(qh), make_qk(kh))

    def fn(li, s1, s2, tokens):
        if li != 0:
            return s1, s2
        out = []
        for br, sx in [(1, s1), (2, s2)]:
            qc, kc = tabs[br]
            out.append(scores_from_factors(qc, kc, tokens, HD).to(sx.dtype))
        return out[0], out[1]
    return fn


# joint svd-r (per head, both factor matrices truncated)
for r in [4, 16, 64]:
    def tr(X, r=r):
        out = torch.empty_like(X)
        for hh in range(NH):
            U, S, Vt = torch.linalg.svd(X[:, hh], full_matrices=False)
            out[:, hh] = U[:, :r] @ torch.diag(S[:r]) @ Vt[:r]
        return out
    d = ce(factor_patch(tr)) - CE0
    dl = 2 * 2 * NH * 32 * r * (V + HD + 1)
    results['arms'][f'joint svd{r}'] = {'dce': d, 'ratio': dl / FULL}
    print(f'joint svd{r}: dCE {d:+.4f} ratio {dl / FULL:.3f}', flush=True)

# joint band-m
for mm in [8, 24, 48]:
    def bandmask(X, mm=mm):
        d = HD // 2
        out = torch.empty_like(X)
        for hh in range(NH):
            xa, xb = X[:, hh, :d], X[:, hh, d:]
            mass = (xa ** 2).sum(0) + (xb ** 2).sum(0)
            keep = torch.zeros(d, dtype=torch.bool, device=DEV)
            keep[mass.argsort(descending=True)[:mm]] = True
            out[:, hh] = X[:, hh] * torch.cat([keep, keep]).float()
        return out
    d = ce(factor_patch(bandmask)) - CE0
    dl = 2 * 2 * NH * 32 * V * 2 * mm
    results['arms'][f'joint band{mm}'] = {'dce': d, 'ratio': dl / FULL}
    print(f'joint band{mm}: dCE {d:+.4f} ratio {dl / FULL:.3f}', flush=True)


# joint positional (per-Δ mean of every head-branch's scores)
def posavg(li, s1, s2, tokens):
    if li != 0:
        return s1, s2
    T = tokens.shape[-1]
    d_idx = (torch.arange(T, device=DEV)[:, None]
             - torch.arange(T, device=DEV)[None, :]).clamp(min=0)
    out = []
    for sx in (s1, s2):
        sxn = sx.clone()
        for hh in range(NH):
            sc = sx[:, hh].float()
            sums = torch.zeros(T, device=DEV).index_add_(
                0, d_idx.flatten(), sc.mean(0).flatten())
            cnts = torch.zeros(T, device=DEV).index_add_(
                0, d_idx.flatten(), torch.ones(T * T, device=DEV))
            sxn[:, hh] = (sums / cnts.clamp(min=1))[d_idx][None].to(sx.dtype)
        out.append(sxn)
    return out[0], out[1]


d = ce(posavg) - CE0
dl = 2 * NH * 32 * (2 * 512 - 1)
results['arms']['joint positional'] = {'dce': d, 'ratio': dl / FULL}
print(f'joint positional: dCE {d:+.4f} ratio {dl / FULL:.2e}', flush=True)


def zero_all(li, s1, s2, tokens):
    return (torch.zeros_like(s1), s2) if li == 0 else (s1, s2)


d = ce(zero_all) - CE0
results['arms']['joint zero'] = {'dce': d, 'ratio': 0.0}
print(f'joint zero: dCE {d:+.4f}', flush=True)

with open(OUT, 'w') as fh:
    json.dump(results, fh, indent=2)
print('joint families done')
