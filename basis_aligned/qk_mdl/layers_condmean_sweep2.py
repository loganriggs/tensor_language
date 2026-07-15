"""Depth sweep of the L1 result: for layers L in {1,2,3,5,8,12,17} of bilin18,
estimate conditional-mean QK factor tables (post-QK-norm pre-RoPE, unit-RMS)
from one live pass over 524k tokens (accumulators on CPU), then patch each
layer alone and audit full-model dCE at T=512. Zero-scores control per layer
gives the load-bearing denominator. Question: is selection ~0th-order in
context at every depth, or only near the bottom?"""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward
from tier2_folding import scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
LAYERS = [4, 6, 7, 9, 10, 11, 13, 14, 15, 16]
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/layers_condmean_sweep2.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]

acc = {(L, n): torch.zeros(V, NH * HD) for L in LAYERS for n in ('q1', 'k1', 'q2', 'k2')}
cnt = torch.zeros(V)
LMAX = max(LAYERS)


@torch.no_grad()
def capture(idx):
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    B, T = idx.shape
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    flat = idx.reshape(-1).cpu()
    cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        if li in LAYERS:
            for name, lin in (('q1', a.c_q), ('k1', a.c_k), ('q2', a.c_q2), ('k2', a.c_k2)):
                z = F.rms_norm(lin(h).view(B, T, NH, HD), (HD,))
                acc[(li, name)].index_add_(0, flat, z.reshape(-1, NH * HD).float().cpu())
        if li == LMAX:
            return
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cos, sin)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))


for i in range(0, len(TRAIN), 8):
    capture(TRAIN[i:i + 8, :-1].to(DEV))
    if i % 256 == 0:
        print(f'  estimate {i}/{len(TRAIN)}', flush=True)
seen = cnt > 0
print(f'{int(seen.sum())} vocab rows seen', flush=True)


def layer_tables(L):
    out = {}
    for n in ('q1', 'k1', 'q2', 'k2'):
        a = acc[(L, n)]
        t = a / cnt.clamp_min(1)[:, None]
        t[~seen] = a.sum(0) / cnt.sum()
        t = t.view(V, NH, HD)
        t = t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())
        out[n] = t.to(DEV)
    return out


@torch.no_grad()
def audit_ce(L=None, tabs=None, zero=False):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != L:
                return s1, s2
            if zero:
                return torch.zeros_like(s1), torch.zeros_like(s2)
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16',
                                   score_patch=None if L is None else patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


base = audit_ce()
print(f'baseline CE {base:.4f}', flush=True)
res = {'baseline_ce': base, 'layers': {}}
for L in LAYERS:
    tabs = layer_tables(L)
    dz = audit_ce(L=L, zero=True) - base
    dc = audit_ce(L=L, tabs=tabs) - base
    res['layers'][L] = {'zero': dz, 'condmean': dc}
    print(f'L{L}: zero {dz:+.4f}  cond-mean {dc:+.4f}', flush=True)
    del tabs
    torch.cuda.empty_cache()
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('layers condmean sweep done', flush=True)
