"""L1 first-order codebooks on bilin18: layer-1 QK factors cannot be folded
from weights (inputs are contextual), so estimate CONDITIONAL-MEAN factor
tables q̄(t), k̄(t) per branch from data (post-QK-norm, pre-RoPE), then patch
layer-1 scores with scores_from_factors on those tables. Arms: raw cond-mean,
unit-RMS renormalized, vq256/vq1024 on the winner, and zero-scores control.
ΔCE audited at T=512 through the full 18-layer model (binding metric)."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward
from tier2_folding import scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/l1_condmean_qk.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]

# ---- pass A: conditional means of layer-1 post-QK-norm pre-RoPE factors ----
acc = {n: torch.zeros(V, NH * HD, device=DEV) for n in ('q1', 'k1', 'q2', 'k2')}
cnt = torch.zeros(V, device=DEV)


@torch.no_grad()
def capture(idx):
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    B, T = idx.shape
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    for li in (0, 1):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        if li == 1:
            flat = idx.reshape(-1)
            for name, lin in (('q1', a.c_q), ('k1', a.c_k), ('q2', a.c_q2), ('k2', a.c_k2)):
                z = F.rms_norm(lin(h).view(B, T, NH, HD), (HD,))
                acc[name].index_add_(0, flat, z.reshape(-1, NH * HD).float())
            cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
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
gmean = {n: (a.sum(0) / cnt.sum()) for n, a in acc.items()}
tables = {}
for n, a in acc.items():
    t = a / cnt.clamp_min(1)[:, None]
    t[~seen] = gmean[n]
    tables[n] = t.view(V, NH, HD)
flat_audit = AUDIT[:, :-1].reshape(-1)
cov = seen[flat_audit.to(DEV)].float().mean().item()
print(f'coverage of audit tokens by cond-mean tables: {cov:.4f} '
      f'({int(seen.sum())} vocab rows seen)', flush=True)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


# ---- pass B: audit via score_patch at layer 1 ----
@torch.no_grad()
def audit_ce(tabs=None, zero=False):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 1:
                return s1, s2
            if zero:
                return torch.zeros_like(s1), torch.zeros_like(s2)
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16', score_patch=patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


base = 0.0
res = {'coverage': cov, 'arms': {}}


def run(name, **kw):
    ce = audit_ce(**kw)
    res['arms'][name] = ce - base
    print(f'{name}: dCE {ce - base:+.4f}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)


# baseline: unpatched
tot, n = 0.0, 0
for i in range(0, len(AUDIT), 4):
    b = AUDIT[i:i + 4].to(DEV)
    logits = reference_forward(m, b[:, :-1], 'bf16').float()
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
    tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
base = tot / n
res['baseline_ce'] = base
print(f'baseline CE {base:.4f}', flush=True)

run('L1 zero-scores (control)', zero=True)
run('L1 cond-mean raw', tabs=tables)
tab_u = {k: unit_rms(v) for k, v in tables.items()}
run('L1 cond-mean unit-RMS', tabs=tab_u)


def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k].to(X.device)].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 4096):
            xx = X[i:i + 4096]
            assign[i:i + 4096] = ((xx * xx).sum(1, True) - 2 * xx @ C.T
                                  + (C * C).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C)
        c2 = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X)
        c2.index_add_(0, assign, torch.ones(len(X), device=X.device))
        nz = c2 > 0
        C[nz] = Cn[nz] / c2[nz][:, None]
    return assign, C


def vq_tables(src, k):
    out = {n: torch.empty_like(t) for n, t in src.items()}
    for h in range(NH):
        for br, (qn, kn) in enumerate((('q1', 'k1'), ('q2', 'k2'))):
            X = torch.cat([src[qn][:, h], src[kn][:, h]], 1)   # (V, 2*HD) shared partition
            assign, C = kmeans(X, k, seed=h * 2 + br)
            out[qn][:, h] = C[assign][:, :HD]
            out[kn][:, h] = C[assign][:, HD:]
    return out


# vq on whichever normalization won (decided at runtime by smaller dCE)
best = tables if res['arms']['L1 cond-mean raw'] <= res['arms']['L1 cond-mean unit-RMS'] else tab_u
bname = 'raw' if best is tables else 'unit-RMS'
for kk in (256, 1024):
    run(f'L1 cond-mean {bname} + vq{kk}', tabs=vq_tables(best, kk))
print('l1 condmean done', flush=True)
