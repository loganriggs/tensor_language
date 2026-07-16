"""Method E, experiment 1 (Logan's pick): BACKWARD (unembedding-relative)
metric for the stream-table quantization. Forward vq (current) clusters table
rows by L2 in activation space; the backward metric weights each dimension by
its downstream consumption — diagonal Fisher E[(dLoss/d stream)^2], estimated
by backprop through the LIVE model. Same centroid rule, different ASSIGNMENTS
(whitened-space k-means). Test at k where forward-L2 has real cost: composed
windowed-D W=4 audits, k in {64, 256}, both metrics. If Logan's conjecture is
right, the backward optimum differs — Fisher-vq should win at small k.
MDL convention (set this tick, logged): structural bits and estimation-token
counts reported side by side, never converted into each other."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/e1_backward_vq.json'
W = 4
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]
SNAMES = [f'{t}{l}' for l in range(17) for t in ('attn', 'mlp')] + ['attn17', 'mlp17']


def created_layer(nm):
    return int(nm[4:]) if nm.startswith('attn') else int(nm[3:])


# ---- pass 1: diagonal Fisher per stream (backprop through live model) ----
fisher = {nm: torch.zeros(D, device=DEV) for nm in SNAMES}
NB = 48
for bi in range(NB):
    idx = TRAIN[bi * 2:(bi + 1) * 2, :-1].to(DEV)
    tgt = TRAIN[bi * 2:(bi + 1) * 2, 1:].to(DEV)
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    grabbed = {}
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        attn_out.retain_grad()
        grabbed[f'attn{li}'] = attn_out
        x = x + attn_out
        rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
        mlp_out = blk.mlp(x * rms2)
        mlp_out.retain_grad()
        grabbed[f'mlp{li}'] = mlp_out
        x = x + mlp_out
    xf = F.rms_norm(x, (x.size(-1),))
    logits = 30 * torch.tanh(m.lm_head(xf) / 30)
    loss = F.cross_entropy(logits.reshape(-1, V).float(), tgt.reshape(-1))
    loss.backward()
    for nm, t in grabbed.items():
        if t.grad is not None:
            fisher[nm] += t.grad.float().pow(2).sum((0, 1))
    m.zero_grad(set_to_none=True)
    if bi % 12 == 0:
        print(f'  fisher batch {bi}/{NB}', flush=True)
fisher = {nm: (f / NB).cpu() for nm, f in fisher.items()}
torch.save(fisher, f'{QK}/stream_fisher.pt')
print('fisher built + saved', flush=True)

RAW = torch.load(f'{QK}/stream_tables.pt')


def kmeans(X, k, seed):
    g = torch.Generator(); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k].to(X.device)].clone()
    for _ in range(10):
        a_ = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 2048):
            xx = X[i:i + 2048]
            a_[i:i + 2048] = ((xx * xx).sum(1, True) - 2 * xx @ C.T
                              + (C * C).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C)
        c2 = torch.zeros(k, device=X.device)
        Cn.index_add_(0, a_, X)
        c2.index_add_(0, a_, torch.ones(len(X), device=X.device))
        nz = c2 > 0
        C[nz] = Cn[nz] / c2[nz][:, None]
    return a_, C


def vq_tables(k, metric):
    out = {}
    for nm, t in RAW.items():
        X = t.float().to(DEV)
        if metric == 'fisher':
            wgt = fisher[nm].to(DEV).clamp_min(fisher[nm].max().item() * 1e-6).sqrt()
            a_, _ = kmeans(X * wgt[None], k, seed=hash(nm) % 2**31)
        else:
            a_, _ = kmeans(X, k, seed=hash(nm) % 2**31)
        # centroids in ORIGINAL space = mean of members (same rule both metrics)
        C = torch.zeros(k, D, device=DEV)
        c2 = torch.zeros(k, device=DEV)
        C.index_add_(0, a_, X)
        c2.index_add_(0, a_, torch.ones(V, device=DEV))
        nz = c2 > 0
        C[nz] = C[nz] / c2[nz][:, None]
        out[nm] = C[a_].half().cpu()
        del X
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def audit_ce(tables, Wa=W):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
        idx_cpu = idx.cpu()
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        streams, tabs = {}, {}
        for li, blk in enumerate(m.transformer.h):
            lam0, lam1 = blk.lambdas[0], blk.lambdas[1]
            x = lam0 * x + lam1 * x0
            for nm in streams:
                streams[nm] = lam0 * streams[nm]
                tabs[nm] = lam0 * tabs[nm]
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            h_qk = h
            if li >= 1:
                old = [nm for nm in streams if created_layer(nm) < li - Wa]
                if old:
                    xp = x
                    for nm in old:
                        xp = xp - streams[nm] + tabs[nm]
                    h_qk = F.rms_norm(xp, (xp.size(-1),))
            qn = lambda lin: apply_rot(F.rms_norm(lin(h_qk).view(B, T, NH, HD), (HD,)), cosb, sinb)
            q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
            v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + attn_out
            streams[f'attn{li}'] = attn_out
            tabs[f'attn{li}'] = tables[f'attn{li}'][idx_cpu].to(DEV, x.dtype)
            rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x * rms2)
            x = x + mlp_out
            streams[f'mlp{li}'] = mlp_out
            tabs[f'mlp{li}'] = tables[f'mlp{li}'][idx_cpu].to(DEV, x.dtype)
        xf = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(xf) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


tot, n = 0.0, 0
with torch.no_grad():
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        B, T = idx.shape
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
            q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
            v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        xf = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(xf) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
base = tot / n
res = {'baseline_ce': base, 'arms': {}}
print(f'baseline {base:.4f}', flush=True)
for k in (64, 256):
    for metric in ('l2', 'fisher'):
        tabs = vq_tables(k, metric)
        d = audit_ce(tabs) - base
        res['arms'][f'W={W} vq{k} {metric}'] = d
        print(f'W={W} vq{k} {metric}: dCE {d:+.4f}', flush=True)
        with open(OUT, 'w') as fh:
            json.dump(res, fh, indent=2)
print('e1 backward vq done', flush=True)
