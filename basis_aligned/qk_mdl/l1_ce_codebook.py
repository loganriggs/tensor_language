"""CE-train the L1 vq256 codebook (from l1_condmean_qk.py): re-estimate
cond-mean factor tables, kmeans shared [q|k] partition per head-branch,
freeze assignments, train the 1M class-table floats through the frozen
18-layer model with the score patch at layer 1. Saves tables to l1_tables.pt."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward
from tier2_folding import scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
K = 256
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/l1_ce_codebook.json'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]

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
seen = cnt > 0
tables = {}
for n, a in acc.items():
    t = a / cnt.clamp_min(1)[:, None]
    t[~seen] = (a.sum(0) / cnt.sum())
    t = t.view(V, NH, HD)
    tables[n] = t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())
print('cond-mean tables built', flush=True)


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


assigns, cbs = {}, {}
for h in range(NH):
    for br, (qn, kn) in enumerate((('q1', 'k1'), ('q2', 'k2'))):
        X = torch.cat([tables[qn][:, h], tables[kn][:, h]], 1)
        a_, C_ = kmeans(X, K, seed=h * 2 + br)
        assigns[(h, br)] = a_
        cbs[(h, br)] = C_.clone().requires_grad_(True)
torch.save({'tables': {k: v.cpu() for k, v in tables.items()},
            'assigns': {f'{h}_{br}': v.cpu() for (h, br), v in assigns.items()}},
           'l1_tables.pt')
del acc, cnt, tables, seen
torch.cuda.empty_cache()
print('vq built + saved', flush=True)


def vq_now():
    out = {n: torch.empty(V, NH, HD, device=DEV) for n in ('q1', 'k1', 'q2', 'k2')}
    for h in range(NH):
        for br, (qn, kn) in enumerate((('q1', 'k1'), ('q2', 'k2'))):
            row = cbs[(h, br)][assigns[(h, br)]]
            out[qn][:, h] = row[:, :HD]
            out[kn][:, h] = row[:, HD:]
    return out


def sff_grad(qh, kh, tokens, hd):
    # scores_from_factors is @torch.no_grad-decorated; differentiable copy
    Fq, Fk = qh[tokens], kh[tokens]
    d = hd // 2
    T = tokens.shape[1]
    cos, sin = rope_tables(T, hd, tokens.device, qh.dtype, 'bf16')
    cosD = torch.einsum('if,jf->ijf', cos, cos) + torch.einsum('if,jf->ijf', sin, sin)
    sinD = torch.einsum('if,jf->ijf', sin, cos) - torch.einsum('if,jf->ijf', cos, sin)
    qa, qb = Fq[..., :d], Fq[..., d:]
    ka, kb = Fk[..., :d], Fk[..., d:]
    s = (torch.einsum('bihf,bjhf,ijf->bhij', qa, ka, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', qb, kb, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', qb, ka, sinD)
         - torch.einsum('bihf,bjhf,ijf->bhij', qa, kb, sinD))
    return s / hd


def forward(idx, grad=False):
    tabs = vq_now()

    def patch(li, s1, s2):
        if li != 1:
            return s1, s2
        n1 = sff_grad(tabs['q1'], tabs['k1'], idx, HD)
        n2 = sff_grad(tabs['q2'], tabs['k2'], idx, HD)
        return n1.to(s1.dtype), n2.to(s2.dtype)

    if grad:
        return _rf_grad(idx, patch)
    return reference_forward(m, idx, 'bf16', score_patch=patch)


def _rf_grad(idx, patch):
    # reference_forward is @torch.no_grad; replicate with grad enabled
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    B, T = idx.shape
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cos, sin)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        s1, s2 = patch(li, s1, s2)
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30)


@torch.no_grad()
def audit():
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        logits = forward(b[:, :-1]).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot / n


tot, n = 0.0, 0
with torch.no_grad():
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        logits = reference_forward(m, b[:, :-1], 'bf16').float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
base = tot / n
d0 = audit() - base
print(f'baseline {base:.4f}; L1 vq{K} L2-fit dCE {d0:+.4f}', flush=True)

params = list(cbs.values())
print(f'training {sum(p.numel() for p in params)/1e6:.1f}M table floats', flush=True)
opt = torch.optim.Adam(params, lr=1e-3)
g = torch.Generator(); g.manual_seed(1)
for step in range(1500):
    b = TRAIN[torch.randint(0, len(TRAIN), (4,), generator=g)].to(DEV)
    logits = forward(b[:, :-1], grad=True).float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if step % 300 == 0:
        print(f'  step {step} CE {loss.item():.4f}', flush=True)
dT = audit() - base
print(f'L1 vq{K} CE-trained dCE {dT:+.4f}', flush=True)
json.dump({'baseline': base, 'l2fit': d0, 'ce_trained': dT},
          open(OUT, 'w'), indent=2)
torch.save({(f'{h}_{br}'): cbs[(h, br)].detach().cpu()
            for h in range(NH) for br in (0, 1)}, 'l1_cbs_trained.pt')
print('l1 ce codebook done', flush=True)
