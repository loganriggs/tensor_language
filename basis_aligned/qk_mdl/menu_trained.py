"""Stage-B flagship: menu-static, jointly CE-trained. Layers {8,14,15,17}
zeroed; every other layer 1-17 gets vq256 class tables (shared [q|k] partition
per head-branch, from the all17 cond-mean tables); L0 stays live (exact fold).
No live QK selection anywhere above L0. ~15M trainable class-table floats,
protocol wants ~3M train tokens. Composed L2-fit start point is ~+1.8 (stage A);
question: does joint CE repair close a 10x superadditive composition blowup?"""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward

torch.manual_seed(0)
DEV = 'cuda'
ZERO_L = {8, 14, 15, 17}
TAB_L = [L for L in range(1, 18) if L not in ZERO_L]
K = 256
STEPS = 4500
BATCH = 2
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/menu_trained.json'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 6144, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]
print(f'train tokens: {TRAIN.numel()/1e6:.2f}M', flush=True)

raw = torch.load('all17_tables.pt')
assigns, cbs = {}, {}
for L in TAB_L:
    tabs = {n: raw[f'{L}_{n}'].float() for n in ('q1', 'k1', 'q2', 'k2')}
    for br, (qn, kn) in enumerate((('q1', 'k1'), ('q2', 'k2'))):
        for h in range(NH):
            X = torch.cat([tabs[qn][:, h], tabs[kn][:, h]], 1).to(DEV)
            g = torch.Generator(); g.manual_seed(L * 100 + h * 2 + br)
            C = X[torch.randperm(V, generator=g)[:K].to(DEV)].clone()
            for _ in range(12):
                a_ = torch.empty(V, dtype=torch.long, device=DEV)
                for i in range(0, V, 4096):
                    xx = X[i:i + 4096]
                    a_[i:i + 4096] = ((xx * xx).sum(1, True) - 2 * xx @ C.T
                                      + (C * C).sum(1)[None]).argmin(1)
                Cn = torch.zeros_like(C)
                c2 = torch.zeros(K, device=DEV)
                Cn.index_add_(0, a_, X)
                c2.index_add_(0, a_, torch.ones(V, device=DEV))
                nz = c2 > 0
                C[nz] = Cn[nz] / c2[nz][:, None]
            assigns[(L, h, br)] = a_
            cbs[(L, h, br)] = C.requires_grad_(True)
del raw
print(f'vq built: {len(cbs)} codebooks, '
      f'{sum(c.numel() for c in cbs.values())/1e6:.1f}M floats', flush=True)


def layer_factors(L, idx):
    """(B,T,NH,HD) q,k factor pairs per branch from trainable codebooks."""
    out = []
    for br in range(2):
        rows = torch.stack([cbs[(L, h, br)][assigns[(L, h, br)][idx]]
                            for h in range(NH)], 2)      # (B,T,NH,2*HD)
        out.append((rows[..., :HD], rows[..., HD:]))
    return out


def cs_scores(Fq, Fk, hd):
    d = hd // 2
    T = Fq.shape[1]
    cos, sin = rope_tables(T, hd, DEV, torch.float32, 'bf16')
    cosD = torch.einsum('if,jf->ijf', cos, cos) + torch.einsum('if,jf->ijf', sin, sin)
    sinD = torch.einsum('if,jf->ijf', sin, cos) - torch.einsum('if,jf->ijf', cos, sin)
    qa, qb = Fq[..., :d], Fq[..., d:]
    ka, kb = Fk[..., :d], Fk[..., d:]
    s = (torch.einsum('bihf,bjhf,ijf->bhij', qa, ka, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', qb, kb, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', qb, ka, sinD)
         - torch.einsum('bihf,bjhf,ijf->bhij', qa, kb, sinD))
    return s / hd


def forward(idx):
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
        if li == 0 or li in ZERO_L or li in TAB_L:
            v = a.c_v(h).view(B, T, NH, HD)
            if v1 is None:
                v1 = v
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        if li in ZERO_L:
            pass                                          # attention silenced
        else:
            if li == 0:
                qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cos, sin)
                q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
                s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
                s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            else:
                (fq1, fk1), (fq2, fk2) = layer_factors(li, idx)
                s1, s2 = cs_scores(fq1, fk1, HD), cs_scores(fq2, fk2, HD)
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
print(f'baseline {base:.4f}; menu-static vq{K} L2-fit dCE {d0:+.4f}', flush=True)
res = {'baseline': base, 'l2fit': d0, 'checkpoints': {}}

params = list(cbs.values())
opt = torch.optim.Adam(params, lr=1e-3)
g = torch.Generator(); g.manual_seed(1)
for step in range(STEPS):
    b = TRAIN[torch.randint(0, len(TRAIN), (BATCH,), generator=g)].to(DEV)
    logits = forward(b[:, :-1]).float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if step % 300 == 0:
        print(f'  step {step} CE {loss.item():.4f}', flush=True)
    if step in (1500, 3000):
        d = audit() - base
        res['checkpoints'][step] = d
        print(f'  held-out @{step}: dCE {d:+.4f}', flush=True)
        with open(OUT, 'w') as fh:
            json.dump(res, fh, indent=2)
dT = audit() - base
res['ce_trained'] = dT
print(f'MENU-STATIC vq{K} CE-trained: dCE {dT:+.4f}', flush=True)
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2)
torch.save({f'{L}_{h}_{br}': cbs[(L, h, br)].detach().cpu()
            for (L, h, br) in cbs}, 'menu_cbs_trained.pt')
print('menu trained done', flush=True)
