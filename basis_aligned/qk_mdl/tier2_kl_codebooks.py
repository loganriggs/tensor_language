"""KL-distilled vq codebooks (teacher = original model) — separates faithful
compression from domain adaptation in the CE-trained results.

Original docstring: CE-trained vq codebooks for bilin18 layer-0 (the basis_aligned e7 lesson
applied to attention): freeze each token's cluster ASSIGNMENT, make the
centroid factor tables trainable, train through the frozen model on pile-10k
CE (train chunks disjoint from the audit set), bf16 + grad clipping (the e7b
divergence lesson). Joint (all 9 heads, both branches) for k in {16, 64, 256}.

Reports the frontier shift: joint dCE before (L2-fit centroids) -> after
(CE-trained centroids) at identical DL.
"""

import json
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
from tier2_folding import branch_factors

torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/tier2_kl_codebooks.json'

m, cfg = load_elriggs('bilin18')
m.to(torch.bfloat16)
for p in m.parameters():
    p.requires_grad_(False)
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
V = cfg['vocab_size']

ALL = build_eval_tokens(n_chunks=20 + 128, seq_len=513)
AUDIT = ALL[4:20]          # same audit chunks as tier2_audit (indices 4..19)
TRAIN = ALL[20:].to(DEV)
FACT = {br: branch_factors(m, br, dtype=torch.float32) for br in (1, 2)}


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


def scores_all_heads(qtab, ktab, qassign, kassign, tokens):
    """Differentiable branch scores (B, NH, T, T) from centroid tables.
    qtab/ktab: (NH, k, HD) trainable; assigns: (V, NH) long."""
    B, T = tokens.shape
    d = HD // 2
    cos, sin = rope_tables(T, HD, tokens.device, torch.float32, 'bf16')
    cosD = torch.einsum('if,jf->ijf', cos, cos) + torch.einsum('if,jf->ijf', sin, sin)
    sinD = torch.einsum('if,jf->ijf', sin, cos) - torch.einsum('if,jf->ijf', cos, sin)
    Fq = torch.stack([qtab[h][qassign[tokens][:, :, h]] for h in range(NH)], 2)
    Fk = torch.stack([ktab[h][kassign[tokens][:, :, h]] for h in range(NH)], 2)
    qa, qb = Fq[..., :d], Fq[..., d:]
    ka, kb = Fk[..., :d], Fk[..., d:]
    s = (torch.einsum('bihf,bjhf,ijf->bhij', qa, ka, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', qb, kb, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', qb, ka, sinD)
         - torch.einsum('bihf,bjhf,ijf->bhij', qa, kb, sinD))
    return s / HD


def forward_train(tokens, params):
    """bf16 forward with layer-0 scores from trainable codebooks."""
    x = m.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = tokens.shape
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        nh, hd = a.n_head, a.head_dim
        cos, sin = rope_tables(T, hd, tokens.device, x.dtype, 'bf16')
        cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]

        def qk(lin):
            z = lin(h).view(B, T, nh, hd)
            return apply_rot(F.rms_norm(z, (hd,)), cosr, sinr)

        v = a.c_v(h).view(B, T, nh, hd)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        mask = torch.tril(torch.ones(T, T, device=tokens.device, dtype=torch.bool))
        if li == 0:
            s1 = scores_all_heads(params['q1'], params['k1'],
                                  params['aq1'], params['ak1'], tokens).to(x.dtype)
            s2 = scores_all_heads(params['q2'], params['k2'],
                                  params['aq2'], params['ak2'], tokens).to(x.dtype)
        else:
            q, k = qk(a.c_q), qk(a.c_k)
            q2, k2 = qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / hd
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / hd
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30)


@torch.no_grad()
def audit_ce(params=None, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), batch):
        b = AUDIT[i:i + batch].to(DEV)
        if params is None:
            from tier2_model import reference_forward
            logits = reference_forward(m, b[:, :-1]).float()
        else:
            logits = forward_train(b[:, :-1], params).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


CE0 = audit_ce()
print(f'baseline CE (bf16, T=512): {CE0:.4f}')
results = {'baseline_ce': CE0, 'runs': {}}

for K in [64]:
    # build joint vq codebooks (separate q/k clusterings per head-branch would
    # change DL; match tier2_audit: one clustering on [q|k] per head-branch)
    params = {}
    for br in (1, 2):
        qh_all, kh_all = FACT[br]
        qt, kt, aq, ak = [], [], [], []
        for hh in range(NH):
            C, assign = kmeans(torch.cat([qh_all[:, hh], kh_all[:, hh]], 1), K)
            qt.append(C[:, :HD].clone())
            kt.append(C[:, HD:].clone())
            aq.append(assign)
            ak.append(assign)
        params[f'q{br}'] = torch.stack(qt).requires_grad_(True)
        params[f'k{br}'] = torch.stack(kt).requires_grad_(True)
        params[f'aq{br}'] = torch.stack(aq, 1)   # (V, NH)
        params[f'ak{br}'] = torch.stack(ak, 1)
    dce_before = audit_ce(params) - CE0
    print(f'k={K}: joint dCE before CE-training {dce_before:+.4f}')

    opt = torch.optim.Adam([params[f'{t}{br}'] for t in 'qk' for br in (1, 2)], lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1200)
    g = torch.Generator(); g.manual_seed(0)
    run = None
    for step in range(1200):
        b = TRAIN[torch.randint(0, len(TRAIN), (4,), generator=g)]
        with torch.no_grad():
            from tier2_model import reference_forward
            t_logits = reference_forward(m, b[:, :-1])
        s_logits = forward_train(b[:, :-1], params)
        sf = s_logits.reshape(-1, s_logits.shape[-1])
        tf = t_logits.reshape(-1, t_logits.shape[-1])
        loss = 0.0
        for i0 in range(0, sf.shape[0], 2048):
            p_t = F.softmax(tf[i0:i0 + 2048].float(), dim=-1)
            loss = loss + -(p_t * F.log_softmax(sf[i0:i0 + 2048].float(), dim=-1)).sum(-1).sum()
        loss = loss / sf.shape[0]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [params[f'{t}{br}'] for t in 'qk' for br in (1, 2)], 1.0)
        opt.step(); sched.step()
        run = loss.item() if run is None else 0.95 * run + 0.05 * loss.item()
        if step % 200 == 0:
            print(f'  k={K} step {step:5d} train CE (ema) {run:.4f}', flush=True)
    dce_after = audit_ce(params) - CE0
    dl_bits = 2 * NH * (32 * K * 2 * HD + V * math.log2(K))
    results['runs'][f'vq{K}'] = {
        'dce_before': dce_before, 'dce_after': dce_after,
        'dl_bits': dl_bits, 'ratio': dl_bits / (32 * 2 * 2 * V * HD * NH)}
    print(f'k={K}: joint dCE {dce_before:+.4f} -> {dce_after:+.4f} '
          f'(DL ratio {results["runs"][f"vq{K}"]["ratio"]:.2e})', flush=True)
    torch.save({f'{t}{br}': params[f'{t}{br}'].detach().cpu()
                for t in 'qk' for br in (1, 2)},
               f'/workspace/tensor_language/basis_aligned/qk_mdl/kl_codebook_k{K}.pt')
    with open(OUT, 'w') as fh:
        json.dump(results, fh, indent=2)
print('kl codebooks done')
