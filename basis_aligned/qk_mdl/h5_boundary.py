"""Boundary confirmation (CARD-2): rerun the H5 cleaning arms on NATURAL-TEXT
repeats (pile 256-token snippets, repeated A+A) — if cleaning hurts here too,
the context-mixed-identity boundary holds at corpus scale.
Original: Why does bilin18 under-cash its induction head? (WW-6: low-rank filtering
H5's output IMPROVED repeats by 0.33.) Two hypotheses, one harness:
 A. content noise — replace H5's v-content with cond-mean-by-source-token
    (clean identity, live pattern). If repeats improve, the carried content
    was noisy, not the match.
 B. amplitude starvation — scale H5's output by alpha in {1.5, 2, 4}. If
    repeats improve monotonically, the model learned too small a gain.
Audits: natural + repeat-2nd-half (A+A random, len 256)."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
L, HEAD = 5, 5
OUT = f'{QK}/h5_boundary.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]
EST = build_eval_tokens(n_chunks=20 + 512, seq_len=513)[20:][:, :-1]
g = torch.Generator(); g.manual_seed(7)
REP_RAND = torch.randint(0, V, (16, 256), generator=g).repeat(1, 2)
NAT = build_eval_tokens(n_chunks=16, seq_len=257)[:, :-1]      # natural 256-tok snippets
REP = torch.cat([NAT, NAT], dim=1)                              # natural A+A

# cond-mean v-content for head 5 at L5 (post-lerp v, by current token)
acc = torch.zeros(V, HD)
cnt = torch.zeros(V)


@torch.no_grad()
def forward(idx, vbar=None, alpha=None, grab_v=False):
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
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
        if li == L:
            if grab_v:
                return v[:, :, HEAD]
            if vbar is not None:
                v = v.clone()
                v[:, :, HEAD] = vbar[idx.cpu()].to(DEV, v.dtype)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if li == L and alpha is not None:
            pat = pat.clone()
            pat[:, HEAD] = pat[:, HEAD] * alpha
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    xf = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(xf) / 30)


with torch.no_grad():
    for i in range(0, len(EST), 8):
        idx = EST[i:i + 8].to(DEV)
        vh = forward(idx, grab_v=True)
        flat = idx.reshape(-1).cpu()
        acc.index_add_(0, flat, vh.reshape(-1, HD).float().cpu())
        cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
vbar = acc / cnt.clamp_min(1)[:, None]
vbar[cnt == 0] = acc.sum(0) / cnt.sum()
print('vbar built', flush=True)


@torch.no_grad()
def ce_eval(tokens, second=False, **kw):
    tot, n = 0.0, 0
    for i in range(0, len(tokens), 4):
        b = tokens[i:i + 4].to(DEV)
        logits = forward(b[:, :-1], **kw).float()
        tgt = b[:, 1:]
        if second:
            logits, tgt = logits[:, 256:], tgt[:, 256:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        tot += ce.item() * tgt.numel(); n += tgt.numel()
    return tot / n


bn = ce_eval(AUDIT)
br = ce_eval(REP, second=True)
res = {'baseline': {'natural': bn, 'repeat': br}}
print(f'baseline natural {bn:.4f} repeat {br:.4f}', flush=True)
arms = [('NATURAL repeats: v-content cleaned', dict(vbar=vbar))]
# also random-repeat reference on the same harness (should IMPROVE, reproducing WW-7)
br_rand = ce_eval(REP_RAND, second=True)
dn_r = ce_eval(REP_RAND, second=True, vbar=vbar) - br_rand
res['random-repeat reference: cleaned'] = {'d_repeat': dn_r}
print(f'random-repeat reference cleaned: {dn_r:+.4f} (expect negative/improve)', flush=True)
for name, kw in arms:
    dn = ce_eval(AUDIT, **kw) - bn
    dr = ce_eval(REP, second=True, **kw) - br
    res[name] = {'d_natural': dn, 'd_repeat': dr}
    print(f'{name}: natural {dn:+.4f} · repeat {dr:+.4f}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('h5 boundary done', flush=True)
