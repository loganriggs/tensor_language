"""WW-5 causal check: if H7 is a rank-1 gain head, replacing its output by
its projection onto the top-k deviation-PCA directions (+ the constant mean
direction, live scalar coefficients) should be free. Arms k in {1,2,4,8};
audits: natural + repeat-2nd-half. H5 same treatment as contrast (expect
rank-1 to HURT: identity content is high-rank)."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
L = 5
OUT = f'{QK}/h7_rank1.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]
g = torch.Generator(); g.manual_seed(7)
REP = torch.randint(0, V, (16, 256), generator=g).repeat(1, 2)
Wo = m.transformer.h[L].attn.c_proj.weight.detach().float()

# rebuild deviation PCs quickly from a smaller sample + the mean direction
TOKS = build_eval_tokens(n_chunks=48, seq_len=513)[:, :-1]


@torch.no_grad()
def forward(idx, head=None, basis=None):
    """basis: (k, D) orthonormal — H`head`'s output projected onto span(basis)."""
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    grab = None
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
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v)          # (B,T,NH,HD)
        if li == L and head is not None:
            o = y[:, :, head] @ Wo[:, head * HD:(head + 1) * HD].T.to(DEV)  # (B,T,D)
            if basis is None:
                grab = o
            else:
                proj = torch.einsum('btd,kd->btk', o, basis)
                o_new = torch.einsum('btk,kd->btd', proj, basis)
                # subtract full head contribution, add projected one
                full = y.reshape(B, T, -1) @ m.transformer.h[L].attn.c_proj.weight.T
                delta = o_new - o
                x = x + full + delta
                x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
                continue
        x = x + a.c_proj(y.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    if grab is not None:
        return grab
    xf = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(xf) / 30)


# build bases: mean direction + top deviation PCs per head
bases = {}
for head in (7, 5):
    outs, toks = [], []
    with torch.no_grad():
        for i in range(0, len(TOKS), 4):
            idx = TOKS[i:i + 4].to(DEV)
            outs.append(forward(idx, head=head).reshape(-1, D))
            toks.append(idx.reshape(-1))
    O = torch.cat(outs)
    mu = O.mean(0, keepdim=True)
    Od = O - mu
    C = (Od.T @ Od) / len(Od)
    evals, evecs = torch.linalg.eigh(C)
    pcs = evecs.flip(1)[:, :8].T                            # (8, D)
    stack = torch.cat([F.normalize(mu, dim=1), pcs])
    Qmat, _ = torch.linalg.qr(stack.T)
    bases[head] = Qmat.T                                    # (9, D) orthonormal
print('bases built', flush=True)


@torch.no_grad()
def ce_eval(tokens, head=None, basis=None, second=False):
    tot, n = 0.0, 0
    for i in range(0, len(tokens), 4):
        b = tokens[i:i + 4].to(DEV)
        logits = forward(b[:, :-1], head=head, basis=basis).float()
        tgt = b[:, 1:]
        if second:
            logits, tgt = logits[:, 256:], tgt[:, 256:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        tot += ce.item() * tgt.numel(); n += tgt.numel()
    return tot / n


res = {}
bn = ce_eval(AUDIT)
br = ce_eval(REP, second=True)
res['baseline'] = {'natural': bn, 'repeat': br}
print(f'baseline natural {bn:.4f} repeat {br:.4f}', flush=True)
for head in (7, 5):
    for kk in (1, 2, 4, 8):
        basis = bases[head][:1 + kk]
        dn = ce_eval(AUDIT, head=head, basis=basis) - bn
        dr = ce_eval(REP, head=head, basis=basis, second=True) - br
        res[f'H{head} rank{kk}+mean'] = {'d_natural': dn, 'd_repeat': dr}
        print(f'H{head} rank-{kk}(+mean): natural {dn:+.4f} · repeat {dr:+.4f}', flush=True)
        with open(OUT, 'w') as fh:
            json.dump(res, fh, indent=2)
print('h7 rank1 done', flush=True)
