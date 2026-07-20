"""GROUNDING done right: compute gain coefficients on WINDOWED inputs
(streams older than W replaced by cond-mean tables, lambda-rescaled) and
R2 vs live coefficients, for the L16 consumer (mlp16 dir0) and the L15
feeder (boundary feature u0), W in {0,1,2,4,6}. The grounding curve = how
much of each coefficient is token-static-plus-window. Replaces the invalid
per-token estimate from MA-4."""

import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/ground_coeff.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
tok = AutoTokenizer.from_pretrained('gpt2')
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]
CLS = torch.load(f'{QK}/ngram2_pairclass.pt')['cls']
E_hat = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))

# token corpus frequency for junk filtering
cnt = torch.zeros(V)
TR = build_eval_tokens(n_chunks=276, seq_len=513)[20:]
for i in range(0, len(TR), 64):
    fl = TR[i:i + 64].reshape(-1)
    cnt.index_add_(0, fl, torch.ones_like(fl, dtype=torch.float))
class_freq = torch.zeros(256)
for c in range(256):
    ids = (CLS == c)
    class_freq[c] = cnt[ids].sum()
content_classes = set((class_freq > class_freq.median()).nonzero().squeeze(1).tolist())

# mlp16 deviation dirs (recompute quickly from a small sample)
EST = TR[:64, :-1]


@torch.no_grad()
def run(idx, mode=None, arg=None):
    """mode: None | ('head',(l,h)) | ('h7dir',dvec) | ('mlp16dir',(dvec,)) |
    ('mlp16rand',dvec) | ('block',(h,cq,ck)). Returns logits, affected mask,
    and mlp16_out grab when needed."""
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    cls_pos = CLS.to(DEV)[idx]
    affected = torch.ones(B, T, dtype=torch.bool, device=DEV)
    grab16 = None
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
        if mode == 'head' and arg[0] == li:
            s1 = s1.clone(); s2 = s2.clone()
            s1[:, arg[1]] = 0.0; s2[:, arg[1]] = 0.0
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if mode == 'block' and li == 0:
            hh, cq, ck = arg
            bm = (cls_pos[:, :, None] == cq) & (cls_pos[:, None, :] == ck)
            pat = pat.clone()
            pat[:, hh] = pat[:, hh].masked_fill(bm, 0.0)
            affected = bm.any(-1)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if mode == 'h7dir' and li == 5:
            Wo = a.c_proj.weight.detach().float()
            o7 = y[:, :, 7].float() @ Wo[:, 7 * HD:(7 + 1) * HD].T   # (B,T,D)
            proj = torch.einsum('btd,d->bt', o7, arg)
            delta = -proj[..., None] * arg[None, None]
            x = x + a.c_proj(y.reshape(B, T, -1)) + delta.to(x.dtype)
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
            continue
        x = x + a.c_proj(y.reshape(B, T, -1))
        rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
        mlp_out = blk.mlp(x * rms2)
        if li == 16:
            grab16 = mlp_out
            if mode in ('mlp16dir', 'mlp16rand'):
                dvec = arg
                proj = torch.einsum('btd,d->bt', mlp_out.float(), dvec)
                mlp_out = mlp_out - (proj[..., None] * dvec[None, None]).to(mlp_out.dtype)
        x = x + mlp_out
    xf = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(xf) / 30), affected, grab16


# mlp16 dirs from deviation PCA (token-mean over EST sample)
acc = torch.zeros(V, D); c_ = torch.zeros(V)
outs, toks = [], []
for i in range(0, len(EST), 4):
    idx = EST[i:i + 4].to(DEV)
    _, _, g16 = run(idx)
    fl = idx.reshape(-1).cpu()
    acc.index_add_(0, fl, g16.reshape(-1, D).float().cpu())
    c_.index_add_(0, fl, torch.ones_like(fl, dtype=torch.float))
    outs.append(g16.reshape(-1, D).float()); toks.append(fl)
mean16 = acc / c_.clamp_min(1)[:, None]
devs = torch.cat(outs) - mean16[torch.cat(toks)].to(DEV)
Cv = (devs.T @ devs) / len(devs)
ev, evec = torch.linalg.eigh(Cv)
mdirs = evec.flip(1)[:, :4].T                                    # (4, D)
g = torch.Generator(); g.manual_seed(11)
rnd = F.normalize(torch.randn(D, generator=g), dim=0).to(DEV)

blk16 = m.transformer.h[16].mlp
WL = blk16.Left.weight.detach().float()      # (4608, D)
WR = blk16.Right.weight.detach().float()
WD = blk16.Down.weight.detach().float()      # (D, 4608)
bD = blk16.Down_bias.detach().float()

def quad_form(d):
    a = WD.T @ d.to(WD.device)                # (4608,)
    M = torch.einsum('j,jd,je->de', a, WL, WR)
    return 0.5 * (M + M.T), float(d.to(bD.device) @ bD)

# GATE: c_d from the form vs live projection, on one batch
idx = EST[:2].to(DEV)
_, _, g16 = run(idx)
x_pre = None
# recompute the mlp16 INPUT x_hat for the same batch
B, T = idx.shape
x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),))
x0, v1 = x, None
maskT = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
with torch.no_grad():
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
        pat = (s1 * s2).masked_fill(~maskT, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        if li == 16:
            rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            x_pre = (x * rms2).float()
            break
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
d0 = mdirs[0].contiguous()
M0, b0 = quad_form(d0)
c_form = torch.einsum('btd,de,bte->bt', x_pre, M0.to(DEV, torch.float32), x_pre) + b0
c_live = torch.einsum('btd,d->bt', g16.float(), d0)
gate = float((c_form - c_live).abs().max() / c_live.abs().max().clamp_min(1e-9))
print(f'GATE rel-max dev: {gate:.2e} {"PASS" if gate < 1e-3 else "FAIL"}', flush=True)
assert gate < 1e-3, 'quadratic form gate failed'


# ---- Sigma of the mlp16 input over data ----
S = torch.zeros(D, D, device=DEV)
mu = torch.zeros(D, device=DEV)
nS = 0
with torch.no_grad():
    for i0 in range(0, 48, 4):
        id2 = EST[i0:i0 + 4].to(DEV)
        B2, T2 = id2.shape
        x = m.transformer.wte(id2); x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        maskT = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T2, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B2, T2, NH, HD), (HD,)), cosb, sinb)
            q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
            v = a.c_v(h).view(B2, T2, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~maskT, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B2, T2, -1))
            if li == 16:
                rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
                xh = (x * rms2).float().reshape(-1, D)
                S += xh.T @ xh
                mu += xh.sum(0)
                nS += xh.shape[0]
                break
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
mu = mu / nS
S = S / nS - torch.outer(mu, mu)
evS, evecS = torch.linalg.eigh(S.cpu().double())
Shalf = (evecS * evS.clamp_min(1e-10).sqrt()) @ evecS.T          # Sigma^1/2
print('Sigma built', flush=True)



E_hat_c = E_hat.cpu()
U_c = m.lm_head.weight.detach().float().cpu()


def name_vec(v):
    sims = F.cosine_similarity(E_hat_c, v.cpu()[None], dim=1)
    top = sims.abs().topk(4).indices.tolist()
    lens = F.rms_norm(v.cpu()[None], (D,)) @ U_c.T
    ltop = lens[0].abs().topk(4).indices.tolist()
    return {'emb_nn': [tok.decode([t]) for t in top],
            'lens': [tok.decode([t]) for t in ltop]}


Sinvhalf = (evecS * (1.0 / evS.clamp_min(1e-8).sqrt())) @ evecS.T
dd0 = mdirs[0].contiguous()
M0_, b0_ = quad_form(dd0)
W0 = Shalf @ M0_.cpu().double() @ Shalf
lamW0, vecW0 = torch.linalg.eigh(0.5 * (W0 + W0.T))
o0 = lamW0.abs().argsort(descending=True)
u0 = F.normalize((Sinvhalf @ vecW0[:, o0[0]]).float(), dim=0).to(DEV)   # the boundary feature
print('u0 emb_nn:', name_vec(u0)['emb_nn'], flush=True)

# L15 quadratic form for output direction e = u0
blk15 = m.transformer.h[15].mlp
WL15 = blk15.Left.weight.detach().float()
WR15 = blk15.Right.weight.detach().float()
WD15 = blk15.Down.weight.detach().float()
bD15 = blk15.Down_bias.detach().float()
a15 = WD15.T @ u0
M15 = torch.einsum('j,jd,je->de', a15, WL15, WR15)
M15 = 0.5 * (M15 + M15.T)
b15 = float(u0 @ bD15)


RAW = torch.load(f'{QK}/stream_tables.pt')
RAW = {nm: torch.nan_to_num(t.float(), posinf=65504.0, neginf=-65504.0).half()
       for nm, t in RAW.items()}


def created_layer(nm):
    return int(nm[4:] if nm.startswith('attn') else nm[3:])


M16 = M0_.float().to(DEV)
b16 = b0_
targets = {16: (M16, b16, dd0), 15: (M15.to(DEV), b15, u0)}
WS = [0, 1, 2, 4, 6]
buf = {L: {W: [] for W in WS} for L in targets}
buf_live = {L: [] for L in targets}

with torch.no_grad():
    for i0 in range(0, 64, 4):
        idx = EST[i0:i0 + 4].to(DEV)
        B, T = idx.shape
        idx_cpu = idx.cpu()
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        maskT = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
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
            qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
            q, kk_, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
            v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, kk_) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~maskT, 0.0)
            attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + attn_out
            streams[f'attn{li}'] = attn_out
            tabs[f'attn{li}'] = RAW[f'attn{li}'][idx_cpu].to(DEV, x.dtype)
            if li in targets:
                Mt, bt, _ = targets[li]
                rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
                xh = (x * rms2).float()
                c_live = torch.einsum('btd,de,bte->bt', xh, Mt, xh) + bt
                buf_live[li].append(c_live.reshape(-1).cpu())
                for W in WS:
                    old = [nm for nm in streams if created_layer(nm) < li - W]
                    xp = x
                    for nm in old:
                        xp = xp - streams[nm] + tabs[nm]
                    rp = xp.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
                    xhp = (xp * rp).float()
                    cW = torch.einsum('btd,de,bte->bt', xhp, Mt, xhp) + bt
                    buf[li][W].append(cW.reshape(-1).cpu())
            rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x * rms2)
            x = x + mlp_out
            streams[f'mlp{li}'] = mlp_out
            tabs[f'mlp{li}'] = RAW[f'mlp{li}'][idx_cpu].to(DEV, x.dtype)

res = {}
for L in targets:
    cl = torch.cat(buf_live[L])
    var = ((cl - cl.mean()) ** 2).sum()
    res[f'L{L}'] = {}
    for W in WS:
        cw = torch.cat(buf[L][W])
        r2 = float(1 - ((cl - cw) ** 2).sum() / var)
        res[f'L{L}'][f'W={W}'] = round(r2, 3)
    print(f'L{L} grounding R2 by window:', res[f'L{L}'], flush=True)
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2)
print('ground coeff done', flush=True)
