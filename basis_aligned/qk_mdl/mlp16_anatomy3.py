"""MLP16 ANATOMY, experiment 3: (a) CAUSAL rank-r form replacement — at L16,
out += (c_k - c)·d with c_k = sum_{r<=k} lam_r (u_r·x_hat)^2 from the whitened
eigenbasis (u_r = Sigma^-1/2 v_r); dCE + R^2(c_k, c), k in {4,16,64}, dirs 0-3
jointly; (b) EXACT stream-pair variance split of dir0 coefficient (which
stream interactions feed the boundary features). Original exp 2: DATA-WHITENED spectrum + stream-pair split.
MA-1: weight-space M_d is high-rank (~600); behavioral gain is rank-4-16 →
the concentration must be in the data. Compute Sigma (cov of mlp16 INPUT
x_hat over data), whitened form W = Sigma^1/2 M Sigma^1/2, spectrum + named
top data-space features; plus EXACT stream-pair variance split of c_d.
Original exp 1 (Logan's pick 2026-07-20): the gain
coefficient of output direction d is the EXACT quadratic form
  c_d(x) = x_hat^T M_d x_hat + d.b_D,   M_d = sum_j (W_D^T d)_j W_L[j] (x) W_R[j].
GATE: reproduce live coefficients to fp tolerance. Then:
  (a) eigen-anatomy of sym(M_d) for dirs 0-3: effective rank of the form;
  (b) name top eigenvectors (embedding-class + logit lens);
  (c) EXACT stream-pair split of c_d over data: which stream interactions
      drive the fast structural state;
  (d) rank-r causal replacement: out' = out + (c_r - c) d, dCE + coeff R^2."""

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
OUT = f'{QK}/mlp16_anatomy3.json'
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


Sinvhalf = (evecS * (1.0 / evS.clamp_min(1e-8).sqrt())) @ evecS.T

FORMS = {}
for di in range(4):
    dd = mdirs[di].contiguous()
    M, bconst = quad_form(dd)
    W = Shalf @ M.cpu().double() @ Shalf
    lamW, vecW = torch.linalg.eigh(0.5 * (W + W.T))
    order = lamW.abs().argsort(descending=True)
    U64 = (Sinvhalf @ vecW[:, order[:64]]).float().T.contiguous()   # (64, D) raw-space
    FORMS[di] = {'M': M.to(DEV), 'b': bconst, 'lam': lamW[order[:64]].float().to(DEV),
                 'U': U64.to(DEV), 'd': dd}
print('forms built', flush=True)


@torch.no_grad()
def audit_formrep(k):
    """replace all four dirs' coefficients by rank-k data-space approximations."""
    tot, n, sse, sst = 0.0, 0, 0.0, 0.0
    for i0 in range(0, len(AUDIT), 4):
        b = AUDIT[i0:i0 + 4].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
        x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        maskT = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
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
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            xh = (x * rms2).float()
            mlp_out = blk.mlp(x * rms2)
            if li == 16 and k is not None:
                delta = torch.zeros_like(mlp_out, dtype=torch.float32)
                for di, Fm in FORMS.items():
                    c = torch.einsum('btd,de,bte->bt', xh, Fm['M'].float(), xh) + Fm['b']
                    proj = torch.einsum('btd,rd->btr', xh, Fm['U'][:k])
                    ck = torch.einsum('btr,r->bt', proj ** 2, Fm['lam'][:k]) + Fm['b']
                    delta += (ck - c)[..., None] * Fm['d'][None, None]
                    if di == 0:
                        cm = c.mean()
                        sse += float(((ck - c) ** 2).sum())
                        sst += float(((c - cm) ** 2).sum())
                mlp_out = mlp_out + delta.to(mlp_out.dtype)
            x = x + mlp_out
        xf = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(xf) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
    r2 = 1 - sse / max(sst, 1e-9) if sst else None
    return tot / n, r2


base, _ = audit_formrep(None)
res = {'baseline': base, 'arms': {}}
print(f'baseline {base:.4f}', flush=True)
for k in (64, 16, 4):
    ce_k, r2 = audit_formrep(k)
    res['arms'][f'rank-{k} forms (4 dirs)'] = {'dce': round(ce_k - base, 4),
                                               'dir0_R2': round(r2, 3)}
    print(f'rank-{k}: dCE {ce_k - base:+.4f} | dir0 coeff R2 {r2:.3f}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)

# (b) stream-pair split of dir0 coefficient variance
M0 = FORMS[0]['M'].float()
P = 1024
g3 = torch.Generator(); g3.manual_seed(13)
pair_var = None
names = None
with torch.no_grad():
    idx = EST[48:52].to(DEV)
    B, T = idx.shape
    x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    streams = [x.clone()]; names = ['emb']
    v1 = None
    maskT = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    for li, blk in enumerate(m.transformer.h):
        lam0, lam1 = blk.lambdas[0], blk.lambdas[1]
        x = lam0 * x + lam1 * x0
        streams = [lam0 * sfp for sfp in streams]
        streams[0] = streams[0] + lam1 * x0
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
        streams.append(attn_out); names.append(f'attn{li}')
        if li == 16:
            rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            S_ = len(streams)
            sp = torch.stack([(sfp * rms2).float().reshape(-1, D) for sfp in streams])  # (S, N, D)
            pi = torch.randperm(sp.shape[1], generator=g3)[:P].to(DEV)
            spp = sp[:, pi]
            Ms = torch.einsum('snd,de->sne', spp, M0)
            terms = torch.einsum('sne,tne->stn', Ms, spp)          # (S, S, P)
            tc = terms - terms.mean(-1, keepdim=True)
            ctot = terms.sum((0, 1))
            cvar = (ctot - ctot.mean()).pow(2).mean()
            pair_var = (tc * (ctot - ctot.mean())[None, None]).mean(-1) / cvar.clamp_min(1e-9)
            break
        rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
        mlp_out = blk.mlp(x * rms2)
        x = x + mlp_out
        streams.append(mlp_out); names.append(f'mlp{li}')
pv = pair_var.cpu()
flat = pv.abs().view(-1)
tops = flat.topk(8).indices.tolist()
S_ = pv.shape[0]
top_pairs = [(f'{names[i_ // S_]}x{names[i_ % S_]}', round(float(pv.view(-1)[i_]), 3)) for i_ in tops]
res['dir0_stream_pair_covshare_top'] = top_pairs
print('dir0 coefficient variance, top stream-pair cov shares:', top_pairs[:5], flush=True)
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2)
print('mlp16 anatomy exp3 done', flush=True)
