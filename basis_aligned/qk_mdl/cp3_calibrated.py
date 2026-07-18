"""Round 3: NULL-CALIBRATED monosemanticity. Candidates scored as percentiles
against 8 matched random atoms (same type, same site) - absorbs mechanical
baselines (CP-2 confound). Metrics: fire-consistency; median per-position
participation ratio; ALIGN = effect-mass share on top-5 firing-context classes."""

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
OUT = f'{QK}/cp3_calibrated.json'
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
# H7 direction: reuse the deviation-PCA approach on H7 outputs (small sample)
o7s = []
for i in range(0, 32, 4):
    idx = EST[i:i + 4].to(DEV)
    B, T = idx.shape
    x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),))
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
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if li == 5:
            Wo = a.c_proj.weight.detach().float()
            o7s.append((y[:, :, 7].float() @ Wo[:, 7 * HD:8 * HD].T).reshape(-1, D))
            break
        x = x + a.c_proj(y.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
O7 = torch.cat(o7s)
Cv7 = (O7.T @ O7) / len(O7)
h7dir = torch.linalg.eigh(Cv7)[1][:, -1].contiguous()
print('dirs built', flush=True)

base_logits = []
for i in range(0, len(AUDIT), 4):
    b = AUDIT[i:i + 4].to(DEV)
    lg, _, _ = run(b[:, :-1])
    base_logits.append(lg.cpu())


g2 = torch.Generator(); g2.manual_seed(23)


def rand_dirs(n):
    return [F.normalize(torch.randn(D, generator=g2), dim=0).to(DEV) for _ in range(n)]


content_list = sorted(content_classes)
import random as _rnd
_rnd.seed(5)


def rand_blocks(n):
    out = []
    while len(out) < n:
        out.append((_rnd.randrange(NH), _rnd.choice(content_list), _rnd.choice(content_list)))
    return out


SITES = {
    'mlp16-direction': dict(mode='mlp16dir',
        cands=[('mlp16 dir0', mdirs[0].contiguous()), ('mlp16 dir1', mdirs[1].contiguous()),
               ('mlp16 dir3', mdirs[3].contiguous())],
        nulls=[(f'null{j}', d) for j, d in enumerate(rand_dirs(8))]),
    'h7-direction': dict(mode='h7dir',
        cands=[('H7 principal dir', h7dir.to(DEV))],
        nulls=[(f'null{j}', d) for j, d in enumerate(rand_dirs(8))]),
    'L0-block': dict(mode='block',
        cands=[('H6 c78->c161', (6, 78, 161)), ('H5 c127->c161', (5, 127, 161)),
               ('H2 c127->c148', (2, 127, 148))],
        nulls=[(f'null{j}', b) for j, b in enumerate(rand_blocks(8))]),
}


def score_atom(mode, arg):
    dvecs, ctx_toks = [], []
    for bi, i in enumerate(range(0, len(AUDIT), 4)):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        lg, aff, _ = run(idx, mode=mode, arg=arg)
        dl = (lg - base_logits[bi].to(DEV)).float()
        dvecs.append(dl[aff].cpu())
        ctx_toks.append(idx[aff].cpu())
    DL = torch.cat(dvecs)
    CT = torch.cat(ctx_toks)
    norms = DL.norm(dim=1)
    q = norms.quantile(0.9)
    sel = norms >= q
    FIRE, FT = DL[sel][:300], CT[sel][:300]
    subn = F.normalize(FIRE, dim=1)
    cons = float((subn @ subn.T).mean())
    absF = FIRE.abs()
    pr_pos = (absF.sum(1) ** 2) / (absF.shape[1] * (absF ** 2).sum(1).clamp_min(1e-12))
    pr_med = float(pr_pos.median())
    fire_classes = torch.bincount(CLS[FT], minlength=256).topk(5).indices
    mean_dl = FIRE.mean(0).abs()
    in_fire = torch.isin(CLS, fire_classes.cpu())
    align = float(mean_dl[in_fire].sum() / mean_dl.sum().clamp_min(1e-12))
    return dict(cons=cons, pr_med=pr_med, align=align,
                supp=[tok.decode([w]) for w in (-FIRE.mean(0)).topk(4).indices.tolist()])


res = {'sites': {}}
for site, spec in SITES.items():
    nulls = [score_atom(spec['mode'], arg) for _, arg in spec['nulls']]
    entries = []
    for name, arg in spec['cands']:
        sc = score_atom(spec['mode'], arg)
        pct = {}
        for k in ('cons', 'pr_med', 'align'):
            nv = sorted(n[k] for n in nulls)
            pct[k] = round(sum(1 for v in nv if v < sc[k]) / len(nv), 2)
        entries.append({'atom': name,
                        **{k: round(sc[k], 3) for k in ('cons', 'pr_med', 'align')},
                        'percentile_vs_null': pct, 'suppresses': sc['supp'],
                        'null_band': {k: [round(min(n[k] for n in nulls), 3),
                                          round(max(n[k] for n in nulls), 3)]
                                      for k in ('cons', 'pr_med', 'align')}})
        print(f"{site} | {name}: cons {sc['cons']:.2f} (pct {pct['cons']}) "
              f"align {sc['align']:.2f} (pct {pct['align']}) supp {sc['supp'][:3]}", flush=True)
    res['sites'][site] = entries
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('cp3 calibrated done', flush=True)
