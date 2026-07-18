"""Monosemanticity metric, round 2: POSITIVE CONTROLS + refined metrics.
Round 1 falsified all energy-top layer-0 class-pair blocks (conc~0, cons~0.1)
but the selection was junk-class-dominated and the metrics possibly too harsh.
Score KNOWN-GOOD atoms with refined metrics; if they also fail, the metric is
broken; if they pass, the layer-0 falsification stands.

Atoms scored (ablate atom -> effect vector over affected positions):
  - L5.H7 rank-1 direction (zero its output projection onto the dir)
  - L5.H5 head (zero scores)          [known: noisy identity carrier]
  - mlp16 deviation dirs 0 and 3      [named: legal-cite / markup structure]
  - random-direction control at mlp16 (should be diffuse/inconsistent)
  - 3 CONTENT class-pair blocks at L0 (junk classes excluded by frequency)
Refined metrics per atom:
  - PR: participation ratio of mean effect = (S|d|)^2 / (V * S d^2)  (low=concentrated)
  - class-mass: share of |effect| mass in the top output CLASS (emb kmeans-256)
  - fire-consistency: mean pairwise cos of per-position effects, top-decile
    |effect| positions only (what it does WHEN it acts)
  - top promoted/suppressed tokens"""
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
OUT = f'{QK}/cp2_controls.json'
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

ATOMS = [('L5.H7 rank-1 direction', 'h7dir', h7dir),
         ('L5.H5 head', 'head', (5, 5)),
         ('mlp16 dir0 (legal-cite structure)', 'mlp16dir', mdirs[0].contiguous()),
         ('mlp16 dir3 (markup structure)', 'mlp16dir', mdirs[3].contiguous()),
         ('mlp16 RANDOM direction (control)', 'mlp16rand', rnd)]
for (hh, cq, ck) in [(6, 78, 161), (5, 127, 161), (2, 127, 148)]:   # content-class blocks (freq-filtered)
    ATOMS.append((f'L0 block H{hh} c{cq}->c{ck}', 'block', (hh, cq, ck)))

res = {'atoms': []}
CLS_DEV = CLS.to(DEV)
for name, mode, arg in ATOMS:
    dvecs = []
    for bi, i in enumerate(range(0, len(AUDIT), 4)):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        lg, aff, _ = run(idx, mode=mode, arg=arg)
        dl = (lg - base_logits[bi].to(DEV)).float()
        dvecs.append(dl[aff].cpu())
    DL = torch.cat(dvecs)
    norms = DL.norm(dim=1)
    thresh = norms.quantile(0.9)
    FIRE = DL[norms >= thresh][:300]
    mean_dl = FIRE.mean(0)
    absd = mean_dl.abs()
    pr = float((absd.sum() ** 2) / (len(absd) * (absd ** 2).sum().clamp_min(1e-12)))
    cmass = torch.zeros(256)
    cmass.index_add_(0, CLS, absd)
    top_class = int(cmass.argmax())
    class_share = float(cmass.max() / cmass.sum().clamp_min(1e-12))
    subn = F.normalize(FIRE, dim=1)
    cons = float((subn @ subn.T).mean())
    entry = {'atom': name,
             'participation_ratio': round(pr, 4),
             'top_class_share': round(class_share, 3),
             'top_class_exemplars': [tok.decode([w]) for w in
                                     (CLS == top_class).nonzero().squeeze(1)[:5].tolist()],
             'fire_consistency': round(cons, 3),
             'promotes_on_ablate': [tok.decode([w]) for w in mean_dl.topk(5).indices.tolist()],
             'suppresses_on_ablate': [tok.decode([w]) for w in (-mean_dl).topk(5).indices.tolist()]}
    res['atoms'].append(entry)
    print(f"{name}: PR {pr:.4f} classshare {class_share:.2f} cons {cons:.2f} "
          f"suppresses {entry['suppresses_on_ablate'][:3]}", flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('cp2 controls done', flush=True)
