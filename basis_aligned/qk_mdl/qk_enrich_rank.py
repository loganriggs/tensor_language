"""TICK 244: causal dimensionality of the enrichment signal.

The enrichment delta = clean-minus-corrupt residual at the key position at layer-8
entry restores a median 0.98 (tick 243). Question (L4 rule: sparser representation
for every part): do these deltas live in a SHARED low-dimensional subspace?

Method: collect deltas at 128 random strong-key positions; fit PCA on the last 80
(fit set); on the first 48 (held-out eval set) restore corrupt + P_r(delta) at
layer 8 for rank r in {1,2,4,8,16,32,64,80} plus full delta and a random-basis
control at r=16. Recovery-versus-rank is the causal dimensionality curve.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, reference_forward
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
tok = AutoTokenizer.from_pretrained('gpt2')
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQS = FINEWEB[:128]
NEUTRAL = tok.encode(' one')[0]
RESTORE_L = 8


@torch.no_grad()
def run_p(idx, patch=None):
    """patch = ('resid_add', L, positions, vec): at entry of layer L, positions get
    the CURRENT (corrupt-run) residual plus vec. patch = ('resid', L, positions,
    cache) as before."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    C = {'resid': {}}
    for li, blk in enumerate(m.transformer.h):
        C['resid'][li] = x.clone()
        if patch and patch[0] == 'resid' and li == patch[1]:
            x = x.clone()
            x[:, patch[2]] = patch[3]['resid'][li][:, patch[2]]
        if patch and patch[0] == 'resid_add' and li == patch[1]:
            x = x.clone()
            x[:, patch[2]] = x[:, patch[2]] + patch[3].to(x.dtype)
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def qk(lin):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30), C


# ---- random strong-key positions (same generator as tick 243) ----
g = torch.Generator().manual_seed(7)
POS = []
T = SEQS.shape[1]
for _ in range(160):
    di = int(torch.randint(0, 128, (1,), generator=g))
    p = int(torch.randint(32, T - 2, (1,), generator=g))
    POS.append((di, p))
KEYJ = []
for (di, p) in POS:
    tgt = SEQS[di, p + 1]
    idx = SEQS[di:di + 1, :-1]
    offs = list(range(0, min(16, p + 1)))
    variants = [idx.clone()]
    for off in offs:
        v = idx.clone()
        v[0, p - off] = NEUTRAL
        variants.append(v)
    with torch.no_grad():
        lg = reference_forward(m, torch.cat(variants, 0).to(DEV), 'bf16')[:, p].float()
        lps = F.log_softmax(lg, 1)[:, tgt]
    drops = [(off, float(lps[0] - lps[1 + i])) for i, off in enumerate(offs)]
    drops.sort(key=lambda x: -x[1])
    KEYJ.append(p - drops[0][0] if drops[0][1] > 1.0 else None)
sel = [(POS[i][0], POS[i][1], KEYJ[i]) for i in range(len(POS)) if KEYJ[i] is not None][:128]
print(f'{len(sel)} strong-key positions', flush=True)

# ---- phase 1: collect enrichment deltas at layer-8 entry ----
recs = []   # (di, p, j, tgt, lp_c, lp_x, delta)
for (di, p, j) in sel:
    tgt = int(SEQS[di, p + 1])
    idx_c = SEQS[di:di + 1, :-1].clone().to(DEV)
    idx_x = idx_c.clone()
    idx_x[0, j] = NEUTRAL
    lg_c, cc = run_p(idx_c)
    lg_x, cx = run_p(idx_x)
    lp_c = float(F.log_softmax(lg_c[0, p].float(), 0)[tgt])
    lp_x = float(F.log_softmax(lg_x[0, p].float(), 0)[tgt])
    if lp_c - lp_x < 1.0:
        continue
    delta = (cc['resid'][RESTORE_L][0, j] - cx['resid'][RESTORE_L][0, j]).float().cpu()
    recs.append((di, p, j, tgt, lp_c, lp_x, delta))
print(f'{len(recs)} usable deltas', flush=True)

N_EVAL = 48
fit = torch.stack([r[6] for r in recs[N_EVAL:]])       # fit set
evalr = recs[:N_EVAL]
fitc = fit - fit.mean(0, keepdim=True)
U, S, Vt = torch.linalg.svd(fitc, full_matrices=False)
basis = Vt                                              # (rank, D) rows = PCs
var = (S ** 2 / (S ** 2).sum()).cumsum(0)
print('cum var by rank:', [round(float(var[r - 1]), 3) for r in (1, 2, 4, 8, 16, 32, 64)],
      flush=True)
rand_basis = torch.linalg.qr(torch.randn(D, 16, generator=torch.Generator().manual_seed(1)))[0].T

RANKS = [1, 2, 4, 8, 16, 32, 64, 80]
agg = {f'r{r}': [] for r in RANKS}
agg['full'] = []
agg['rand16'] = []
for (di, p, j, tgt, lp_c, lp_x, delta) in evalr:
    idx_x = SEQS[di:di + 1, :-1].clone().to(DEV)
    idx_x[0, j] = NEUTRAL
    den = lp_c - lp_x

    def rec_of(vec):
        lg2, _ = run_p(idx_x, patch=('resid_add', RESTORE_L, [j], vec.to(DEV)))
        return (float(F.log_softmax(lg2[0, p].float(), 0)[tgt]) - lp_x) / den

    for r in RANKS:
        B_ = basis[:r]
        agg[f'r{r}'].append(rec_of(delta @ B_.T @ B_))
    agg['full'].append(rec_of(delta))
    agg['rand16'].append(rec_of(delta @ rand_basis.T @ rand_basis))
out = {k: {'mean': round(float(np.mean(v)), 3), 'median': round(float(np.median(v)), 3),
           'n': len(v)} for k, v in agg.items() if v}
out['cumvar'] = {str(r): round(float(var[r - 1]), 3) for r in RANKS if r <= len(S)}
print(json.dumps(out, indent=1), flush=True)
json.dump(out, open(f'{QK}/qk_enrich_rank.json', 'w'), indent=2)
print('ENRICH RANK DONE', flush=True)
