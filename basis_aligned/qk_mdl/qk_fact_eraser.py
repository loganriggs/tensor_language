"""TICK 245: the single-fact eraser.

Tick 244 showed enrichment deltas are per-instance nearly-orthogonal. Prediction:
subtracting a fact's OWN delta at its key position in the CLEAN run erases that
fact's prediction with near-zero collateral damage elsewhere in the document.

For 48 random strong-key positions: erase = resid_add(-delta) at layer-8 entry at
the key position; measure (a) targeted drop in the true next token's log-prob at
the query position, (b) collateral = mean |change in log-prob of the actual next
token| over all other positions >= 32 in the document, (c) both again with a
norm-matched random vector (control), (d) cross-fact: if the doc has a second
strong-key position, the change of ITS target under the first fact's erasure.
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
L_ERASE = 8


@torch.no_grad()
def run_p(idx, patch=None):
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
sel = [(POS[i][0], POS[i][1], KEYJ[i]) for i in range(len(POS)) if KEYJ[i] is not None]
# group by doc for the cross-fact condition
bydoc = {}
for (di, p, j) in sel:
    bydoc.setdefault(di, []).append((p, j))
print(f'{len(sel)} strong-key positions in {len(bydoc)} docs', flush=True)

rg = torch.Generator().manual_seed(11)
agg = {'target_drop': [], 'collateral': [], 'rand_target_drop': [], 'rand_collateral': [],
       'crossfact_change': []}
done = 0
for (di, p, j) in sel:
    if done >= 48:
        break
    done += 1
    idx_c = SEQS[di:di + 1, :-1].clone().to(DEV)
    nxt = SEQS[di, 1:].to(DEV)                     # true next token at each position
    Tm = idx_c.shape[1]
    lg_c, cc = run_p(idx_c)
    lp_all_c = F.log_softmax(lg_c[0].float(), -1)[torch.arange(Tm, device=DEV), nxt]
    # delta from neutral substitution
    idx_x = idx_c.clone()
    idx_x[0, j] = NEUTRAL
    _, cx = run_p(idx_x)
    delta = (cc['resid'][L_ERASE][0, j] - cx['resid'][L_ERASE][0, j]).float()
    rand = torch.randn(D, generator=rg).to(DEV)
    rand = rand / rand.norm() * delta.norm()
    others = [q for q in range(32, Tm - 1) if q != p]
    o_idx = torch.tensor(others, device=DEV)
    for vec, tkey, ckey in ((-delta, 'target_drop', 'collateral'),
                            (-rand, 'rand_target_drop', 'rand_collateral')):
        lg_e, _ = run_p(idx_c, patch=('resid_add', L_ERASE, [j], vec))
        lp_all_e = F.log_softmax(lg_e[0].float(), -1)[torch.arange(Tm, device=DEV), nxt]
        agg[tkey].append(float(lp_all_c[p] - lp_all_e[p]))
        agg[ckey].append(float((lp_all_c[o_idx] - lp_all_e[o_idx]).abs().mean()))
        if tkey == 'target_drop':
            for (p2, j2) in bydoc[di]:
                if p2 != p and j2 != j:
                    agg['crossfact_change'].append(float((lp_all_c[p2] - lp_all_e[p2]).abs()))
out = {k: {'mean': round(float(np.mean(v)), 3), 'median': round(float(np.median(v)), 3),
           'n': len(v)} for k, v in agg.items() if v}
print(json.dumps(out, indent=1), flush=True)
json.dump(out, open(f'{QK}/qk_fact_eraser.json', 'w'), indent=2)
print('FACT ERASER DONE', flush=True)
