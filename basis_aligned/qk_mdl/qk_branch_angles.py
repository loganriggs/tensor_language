"""REPRESENTATION/FUNCTION diagnostic (all 18 layers x 9 heads, one shot): bilin18's pattern is a
PRODUCT of two score branches, sc1=(q1.k1)/HD and sc2=(q2.k2)/HD. Per head, are the two branches
carrying DISTINCT structure (genuine two-factor bilinear selection) or is one branch effectively a copy
of the other (the head just squares a single branch)? Measure, on real held-back text, the Pearson
correlation between sc1 and sc2 over the causal (query,key) entries per head. corr~1 => redundant single
branch; corr~0 => the two branches select on different structure (the product matters). Also report the
fraction of pattern variance the sign-agreement carries. Cheap; informs whether the two-branch design is
per-head redundant. No training; audits on FW[448:600].
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600, :128]

@torch.no_grad()
def branch_corr():
    B0 = 8; accs = {(li, h): [] for li in range(NL) for h in range(NH)}
    for s in range(0, 48, B0):
        idx = HELD[s:s+B0].to(DEV); B, T = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        for li in range(NL):
            blk = m.transformer.h[li]; a = blk.attn
            x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
            hcur = F.rms_norm(x, (D,))
            def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            sc1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD
            sc2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            mb = mask.expand(B, NH, T, T)
            for h in range(NH):
                a1 = sc1[:, h][mask.expand(B, T, T)]; a2 = sc2[:, h][mask.expand(B, T, T)]
                accs[(li, h)].append(torch.stack([a1, a2], 1).cpu())
            # advance the real forward
            pat = (sc1*sc2).masked_fill(~mask, 0.0)
            yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            x = x + a.c_proj(yh.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    out = {}
    for (li, h), chunks in accs.items():
        z = torch.cat(chunks, 0).double()             # (n, 2): [sc1, sc2]
        c = torch.corrcoef(z.T)[0, 1].item()
        out[f'L{li}H{h}'] = round(float(c), 3)
    return out

corr = branch_corr()
vals = np.array(list(corr.values()))
res = {'per_head_branch_corr': corr,
       'summary': {'median': round(float(np.median(vals)), 3), 'mean': round(float(np.mean(vals)), 3),
                   'frac_redundant_gt0.9': round(float(np.mean(vals > 0.9)), 3),
                   'frac_distinct_lt0.5': round(float(np.mean(vals < 0.5)), 3),
                   'n_heads': int(vals.size)}}
print("BRANCH CORR summary:", json.dumps(res['summary'], indent=2), flush=True)
# most-distinct and most-redundant heads
order = sorted(corr.items(), key=lambda kv: kv[1])
print("most-distinct (corr low):", order[:6], flush=True)
print("most-redundant (corr high):", order[-6:], flush=True)
json.dump(res, open(f'{QK}/qk_branch_angles.json', 'w'), indent=2)
print("QK BRANCH ANGLES DONE", flush=True)
