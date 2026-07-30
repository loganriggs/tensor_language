"""Causal test of the demoted MLP1 routing claim ('attn0 feeds MLP1 through MLP0, not directly').
At MLP1's interface, compare the composed substitution with ALL streams vs with the A0 stream
ZEROED in MLP1's input assembly (m0 untouched -- so only the DIRECT a0 path is removed) vs with the
M0 stream zeroed (the indirect path removed). First use of the SE harness: paired per-token dCE
differences with standard errors on the held-back audit slice FINEWEB[400:600] (never used for any
selection in the composition arc; rows 400-447 were used once for induction heldout -- noted).
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
HELD = FW[448:600]   # held-back slice (never selected on; 400-447 used once for induction heldout)
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
B0, B1 = m.transformer.h[0], m.transformer.h[1]
L1w = B1.mlp.Left.weight.detach().float(); R1w = B1.mlp.Right.weight.detach().float()
D1w = B1.mlp.Down.weight.detach().float(); b1w = B1.mlp.Down_bias.detach().float()
def T1(u, v):
    return 0.5*(((u @ L1w.T) * (v @ R1w.T)) @ D1w.T + ((v @ L1w.T) * (u @ R1w.T)) @ D1w.T)
lam00 = (B0.lambdas[0]+B0.lambdas[1]).item(); l10, l11 = B1.lambdas[0].item(), B1.lambdas[1].item()


@torch.no_grad()
def forward(idx, mode):
    """mode: None real | 'all' composed all streams | 'noA0' direct a0 zeroed | 'noM0' m0 zeroed."""
    B, T2 = idx.shape
    x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    a0c = m0c = None
    for li in range(NL):
        blk = m.transformer.h[li]; a = blk.attn
        x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T2, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        aout = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T2, -1))
        x = x + aout; hin = F.rms_norm(x, (D,))
        if li == 0:
            a0c = aout.reshape(-1, D); mo = blk.mlp(hin); m0c = mo.reshape(-1, D); x = x + mo; continue
        if li == 1 and mode is not None:
            e = x0.reshape(-1, D); a1f = aout.reshape(-1, D)
            ca0 = 0.0 if mode == 'noA0' else 1.0
            cm0 = 0.0 if mode == 'noM0' else 1.0
            xp = (l10*lam00 + l11)*e + l10*ca0*a0c + l10*cm0*m0c + a1f
            r = xp.pow(2).sum(1)/D
            mo = (T1(xp, xp)/r.unsqueeze(1) + b1w).view(B, T2, D)
            x = x + mo.to(x.dtype); continue
        x = x + blk.mlp(hin)
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

@torch.no_grad()
def percetok(mode):
    ces = []
    for i in range(0, len(HELD), 4):
        b = HELD[i:i+4].to(DEV)
        lg = forward(b[:, :-1], mode)
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1), reduction='none')
        ces.append(ce.cpu())
    return torch.cat(ces)

ce_real = percetok(None)
res = {'n_tokens': int(ce_real.numel()), 'held_slice': '448:600'}
for mode in ('all', 'noA0', 'noM0'):
    ce = percetok(mode)
    dif = (ce - ce_real)
    mean = float(dif.mean()); se = float(dif.std()/np.sqrt(dif.numel()))
    res[mode] = {'dCE': round(mean, 5), 'SE': round(se, 6)}
    print(f"{mode}: dCE +{mean:.5f} (SE {se:.6f})", flush=True)
d_all, d_noA0, d_noM0 = res['all']['dCE'], res['noA0']['dCE'], res['noM0']['dCE']
print(f"\ndirect-a0 path cost: +{d_noA0-d_all:.5f} | indirect (m0) path cost: +{d_noM0-d_all:.5f}", flush=True)
res['direct_a0_cost'] = round(d_noA0-d_all, 5); res['indirect_m0_cost'] = round(d_noM0-d_all, 5)
json.dump(res, open(f'{QK}/qk_mlp1_block_ablate.json', 'w'), indent=2)
print("QK MLP1 BLOCK ABLATE DONE", flush=True)
