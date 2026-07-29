"""Extend the ignoring-normalization result to ATTENTION PATTERNS. Claim: with per-head QK rms and
rope (rope is norm-preserving and commutes with scalars), bilin18's pattern is exactly
  pattern(i,j) = <rq1_i, rk1_j> <rq2_i, rk2_j> * HD / (||q1_i|| ||k1_j|| ||q2_i|| ||k2_j||)
where rq = rope(q_raw) etc. are computed from UN-normalized projections: a multilinear (quartic)
numerator over a product of four norm gauges (each the sqrt of a quadratic form of the residual).
If this gates at machine precision for several layers, the attention pattern joins the MLPs in the
exact tensor-network-with-scalar-gauges picture, and the per-head rms is no longer an obstruction
anywhere in the model.
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
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))

@torch.no_grad()
def check_layer(Ltest, idx):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(Ltest + 1):
        blk = m.transformer.h[li]; a = blk.attn
        x = blk.lambdas[0]*x + blk.lambdas[1]*x0; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat_model = (s1*s2).masked_fill(~mask, 0.0)
        if li == Ltest:
            # gauge form: raw (un-normalized) projections, rope applied to RAW vectors
            def raw(lin): return apply_rot(lin(hcur).view(B, T, NH, HD), cosb, sinb)
            rq, rk, rq2, rk2 = raw(a.c_q), raw(a.c_k), raw(a.c_q2), raw(a.c_k2)
            nq = a.c_q(hcur).view(B, T, NH, HD).norm(dim=-1); nk = a.c_k(hcur).view(B, T, NH, HD).norm(dim=-1)
            nq2 = a.c_q2(hcur).view(B, T, NH, HD).norm(dim=-1); nk2 = a.c_k2(hcur).view(B, T, NH, HD).norm(dim=-1)
            num = torch.einsum('bqhd,bkhd->bhqk', rq, rk) * torch.einsum('bqhd,bkhd->bhqk', rq2, rk2)
            den = (nq*nq2).permute(0, 2, 1).unsqueeze(-1) * (nk*nk2).permute(0, 2, 1).unsqueeze(-2)
            pat_gauge = (num / den.clamp_min(1e-12)).masked_fill(~mask, 0.0)   # rms's sqrt(HD) factors cancel the /HD in each score
            rel = float((pat_gauge - pat_model).norm() / pat_model.norm())
            # also verify rope commutes with the normalization (the step the claim relies on)
            rope_comm = float((apply_rot(F.rms_norm(a.c_q(hcur).view(B, T, NH, HD), (HD,)), cosb, sinb)
                               - F.rms_norm(apply_rot(a.c_q(hcur).view(B, T, NH, HD), cosb, sinb), (HD,))).abs().max())
            return rel, rope_comm
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat_model, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))

idx = COOC[:4].to(DEV)[:, :128]
res = {}
for L in (0, 2, 8, 13, 17):
    rel, rc = check_layer(L, idx)
    res[f'layer{L}'] = {'pattern_gauge_relerr': rel, 'rope_rms_commute_maxabs': rc}
    print(f"layer {L}: pattern gauge rel-err {rel:.2e} | rope/rms commutation max-abs {rc:.2e}", flush=True)
json.dump(res, open(f'{QK}/qk_pattern_gauge.json', 'w'), indent=2)
print("QK PATTERN GAUGE DONE", flush=True)
