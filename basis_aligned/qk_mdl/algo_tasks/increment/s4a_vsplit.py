"""Step 4a: which matrix carries the value payload at heads (8,7),(8,3)?
v_8 = (1-lamb8)*c_v8(h8) + lamb8*v1, lamb8=4.0 -> v8 = -3*c_v8(h8) + 4*c_v0(h0).
Patch clean->corrupted each TERM separately (at heads 7&3 of layer 8 only).
"""
import json, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/increment')
from common import get_model, OUT
from tier2_model import rope_tables, apply_rot

torch.manual_seed(0)
m, cfg = get_model()
NH, D = cfg['n_head'], cfg['n_embd']; HD = D // NH
S = torch.load(f'{OUT}/stimuli.pt')
NA = 30
clean, corr = S['clean'][:NA].cuda(), S['corr'][:NA].cuda()
ca, xa = S['clean_ans'][:NA].cuda(), S['corr_ans'][:NA].cuda()
TARGET = [(8, 7), (8, 3)]


def fwd(idx, swap=None, cache=None):
    """swap: {('vown',li,h): t, ('v1',li,h): t} replace that TERM (pre-lamb-mix)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, idx.device, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    for li, blk in enumerate(m.transformer.h):
        a = blk.attn
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        h = F.rms_norm(x, (D,))
        def qk(lin):
            z = F.rms_norm(lin(h).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)
        vown = a.c_v(h).view(B, T, NH, HD)
        if v1 is None: v1 = vown
        v1u = v1.view_as(vown).clone(); vownu = vown.clone()
        if cache is not None:
            for hh in range(NH):
                cache[('vown', li, hh)] = vownu[:, :, hh].detach().clone()
                cache[('v1', li, hh)] = v1u[:, :, hh].detach().clone()
        if swap is not None:
            for hh in range(NH):
                if ('vown', li, hh) in swap: vownu[:, :, hh] = swap[('vown', li, hh)]
                if ('v1', li, hh) in swap: v1u[:, :, hh] = swap[('v1', li, hh)]
        v = (1 - a.lamb) * vownu + a.lamb * v1u
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


with torch.no_grad():
    caches, Mc, Mx = [], [], []
    for i in range(0, NA, 8):
        cch = {}
        lgc = fwd(clean[i:i+8], cache=cch)[:, -1].float(); caches.append((i, cch))
        lgx = fwd(corr[i:i+8])[:, -1].float()
        n = torch.arange(lgc.shape[0], device='cuda')
        Mc.append((lgc[n, ca[i:i+8]] - lgc[n, xa[i:i+8]]).cpu())
        Mx.append((lgx[n, ca[i:i+8]] - lgx[n, xa[i:i+8]]).cpu())
    M_clean, M_corr = torch.cat(Mc), torch.cat(Mx); GAP = M_clean - M_corr

    res = {}
    for kind in ['vown', 'v1', 'both']:
        Mp = []
        for i, cch in caches:
            kinds = ['vown', 'v1'] if kind == 'both' else [kind]
            swap = {(kd, li, hh): cch[(kd, li, hh)] for kd in kinds for (li, hh) in TARGET}
            lg = fwd(corr[i:i+8], swap=swap)[:, -1].float()
            n = torch.arange(lg.shape[0], device='cuda')
            Mp.append((lg[n, ca[i:i+8]] - lg[n, xa[i:i+8]]).cpu())
        rf = ((torch.cat(Mp) - M_corr) / GAP).mean().item()
        res[kind] = round(rf, 4)
        print(f"clean {kind} term at L8 H7+H3 -> rf {rf:.4f}", flush=True)

json.dump(res, open(f'{OUT}/s4a_vsplit.json', 'w'), indent=2)
print('saved s4a_vsplit.json')
