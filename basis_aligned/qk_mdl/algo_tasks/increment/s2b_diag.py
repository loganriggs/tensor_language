"""Step 2b diagnostics for the top heads (L8H7, L8H3):
  - attention pattern (unnormalized bilinear weights) at the final query position
  - lamb (block-0 v mixing) at layer 8
  - factor patching within layer-8 attention: patch clean PATTERN only vs clean
    V only (per head) into the corrupted run -> is the effect QK- or OV-routed?
"""
import json, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/increment')
from common import get_model, OUT
from tier2_model import rope_tables, apply_rot

torch.manual_seed(0)
m, cfg = get_model()
NH, D, NL = cfg['n_head'], cfg['n_embd'], cfg['n_layer']
HD = D // NH
S = torch.load(f'{OUT}/stimuli.pt')
NA = 30
clean, corr = S['clean'][:NA].cuda(), S['corr'][:NA].cuda()
ca, xa = S['clean_ans'][:NA].cuda(), S['corr_ans'][:NA].cuda()


def forward_factor(idx, swap=None, cache=None):
    """swap: dict with optional keys ('pat', li, h)-> pat_clean [B,q,k],
    ('v', li, h) -> v_clean [B,T,HD] (post-lamb mix). cache collects those."""
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
        v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if cache is not None:
            for hh in range(NH):
                cache[('pat', li, hh)] = pat[:, hh].detach().clone()
                cache[('v', li, hh)] = v[:, :, hh].detach().clone()
        if swap is not None:
            pat = pat.clone(); v = v.clone()
            for hh in range(NH):
                if ('pat', li, hh) in swap: pat[:, hh] = swap[('pat', li, hh)]
                if ('v', li, hh) in swap: v[:, :, hh] = swap[('v', li, hh)]
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    x = F.rms_norm(x, (D,))
    return 30 * torch.tanh(m.lm_head(x) / 30)


def margins(lg):
    n = torch.arange(lg.shape[0], device='cuda')
    return (lg[n, ca[:lg.shape[0]]] - lg[n, xa[:lg.shape[0]]]).cpu()


res = {'lamb_per_layer': [float(blk.attn.lamb) for blk in m.transformer.h]}
print('attn lamb (block-0 v weight) per layer:', [round(v, 3) for v in res['lamb_per_layer']])

with torch.no_grad():
    Mc, Mx, patrows = [], [], {(8, 7): [], (8, 3): []}
    caches = []
    for i in range(0, NA, 8):
        cch = {}
        lgc = forward_factor(clean[i:i+8], cache=cch)[:, -1].float()
        caches.append((i, cch))
        n = torch.arange(lgc.shape[0], device='cuda')
        Mc.append(lgc[n, ca[i:i+8]] - lgc[n, xa[i:i+8]])
        lgx = forward_factor(corr[i:i+8])[:, -1].float()
        Mx.append(lgx[n, ca[i:i+8]] - lgx[n, xa[i:i+8]])
        for (li, hh) in patrows:
            patrows[(li, hh)].append(cch[('pat', li, hh)][:, 7].cpu())  # final query row
    M_clean = torch.cat(Mc).cpu(); M_corr = torch.cat(Mx).cpu()
    GAP = M_clean - M_corr
    for key in patrows:
        rows = torch.cat(patrows[key], 0)  # [NA, 8]
        res[f'pattern_final_query_L{key[0]}H{key[1]}'] = rows.mean(0).tolist()
        print(f'L{key[0]}H{key[1]} pattern at final query (mean over pairs, key pos 0..7):',
              [round(v, 3) for v in rows.mean(0).tolist()])

    # factor patching
    def run_swap(kind, targets):
        Mp = []
        for i, cch in caches:
            swap = {(kind, li, hh): cch[(kind, li, hh)] for (li, hh) in targets}
            lg = forward_factor(corr[i:i+8], swap=swap)[:, -1].float()
            n = torch.arange(lg.shape[0], device='cuda')
            Mp.append((lg[n, ca[i:i+8]] - lg[n, xa[i:i+8]]).cpu())
        Mp = torch.cat(Mp)
        return ((Mp - M_corr) / GAP).mean().item()

    for targets, name in [([(8, 7)], 'L8H7'), ([(8, 3)], 'L8H3'), ([(8, 7), (8, 3)], 'both')]:
        r_pat = run_swap('pat', targets)
        r_v = run_swap('v', targets)
        res[f'factor_{name}'] = {'pattern_only': round(r_pat, 4), 'v_only': round(r_v, 4)}
        print(f'{name}: clean PATTERN only -> rf {r_pat:.4f};  clean V only -> rf {r_v:.4f}')

json.dump(res, open(f'{OUT}/s2b_diag.json', 'w'), indent=2)
print('saved s2b_diag.json')
