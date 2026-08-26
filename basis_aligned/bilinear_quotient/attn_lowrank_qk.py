# attn_lowrank_qk: A NEW MID-PRICE STAND-IN CLASS FOR KERNEL-RESISTANT LAYERS
# (licensed by S1464: attention-pattern edges are extremely low-rank — the composed
# mlp0->pattern edge was 98% rank-8). Claim to test: each head's QK maps are
# themselves low-rank readable — replace ALL FOUR pattern maps (c_q/c_k/c_q2/c_k2)
# with PER-HEAD rank-r truncations (plain SVD of each head's [128,1152] slice;
# input-covariance whitening deliberately omitted — registered assumption) at the 7
# kernel-resistant content layers {5,8,10,13,14,16,17}, one layer at a time,
# everything else live. Values live. fid_opt vs frozen anchors.
# Price: rank-r/head = 4 maps x 9 heads x r x (128+1152) x 16b — r=32: 23.6 Mbit/layer
# (vs 85 Mbit full QK, .037 Mbit kernel). The rung between kernel and full head.
#
# Registered predictions:
#   pred_a attn5 rank-32 >= .80 fid (kernel was -.055 — the sink head's pattern is
#          carried by few directions).
#   pred_b median rank-32 fid across the 7 layers >= .85.
#   pred_c median rank-8 fid >= .60.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_lowrank_qk_results.json'
NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
LAYERS = [5, 8, 10, 13, 14, 16, 17]


def trunc_perhead(W, r):
    """W [1152, 1152] -> per-head rank-r: view [9,128,1152], SVD each slice."""
    Wf = W.float().to(DEV).view(9, 128, D)
    out = torch.zeros_like(Wf)
    for h in range(9):
        U, S, Vt = torch.linalg.svd(Wf[h], full_matrices=False)
        out[h] = (U[:, :r] * S[:r]) @ Vt[:r]
    return out.view(9 * 128, D)


@torch.no_grad()
def fwd_arm(idx, LT, TW):
    """TW: dict q/k/q2/k2 -> truncated [1152,1152] weights for layer LT; None=clean."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        if L == LT and TW is not None:
            qp = (xin.float() @ TW['q'].T).view(B, T, 9, 128)
            kp = (xin.float() @ TW['k'].T).view(B, T, 9, 128)
            q2p = (xin.float() @ TW['q2'].T).view(B, T, 9, 128)
            k2p = (xin.float() @ TW['k2'].T).view(B, T, 9, 128)
        else:
            qp = at.c_q(xin).view(B, T, 9, 128).float()
            kp = at.c_k(xin).view(B, T, 9, 128).float()
            q2p = at.c_q2(xin).view(B, T, 9, 128).float()
            k2p = at.c_k2(xin).view(B, T, 9, 128).float()
        cos, sin = at.rotary(qp)
        q = are(F.rms_norm(qp, (128,)), cos, sin)
        k = are(F.rms_norm(kp, (128,)), cos, sin)
        q2 = are(F.rms_norm(q2p, (128,)), cos, sin)
        k2 = are(F.rms_norm(k2p, (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / 128.0)
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()

    def ce_run(LT, TW):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, LT, TW).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    clean = ce_run(0, None)
    sw = json.load(open(PT + 'optimal_ablation_all_results.json'))['results']
    res = {'clean': round(clean, 4)}
    fids = {}
    for LT in LAYERS:
        at = H[LT].attn
        for r in (8, 32):
            TW = {'q': trunc_perhead(at.c_q.weight, r),
                  'k': trunc_perhead(at.c_k.weight, r),
                  'q2': trunc_perhead(at.c_q2.weight, r),
                  'k2': trunc_perhead(at.c_k2.weight, r)}
            ce_ = ce_run(LT, TW)
            key = f'attn{LT}_r{r}'
            res[key] = round(ce_, 4)
            a = sw[f'attn{LT}']
            fids[key] = round((a['ce_opt'] - ce_) / max(a['ce_opt'] - clean, 1e-6), 4)
            print(f"{key}: ce {ce_:.4f} fid {fids[key]:.4f}", flush=True)
            json.dump({'partial': True, 'res': res, 'fids': fids},
                      open(OUT, 'w'), indent=1)

    import statistics
    med32 = statistics.median([fids[f'attn{L}_r32'] for L in LAYERS])
    med8 = statistics.median([fids[f'attn{L}_r8'] for L in LAYERS])
    pa = fids['attn5_r32'] >= 0.80
    pb = med32 >= 0.85
    pc = med8 >= 0.60
    out = {'ce': res, 'fid_opt': fids, 'median_r32': round(med32, 4),
           'median_r8': round(med8, 4),
           'mbits_per_layer': {'r8': 5.9, 'r32': 23.6, 'full_qk': 85.0,
                               'kernel': 0.037},
           'pred_a_attn5_r32_80': bool(pa), 'pred_b_med32_85': bool(pb),
           'pred_c_med8_60': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"med32 {med32:.4f} med8 {med8:.4f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
