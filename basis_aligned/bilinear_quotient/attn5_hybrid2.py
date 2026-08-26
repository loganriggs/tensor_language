# attn5_hybrid2: BEST-PIECES HYBRID FOR THE TWO HARD LAYERS (board: attn5 .055
# unexplained is #3; attn8 also lags). S1469's whitened low-rank class works for
# generic heads but not sinks; S1448's roster-live works for sinks but leaves the
# other heads on crude kernels. Combine: roster heads LIVE (full QK), non-roster
# heads on whitened rank-32 truncation. Rosters: a5={7} (sink), a8={1,2,3,7}.
# Price: a5 = 8/9 x 23.6 + 1 head full (9.4) ~ 30 Mbit; a8 = 5/9 x 23.6 + 4 x 9.4
# ~ 50.7 Mbit.
# Original header: INPUT-WHITENED per-head truncation (S1467: plain SVD went 0-for-3
# — attn5 r32 = -1.61, WORSE than the kernel, and attn8 r8 = -2.49. The registered
# assumption 'whitening omitted' is the prime suspect: plain SVD ranks directions by
# weight norm, but the stream covariance is far from isotropic, so the kept directions
# can miss the high-variance inputs the sink head reads. Fix: per-layer stream
# covariance Sigma of xin (96 rows), whiten W' = W_head @ Sigma^{1/2}, truncate, map
# back. Same price, same layers, same bars.)
# Original header: A NEW MID-PRICE STAND-IN CLASS FOR KERNEL-RESISTANT LAYERS
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
#   pred_a attn5 hybrid >= .75 fid (parents: .597 kernel+live7, .531 whitened-r32).
#   pred_b attn8 hybrid >= .85 (parents: .794 kernel+roster, .675 whitened-r32).
#   pred_c both hybrids beat BOTH their parents.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn5_hybrid2_results.json'
NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
LAYERS = [5, 8]
ROSTER = {5: {7}, 8: {1, 2, 3, 7}}


def trunc_perhead(W, r, Wh, Whi):
    """Whitened per-head rank-r: SVD of W_head @ Wh, then map back with Whi."""
    Wf = W.float().to(DEV).view(9, 128, D)
    out = torch.zeros_like(Wf)
    for h in range(9):
        U, S, Vt = torch.linalg.svd(Wf[h] @ Wh, full_matrices=False)
        out[h] = ((U[:, :r] * S[:r]) @ Vt[:r]) @ Whi
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
            qf = at.c_q(xin).view(B, T, 9, 128).float()
            kf = at.c_k(xin).view(B, T, 9, 128).float()
            q2f = at.c_q2(xin).view(B, T, 9, 128).float()
            k2f = at.c_k2(xin).view(B, T, 9, 128).float()
            for hh in ROSTER[LT]:
                qp[:, :, hh] = qf[:, :, hh]; kp[:, :, hh] = kf[:, :, hh]
                q2p[:, :, hh] = q2f[:, :, hh]; k2p[:, :, hh] = k2f[:, :, hh]
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

    # per-layer xin covariance whiteners (96 rows)
    CR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    XACC = {L: torch.zeros(D, D, device=DEV) for L in LAYERS}
    ncov = 0
    for i in range(0, 96, 8):
        idx = CR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1_ = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            if L in XACC:
                Xf = xin.float().reshape(-1, D)
                XACC[L] += Xf.T @ Xf
            x, v1_ = blk(x, v1_, x0)
        ncov += idx.shape[0] * T
    WHITEN = {}
    for L in LAYERS:
        Sg = XACC[L] / ncov
        ev, V = torch.linalg.eigh(Sg)
        ev = ev.clamp_min(1e-6)
        WHITEN[L] = (V @ torch.diag(ev.sqrt()) @ V.T,
                     V @ torch.diag(ev.rsqrt()) @ V.T)
    print("whiteners built", flush=True)

    clean = ce_run(0, None)
    sw = json.load(open(PT + 'optimal_ablation_all_results.json'))['results']
    res = {'clean': round(clean, 4)}
    fids = {}
    for LT in LAYERS:
        at = H[LT].attn
        Wh, Whi = WHITEN[LT]
        for r in (32,):
            TW = {'q': trunc_perhead(at.c_q.weight, r, Wh, Whi),
                  'k': trunc_perhead(at.c_k.weight, r, Wh, Whi),
                  'q2': trunc_perhead(at.c_q2.weight, r, Wh, Whi),
                  'k2': trunc_perhead(at.c_k2.weight, r, Wh, Whi)}
            ce_ = ce_run(LT, TW)
            key = f'attn{LT}_r{r}'
            res[key] = round(ce_, 4)
            a = sw[f'attn{LT}']
            fids[key] = round((a['ce_opt'] - ce_) / max(a['ce_opt'] - clean, 1e-6), 4)
            print(f"{key}: ce {ce_:.4f} fid {fids[key]:.4f}", flush=True)
            json.dump({'partial': True, 'res': res, 'fids': fids},
                      open(OUT, 'w'), indent=1)

    PARENTS = {5: 0.597, 8: 0.794}
    LOWR = {5: 0.531, 8: 0.675}
    pa = fids['attn5_r32'] >= 0.75
    pb = fids['attn8_r32'] >= 0.85
    pc = all(fids[f'attn{L}_r32'] > max(PARENTS[L], LOWR[L]) for L in LAYERS)
    out = {'ce': res, 'fid_opt': fids,
           'parents': {'kernel_roster': PARENTS, 'whitened_r32': LOWR},
           'mbits': {'a5_hybrid': 30.4, 'a8_hybrid': 50.7},
           'pred_a_a5_75': bool(pa), 'pred_b_a8_85': bool(pb),
           'pred_c_beats_parents': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"a5 {fids['attn5_r32']} a8 {fids['attn8_r32']}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
