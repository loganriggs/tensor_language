# attn_composite5: REFIT THE KERNELS *IN THE COMPOSITE CONTEXT* (S1454 registered
# next step: the composite ladder stalled at 4.39 and the kernel core — fit on the
# CLEAN stream — is the bottleneck; S1452's calibration lesson applied one level up:
# a kernel is only optimal against the context it was averaged in).
# Fixed-point iteration: v0 = clean-stream kernels; v_{i+1} = per-layer mean patterns
# BY OFFSET measured while the model runs UNDER kernel-all(v_i) (each layer's heads
# read the v_i-corrupted stream; their would-be patterns are averaged before being
# replaced). Same 37 Kbit/layer price. Arms (NR=960, mask >= 64):
#   kernel_all_v0 / kernel_all_v1 / kernel_all_v2 — the iteration trace.
#   best_roster2_v1 — v1 kernels + live roster at {10,13,14,16,17}, a10={2,3,4,5,6}.
#
# Registered predictions:
#   pred_a kernel_all_v1 <= 4.60 (refit gains >= .13 over v0's 4.7302).
#   pred_b best_roster2_v1 <= 4.30 (v4's best with v0 kernels was 4.4125).
#   pred_c v2's gain over v1 <= half of v1's gain over v0 (the iteration converges).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_composite5_results.json'
NMEAN = 24; NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
SPEC = {10: {2, 3, 4, 5, 6}, 13: {0, 5, 8}, 14: {4, 6, 7},
        16: {0, 3, 4, 5}, 17: {0, 1, 2}}
LIVE_BEST = frozenset(SPEC)


@torch.no_grad()
def block_pat(at, xin, B):
    cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
    q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
    k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
    q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
    k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
        * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
    tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    return pat.masked_fill(~tril, 0.0)


def kern_from_mean(mp):
    kern = torch.zeros_like(mp)
    for d_ in range(T):
        idxs = torch.arange(d_, T)
        kern[:, idxs, idxs - d_] = mp[:, idxs, idxs - d_].mean(1).unsqueeze(1)
    return kern


@torch.no_grad()
def sweep(rows, kerns, replace, roster=False, accumulate=False):
    """One pass over rows. If replace: pat -> kerns[L] (roster layers keep SPEC heads
    live when roster=True). If accumulate: return per-layer mean of the WOULD-BE live
    patterns (measured just before replacement)."""
    ACC = {L: torch.zeros(9, T, T) for L in range(18)} if accumulate else None
    nb = 0
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i + 4, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            pat = block_pat(at, xin, B)
            if accumulate:
                ACC[L] += pat.float().mean(0).cpu()
            if replace:
                newp = kerns[L].unsqueeze(0).expand(B, -1, -1, -1).to(pat.dtype).clone()
                if roster and L in SPEC:
                    for hh in SPEC[L]:
                        newp[:, hh] = pat[:, hh]
                pat = newp
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        nb += 1
    if accumulate:
        return {L: kern_from_mean((ACC[L] / nb).to(DEV)) for L in range(18)}
    return None


@torch.no_grad()
def ce_run(EVR, kerns, replace, roster=False):
    s_ = 0.0; n_ = 0
    for i in range(0, NR, 8):
        bb = EVR[i:i + 8].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            pat = block_pat(at, xin, B)
            if replace:
                newp = kerns[L].unsqueeze(0).expand(B, -1, -1, -1).to(pat.dtype).clone()
                if roster and L in SPEC:
                    for hh in SPEC[L]:
                        newp[:, hh] = pat[:, hh]
                pat = newp
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        lo = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                             reduction='none').view(tg.shape)
        mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
        s_ += float(ce[mk].sum()); n_ += int(mk.sum())
    return s_ / max(n_, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    MEANR = cl.fineweb_rows(NMEAN, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()

    v0 = sweep(MEANR, None, replace=False, accumulate=True)
    print("v0 kernels", flush=True)
    v1 = sweep(MEANR, v0, replace=True, accumulate=True)
    print("v1 kernels", flush=True)
    v2 = sweep(MEANR, v1, replace=True, accumulate=True)
    print("v2 kernels", flush=True)

    res = {'clean': round(ce_run(EVR, None, replace=False), 4)}
    for nm, kk in (('kernel_all_v0', v0), ('kernel_all_v1', v1),
                   ('kernel_all_v2', v2)):
        res[nm] = round(ce_run(EVR, kk, replace=True), 4)
        print(f"{nm}: {res[nm]}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    res['best_roster2_v1'] = round(ce_run(EVR, v1, replace=True, roster=True), 4)
    print(f"best_roster2_v1: {res['best_roster2_v1']}", flush=True)

    g1 = res['kernel_all_v0'] - res['kernel_all_v1']
    g2 = res['kernel_all_v1'] - res['kernel_all_v2']
    pa = res['kernel_all_v1'] <= 4.60
    pb = res['best_roster2_v1'] <= 4.30
    pc = g2 <= 0.5 * g1
    out = {'ce': res, 'gain_v1': round(g1, 4), 'gain_v2': round(g2, 4),
           'pred_a_v1_le_460': bool(pa), 'pred_b_roster_v1_le_430': bool(pb),
           'pred_c_converges': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"v1 gain {g1:.4f} v2 gain {g2:.4f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
