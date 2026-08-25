# attn_composite4: TWO FIXES FROM S1452-53. (1) The a10 roster was incomplete —
# screen found {2,3,4,5,6} = .821 per-layer (was .598); use it in the composite.
# (2) S1452's sink-constant FAILURE taught: the CLEAN-context optimal constant is
# miscalibrated in composite context (it lost -.073). The legitimate fix the spec
# already licenses: retrain the constant IN the composite (same price, 128 floats;
# same recipe as the 198-sweep: Adam 3e-3, mean-of-composite-context init, 150 steps,
# 480 rows skip=80, batch 8, scored positions >= 64). Frozen model, only the constant
# trains. Arms (NR=960 held-out, mask >= 64):
#   best_roster2 — kernel_all + live roster at {10,13,14,16,17} with a10 = {2,3,4,5,6}.
#   calib57      — best_roster2 + head 5.7 y-slice = composite-calibrated constant.
#   calib_all5   — best_roster2 + composite-calibrated constants for 5.7 AND 8.{1,2,3,7}
#                  (trained jointly in a second run).
#
# Registered predictions:
#   pred_a best_roster2 <= 4.40 (a10 completion moves the composite >= .05).
#   pred_b calib57 GAINS >= .05 over best_roster2 (recalibration flips S1452's sign).
#   pred_c calib_all5 <= 4.25.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_composite4_results.json'
NMEAN = 24; NR = 960; NTR = 480; STEPS = 150; LR = 3e-3
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
SPEC = {5: {7}, 8: {1, 2, 3, 7}, 10: {2, 3, 4, 5, 6}, 13: {0, 5, 8},
        14: {4, 6, 7}, 16: {0, 3, 4, 5}, 17: {0, 1, 2}}
LIVE_BEST = frozenset({10, 13, 14, 16, 17})
KERNS = {}


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


def fwd_full(idx, mode, consts=None):
    """mode None = clean. Else composite: kernels + LIVE_BEST rosters live.
    consts: {(L, h): tensor[128]} replacing y-slices (may carry grad)."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        pat = block_pat(at, xin, B)
        if mode is not None:
            newp = KERNS[L].unsqueeze(0).expand(B, -1, -1, -1).to(pat.dtype).clone()
            if L in SPEC and L in LIVE_BEST:
                for hh in SPEC[L]:
                    newp[:, hh] = pat[:, hh]
            pat = newp
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        if consts:
            for (cl_, ch), cv in consts.items():
                if cl_ == L:
                    y = y.clone()
                    y[:, :, ch, :] = cv.to(y.dtype)
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def train_consts(TRR, targets):
    """Train constants for targets=[(L,h),...] jointly inside the composite.
    Init = mean of each head's y-slice under the composite context."""
    with torch.no_grad():
        acc = {t: torch.zeros(128, device=DEV) for t in targets}; n = 0
        for i in range(0, 64, 8):
            idx = TRR[i:i + 8, :-1].to(DEV).contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            B = idx.shape[0]
            for L, blk in enumerate(H):
                at = blk.attn
                xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
                xin = F.rms_norm(xm, (D,))
                pat = block_pat(at, xin, B)
                newp = KERNS[L].unsqueeze(0).expand(B, -1, -1, -1).to(pat.dtype).clone()
                if L in SPEC and L in LIVE_BEST:
                    for hh in SPEC[L]:
                        newp[:, hh] = pat[:, hh]
                pat = newp
                v = at.c_v(xin).view(B, T, 9, 128)
                if v1 is None:
                    v1 = v
                vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
                for (cl_, ch) in targets:
                    if cl_ == L:
                        acc[(cl_, ch)] += y[:, :, ch, :].float().mean((0, 1)) * B
                x = xm + at.c_proj(y.reshape(B, T, D))
                x = x + blk.mlp(F.rms_norm(x, (D,)))
            n += B
    consts = {t: torch.nn.Parameter((acc[t] / n).clone()) for t in targets}
    opt = torch.optim.Adam(list(consts.values()), lr=LR)
    curve = []
    g = torch.Generator().manual_seed(7)
    for step in range(STEPS):
        sel = torch.randint(0, NTR, (8,), generator=g)
        bb = TRR[sel].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
        lo = fwd_full(idx, 'comp', consts).float()
        ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                             reduction='none').view(tg.shape)
        loss = ce[:, 64:].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        curve.append(round(float(loss), 4))
        if step % 30 == 0:
            print(f"  step {step} loss {float(loss):.4f}", flush=True)
    return {t: v.detach() for t, v in consts.items()}, curve


@torch.no_grad()
def ce_run(EVR, mode, consts=None):
    s_ = 0.0; n_ = 0
    for i in range(0, NR, 8):
        bb = EVR[i:i + 8].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
        lo = fwd_full(idx, mode, consts).float()
        ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                             reduction='none').view(tg.shape)
        mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
        s_ += float(ce[mk].sum()); n_ += int(mk.sum())
    return s_ / max(n_, 1)


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    for p in m.parameters():
        p.requires_grad_(False)
    MEANR = cl.fineweb_rows(NMEAN, skip=80)[:, :T + 1].contiguous()
    TRR = cl.fineweb_rows(NTR, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()

    with torch.no_grad():
        ACC = {L: torch.zeros(9, T, T) for L in range(18)}
        nb = 0
        for i in range(0, NMEAN, 4):
            idx = MEANR[i:i + 4, :-1].to(DEV).contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            B = idx.shape[0]
            for L, blk in enumerate(H):
                at = blk.attn
                xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
                xin = F.rms_norm(xm, (D,))
                pat = block_pat(at, xin, B)
                ACC[L] += pat.float().mean(0).cpu()
                v = at.c_v(xin).view(B, T, 9, 128)
                if v1 is None:
                    v1 = v
                vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
                x = xm + at.c_proj(y.reshape(B, T, D))
                x = x + blk.mlp(F.rms_norm(x, (D,)))
            nb += 1
        for L in range(18):
            mp = (ACC[L] / nb).to(DEV)
            kern = torch.zeros_like(mp)
            for d_ in range(T):
                idxs = torch.arange(d_, T)
                kern[:, idxs, idxs - d_] = mp[:, idxs, idxs - d_].mean(1).unsqueeze(1)
            KERNS[L] = kern
    print("kernels cached", flush=True)

    res = {}
    res['clean'] = round(ce_run(EVR, None), 4)
    res['best_roster2'] = round(ce_run(EVR, 'comp'), 4)
    print(f"clean {res['clean']} best_roster2 {res['best_roster2']}", flush=True)
    json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    C57, curve57 = train_consts(TRR, [(5, 7)])
    res['calib57'] = round(ce_run(EVR, 'comp', C57), 4)
    print(f"calib57 {res['calib57']}", flush=True)
    json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    CA, curveA = train_consts(TRR, [(5, 7), (8, 1), (8, 2), (8, 3), (8, 7)])
    res['calib_all5'] = round(ce_run(EVR, 'comp', CA), 4)
    print(f"calib_all5 {res['calib_all5']}", flush=True)

    pa = res['best_roster2'] <= 4.40
    pb = (res['best_roster2'] - res['calib57']) >= 0.05
    pc = res['calib_all5'] <= 4.25
    out = {'ce': res, 'gain_calib57': round(res['best_roster2'] - res['calib57'], 4),
           'gain_all5': round(res['calib57'] - res['calib_all5'], 4),
           'curve57': curve57, 'curve_all5': curveA,
           'data_budget': {'train_rows': NTR, 'steps': STEPS, 'batch': 8, 'lr': LR},
           'pred_a_roster2_le_440': bool(pa), 'pred_b_calib_gain_05': bool(pb),
           'pred_c_all5_le_425': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
