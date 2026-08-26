# attn_composite6: TRAIN THE KERNELS AGAINST CE IN-COMPOSITE (S1458: the fixed-point
# moment refit DIVERGED — v1 6.74 > v0 4.73 — because averaging patterns measured on a
# corrupted stream imitates the corruption. The v4 constants worked because they were
# trained against the TRUE objective. Same move here: each layer's kernel is a [9, T]
# offset curve = ~41k trainable params total (identical 37 Kbit/layer price), init v0,
# Adam 3e-3 vs full-model CE inside kernel-all, 300 steps, batch 8 of 480 rows skip=80,
# scored positions >= 64. Frozen model. Loss curve + data budget recorded.
# Arms (NR=960 held-out): kernel_all_v0 (ref) / kernel_all_trained /
# roster_trained (trained kernels + live roster at {10,13,14,16,17}, a10={2,3,4,5,6};
# trained in kernel_all config — mismatch registered as an assumption).
#
# Registered predictions:
#   pred_a kernel_all_trained <= 4.43 (CE-training gains >= .30 where moment refit lost 2.0).
#   pred_b roster_trained <= 4.20.
#   pred_c held-out trained CE within .10 of final train-batch CE (no overfit at 41k params).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_composite6_results.json'
NMEAN = 24; NR = 960; NTR = 480; STEPS = 300; LR = 3e-3
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
SPEC = {10: {2, 3, 4, 5, 6}, 13: {0, 5, 8}, 14: {4, 6, 7},
        16: {0, 3, 4, 5}, 17: {0, 1, 2}}

OFF = (torch.arange(T).unsqueeze(1) - torch.arange(T).unsqueeze(0)).clamp_min(0).to(DEV)
TRIL = torch.tril(torch.ones(T, T, dtype=torch.bool)).to(DEV)


def block_pat(at, xin, B):
    cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
    q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
    k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
    q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
    k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
        * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
    return pat.masked_fill(~TRIL, 0.0)


def expand_curve(curve):
    """curve [9, T] offset values -> [9, T, T] lower-tri pattern."""
    return curve[:, OFF] * TRIL


def fwd_comp(idx, curves, roster=False):
    """Composite fwd; curves = {L: [9,T] tensor (may require grad)}."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        kp = expand_curve(curves[L]).unsqueeze(0).expand(B, -1, -1, -1)
        if roster and L in SPEC:
            live = block_pat(at, xin, B)
            kp = kp.clone()
            for hh in SPEC[L]:
                kp[:, hh] = live[:, hh]
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', kp.to(vv.dtype), vv)
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def fwd_clean(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def v0_curves(rows):
    ACC = {L: torch.zeros(9, T, device=DEV) for L in range(18)}
    CNT = torch.zeros(T, device=DEV)
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
            mp = pat.float().mean(0)                     # [9, T, T]
            for d_ in range(T):
                idxs = torch.arange(d_, T, device=DEV)
                ACC[L][:, d_] += mp[:, idxs, idxs - d_].mean(1)
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        nb += 1
    return {L: (ACC[L] / nb) for L in range(18)}


@torch.no_grad()
def ce_eval(EVR, curves, roster=False, clean=False):
    s_ = 0.0; n_ = 0
    for i in range(0, NR, 8):
        bb = EVR[i:i + 8].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
        lo = (fwd_clean(idx) if clean else fwd_comp(idx, curves, roster)).float()
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

    v0 = v0_curves(MEANR)
    print("v0 curves", flush=True)

    res = {'clean': round(ce_eval(EVR, None, clean=True), 4)}
    res['kernel_all_v0'] = round(ce_eval(EVR, v0), 4)
    print(f"clean {res['clean']} v0 {res['kernel_all_v0']}", flush=True)

    curves = {L: torch.nn.Parameter(v0[L].clone()) for L in range(18)}
    opt = torch.optim.Adam(list(curves.values()), lr=LR)
    g = torch.Generator().manual_seed(11)
    curve_log = []
    for step in range(STEPS):
        sel = torch.randint(0, NTR, (8,), generator=g)
        bb = TRR[sel].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
        lo = fwd_comp(idx, curves).float()
        ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                             reduction='none').view(tg.shape)
        loss = ce[:, 64:].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        curve_log.append(round(float(loss), 4))
        if step % 50 == 0:
            print(f"  step {step} loss {float(loss):.4f}", flush=True)
    trained = {L: curves[L].detach() for L in range(18)}
    train_tail = sum(curve_log[-20:]) / 20

    res['kernel_all_trained'] = round(ce_eval(EVR, trained), 4)
    res['roster_trained'] = round(ce_eval(EVR, trained, roster=True), 4)
    print(f"trained {res['kernel_all_trained']} roster {res['roster_trained']}",
          flush=True)

    pa = res['kernel_all_trained'] <= 4.43
    pb = res['roster_trained'] <= 4.20
    pc = abs(res['kernel_all_trained'] - train_tail) <= 0.10
    out = {'ce': res, 'train_tail_mean20': round(train_tail, 4),
           'loss_curve': curve_log,
           'data_budget': {'train_rows': NTR, 'steps': STEPS, 'batch': 8, 'lr': LR},
           'pred_a_trained_le_443': bool(pa), 'pred_b_roster_le_420': bool(pb),
           'pred_c_no_overfit_10': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
