# ship_mlp2_diag: WHY DOES mlp2 COST 4x IN THE SHIP? (S1482: solo .119, in-ship
# +.470.) Decompose by subset algebra (same context-fit stand-ins, registered
# assumption: fits reused from the all3 context): attn+mlp2 (is it the attention
# context?), attn+mlp1 vs attn+mlp1+mlp2 (is it the mlp1 PAIR?), all4 (does mlp0
# add on top?).
# Original header: ATTENTION THREE-TIER COMPOSITE + THE mlp0/mlp1 PLANKS
# (S1477: the attention composite costs only .162 CE; the board's top two are mlp1
# and mlp0, which have verified per-module planks. This measures the actual glass
# ship so far.) MLP stand-ins are CONTEXT-FIT (lesson 3): tier tables + ridges fit on
# captures taken UNDER the attention composite; at eval they are computed INLINE
# single-pass (stand0 = TIER0[tok] + ridge0(a0); stand1 = TIER1[tok] +
# ridge1([a1, m0_in_stream]) where m0_in_stream is stand0 when mlp0 is also
# replaced). Arms: clean / attn / mlp1 / mlp0 / attn+mlp1 / attn+mlp1+mlp0.
# Original header: THE THREE-TIER CLASS AS A COMPOSITE (S1474: per-layer whitened
# r32 QK >= .84 everywhere; composite lessons 1-3 say per-layer fids do NOT predict
# the composite, so measure it). ALL 18 layers simultaneously on their best class:
# whitened r32 for generic heads (plain SVD at 8/16/17 per S1469 best-of), FULL QK
# for the named roster heads (SPEC below). MLPs/values live. Also measures each
# layer's SINGLE-replacement delta in-script so the compounding factor is
# self-contained. NR=960.
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
#   pred_a attn+mlp2 delta <= .24 (2x solo — attention context alone is mild).
#   pred_b the mlp1 pair is the culprit: delta(attn+m1+m2) - delta(attn+m1) >= .237
#          (2x mlp2's solo — the mlp1 stand-in starves mlp2's inputs).
#   pred_c mlp0 adds little on top: delta(all4) - delta(attn+m1+m2) <= .10.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ship_mlp2_diag_results.json'
NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
LAYERS = list(range(18))
PLAIN = {8, 16, 17}
SPEC = {5: {7}, 8: {1, 2, 3, 7}, 10: {2, 3, 4, 5, 6}, 13: {0, 5, 8},
        14: {4, 6, 7}, 16: {0, 3, 4, 5}, 17: {0, 1, 2}}


def trunc_perhead(W, r, Wh, Whi):
    """Whitened per-head rank-r: SVD of W_head @ Wh, then map back with Whi."""
    Wf = W.float().to(DEV).view(9, 128, D)
    out = torch.zeros_like(Wf)
    for h in range(9):
        U, S, Vt = torch.linalg.svd(Wf[h] @ Wh, full_matrices=False)
        out[h] = ((U[:, :r] * S[:r]) @ Vt[:r]) @ Whi
    return out.view(9 * 128, D)


SHIP = {'t0': None, 'r0': None, 't1': None, 'r1': None, 'r2': None, 'r17': None}


@torch.no_grad()
def fwd_arm(idx, layers, TWALL, mlps=frozenset(), cap=None):
    """layers: attn layers replaced. mlps: subset of {0,1,2,17} replaced inline.
    cap: optional dict collecting fit-pass tensors."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    toks = idx.reshape(-1)
    a0_out = None; a1_out = None; m0_stream = None
    mstream = []
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        if L in layers:
            TW = TWALL[L]
            qp = (xin.float() @ TW['q'].T).view(B, T, 9, 128)
            kp = (xin.float() @ TW['k'].T).view(B, T, 9, 128)
            q2p = (xin.float() @ TW['q2'].T).view(B, T, 9, 128)
            k2p = (xin.float() @ TW['k2'].T).view(B, T, 9, 128)
            if L in SPEC:
                qf = at.c_q(xin).view(B, T, 9, 128).float()
                kf = at.c_k(xin).view(B, T, 9, 128).float()
                q2f = at.c_q2(xin).view(B, T, 9, 128).float()
                k2f = at.c_k2(xin).view(B, T, 9, 128).float()
                for hh in SPEC[L]:
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
        ao = at.c_proj(y.reshape(B, T, D))
        x = xm + ao
        z = F.rms_norm(x, (D,))
        if L == 0:
            a0_out = ao.float()
            if 0 in mlps:
                W0, xm0_, ym0_ = SHIP['r0']
                st = SHIP['t0'][toks].view(B, T, D) \
                    + (ym0_ + (a0_out.reshape(-1, D) - xm0_) @ W0).view(B, T, D)
                mo = st.to(x.dtype)
            else:
                mo = blk.mlp(z)
            m0_stream = mo.float()
            mstream.append(m0_stream)
            x = x + mo
        elif L == 1:
            a1_out = ao.float()
            if 1 in mlps:
                W1, xm1_, ym1_ = SHIP['r1']
                X2 = torch.cat([a1_out, m0_stream], -1).reshape(-1, 2 * D)
                st = SHIP['t1'][toks].view(B, T, D) \
                    + (ym1_ + (X2 - xm1_) @ W1).view(B, T, D)
                mo = st.to(x.dtype)
            else:
                mo = blk.mlp(z)
            mstream.append(mo.float())
            x = x + mo
        elif L == 2:
            if cap is not None:
                cap.setdefault('a2', []).append(ao.float().cpu())
                cap.setdefault('m2', []).append(blk.mlp(z).float().cpu())
            if 2 in mlps:
                W2, xm2_, ym2_ = SHIP['r2']
                X2 = torch.cat([ao.float(), mstream[1]], -1).reshape(-1, 2 * D)
                mo = (ym2_ + (X2 - xm2_) @ W2).view(B, T, D).to(x.dtype)
            else:
                mo = blk.mlp(z)
            mstream.append(mo.float())
            x = x + mo
        elif L == 17:
            if cap is not None:
                cap.setdefault('a17', []).append(ao.float().cpu())
                cap.setdefault('m17', []).append(blk.mlp(z).float().cpu())
                cap.setdefault('mstream', []).append(
                    torch.cat(mstream, -1).cpu())
            if 17 in mlps:
                W7, xm7_, ym7_ = SHIP['r17']
                X7 = torch.cat([ao.float()] + mstream, -1).reshape(-1, 18 * D)
                mo = (ym7_ + (X7 - xm7_) @ W7).view(B, T, D).to(x.dtype)
            else:
                mo = blk.mlp(z)
            x = x + mo
        else:
            mo = blk.mlp(z)
            mstream.append(mo.float())
            x = x + mo
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()

    def ce_run(layers, TWALL):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, layers, TWALL).float()
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

    TWALL = {}
    for LT in LAYERS:
        at = H[LT].attn
        if LT in PLAIN:
            eye = torch.eye(D, device=DEV)
            Wh, Whi = eye, eye
        else:
            Wh, Whi = WHITEN[LT]
        TWALL[LT] = {'q': trunc_perhead(at.c_q.weight, 32, Wh, Whi),
                     'k': trunc_perhead(at.c_k.weight, 32, Wh, Whi),
                     'q2': trunc_perhead(at.c_q2.weight, 32, Wh, Whi),
                     'k2': trunc_perhead(at.c_k2.weight, 32, Wh, Whi)}
    print("attn maps built", flush=True)

    # ---- context-fit the mlp0/mlp1 planks UNDER the attention composite ----
    FITR = cl.fineweb_rows(480, skip=80)[:, :T + 1].contiguous()
    ALLL = frozenset(LAYERS)
    caps = {n: [] for n in ('a0', 'm0', 'a1', 'm1')}
    hks = [H[0].attn.c_proj.register_forward_hook(
               lambda mo, a, o: caps['a0'].append(o.detach().float().cpu())),
           H[0].mlp.register_forward_hook(
               lambda mo, a, o: caps['m0'].append(o.detach().float().cpu())),
           H[1].attn.c_proj.register_forward_hook(
               lambda mo, a, o: caps['a1'].append(o.detach().float().cpu())),
           H[1].mlp.register_forward_hook(
               lambda mo, a, o: caps['m1'].append(o.detach().float().cpu()))]
    for i in range(0, 480, 8):
        fwd_arm(FITR[i:i + 8, :-1].to(DEV).contiguous(), ALLL, TWALL)
    for h in hks:
        h.remove()
    FT = {n: torch.cat(v) for n, v in caps.items()}
    toksF = FITR[:, :-1].reshape(-1)
    print("context capture done", flush=True)

    def tier_table(Y, toks):
        tsum = torch.zeros(50257, D); tcnt = torch.zeros(50257)
        tsum.index_add_(0, toks, Y); tcnt.index_add_(0, toks,
                                                     torch.ones(toks.shape[0]))
        gm = Y.mean(0)
        TAB = torch.where(tcnt.unsqueeze(1) > 0,
                          tsum / tcnt.clamp_min(1).unsqueeze(1), gm.unsqueeze(0))
        Tc = (TAB - gm).to(DEV)
        U_, S_, Vt_ = torch.svd_lowrank(Tc, q=96, niter=4)
        base = gm.to(DEV) + (U_[:, :64] * S_[:64]) @ Vt_[:, :64].T
        keep = tcnt.argsort(descending=True)[:2000]
        TIER = base.clone(); TIER[keep] = TAB[keep].to(DEV)
        return TIER

    def ridge(X, Y):
        Xg = X.to(DEV); Yg = Y.to(DEV)
        xm_ = Xg.mean(0); ym_ = Yg.mean(0)
        Xc = Xg - xm_; Yc = Yg - ym_
        XtX = Xc.T @ Xc
        lam = 0.01 * float(torch.diagonal(XtX).mean())
        W = torch.linalg.solve(XtX + lam * torch.eye(X.shape[1], device=DEV),
                               Xc.T @ Yc)
        return W, xm_, ym_

    def ridge_big(X, Y):
        n, d = X.shape
        xm_ = X.mean(0).to(DEV); ym_ = Y.mean(0).to(DEV)
        XtX = torch.zeros(d, d, device=DEV)
        XtY = torch.zeros(d, Y.shape[1], device=DEV)
        CH = 8192
        for i in range(0, n, CH):
            Xc = X[i:i + CH].to(DEV) - xm_
            Yc = Y[i:i + CH].to(DEV) - ym_
            XtX += Xc.T @ Xc; XtY += Xc.T @ Yc
            del Xc, Yc
        lam = 0.01 * float(torch.diagonal(XtX).mean())
        W = torch.linalg.solve(XtX + lam * torch.eye(d, device=DEV), XtY)
        return W, xm_, ym_

    T0 = tier_table(FT['m0'].reshape(-1, D), toksF)
    R0Y = FT['m0'].reshape(-1, D) - T0[toksF].cpu()
    SHIP['t0'] = T0; SHIP['r0'] = ridge(FT['a0'].reshape(-1, D), R0Y)
    T1 = tier_table(FT['m1'].reshape(-1, D), toksF)
    R1Y = FT['m1'].reshape(-1, D) - T1[toksF].cpu()
    X1 = torch.cat([FT['a1'], FT['m0']], -1).reshape(-1, 2 * D)
    SHIP['t1'] = T1; SHIP['r1'] = ridge(X1, R1Y)
    print("planks 0/1 context-fit", flush=True)

    # second fit pass UNDER the all3 ship for mlp2 / mlp17 inputs
    cap2 = {}
    for i in range(0, 480, 8):
        fwd_arm(FITR[i:i + 8, :-1].to(DEV).contiguous(), ALLL, TWALL,
                frozenset({0, 1}), cap=cap2)
    C2 = {k: torch.cat(v) for k, v in cap2.items()}
    print("ship-context capture 2 done", flush=True)
    # NOTE: mstream captured at L==17 holds blocks 0..16 outputs = 17*D... plus we
    # appended m17? mstream at L17 has 17 entries (0..16). X = [a17 | mstream].
    X2f = torch.cat([C2['a2'],
                     C2['mstream'].reshape(-1, T, 17 * D)[:, :, D:2 * D]
                     .reshape(C2['a2'].shape)], -1).reshape(-1, 2 * D)
    SHIP['r2'] = ridge(X2f, C2['m2'].reshape(-1, D))
    X17f = torch.cat([C2['a17'].reshape(-1, D),
                      C2['mstream'].reshape(-1, 17 * D)], -1)
    SHIP['r17'] = ridge_big(X17f, C2['m17'].reshape(-1, D))
    print("planks 2/17 context-fit", flush=True)

    def ce2(layers, mlps):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, layers, TWALL, mlps).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    res = {}
    ARMS = [('clean', frozenset(), frozenset()),
            ('attn_mlp2', ALLL, frozenset({2})),
            ('attn_m1', ALLL, frozenset({1})),
            ('attn_m1_m2', ALLL, frozenset({1, 2})),
            ('all4', ALLL, frozenset({0, 1, 2}))]
    for nm, ly, mp in ARMS:
        res[nm] = round(ce2(ly, mp), 4)
        print(f"{nm}: {res[nm]}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    cl_ = res['clean']
    d = {k: res[k] - cl_ for k in res if k != 'clean'}
    pair_excess = d['attn_m1_m2'] - d['attn_m1']
    m0_add = d['all4'] - d['attn_m1_m2']
    pa = d['attn_mlp2'] <= 0.24
    pb = pair_excess >= 0.237
    pc = m0_add <= 0.10
    out = {'ce': res, 'deltas': {k: round(v, 4) for k, v in d.items()},
           'pair_excess_m1m2': round(pair_excess, 4), 'm0_addition': round(m0_add, 4),
           'pred_a_attn_m2_le_24': bool(pa), 'pred_b_m1_pair_ge_237': bool(pb),
           'pred_c_m0_le_10': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"deltas {out['deltas']} pair {pair_excess:.4f} m0add {m0_add:.4f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
