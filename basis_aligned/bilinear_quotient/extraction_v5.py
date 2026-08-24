# extraction_v5: RUNG 5 — §1321's fix. The payload must be injected STREAM-CALIBRATED:
# normalize the computed payload per position and scale it as a fraction alpha of the
# local stream rms at L8 entry, instead of a fixed absolute pscale fitted in a healthy
# stream. Fine alpha grid on fit rows; same conditions and benchmark as rung 4.
#
# Registered predictions:
#   pred_a CALIBRATED PAYLOAD TRANSFERS: some alpha > 0 beats axis-only by >= 5 points
#          of the quad gap.
#   pred_b WRONG-SUCCESSOR FLAT at the chosen alpha (<= axis-only + 5 points).
#   pred_c SMALL ALPHA: the chosen alpha <= 0.15.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'extraction_v5_results.json'
NFIT = 12; NR = 24; NMEAN = 24; QSTART = 128; WIN = 64
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
MATCHERS = {2: [5], 3: [8]}
QUAD = {2: {5}, 3: {8}, 8: {3, 4}}
QUADSET = {(2, 5), (3, 8), (8, 3), (8, 4)}
CLOSURE33 = {(L, h) for L in (0, 1, 2) for h in range(9)} | {(3, 8), (5, 7), (8, 3), (8, 4)}
MASK_W = None; FULL = None


@torch.no_grad()
def fwd_plain(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def fit_axis(rows):
    DL = []
    for i in range(0, NFIT, 4):
        idx = rows[i:i + 4, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        wsum = torch.zeros(B, T, D, device=DEV)
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
            q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
            k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
            q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
            k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
                * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
            pat = pat.masked_fill(~FULL, 0.0)
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            if L in MATCHERS:
                for h in MATCHERS[L]:
                    yh = torch.zeros_like(y)
                    yh[:, :, h, :] = y[:, :, h, :]
                    wsum = wsum + at.c_proj(yh.reshape(B, T, D)).float()
            x = xm + at.c_proj(y.reshape(B, T, D))
            if L == 3:
                fullm = blk.mlp(F.rms_norm(x, (D,)))
                blind = blk.mlp(F.rms_norm(x - wsum.to(x.dtype), (D,)))
                DL.append((fullm - blind).float()[:, 160:].reshape(-1, D).cpu())
                break
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    DL = torch.cat(DL)
    Dc = DL - DL.mean(0)
    _, _, V = torch.pca_lowrank(Dc, q=4)
    return V[:, 0].to(DEV)


@torch.no_grad()
def weight_scores_max(idx):
    B = idx.shape[0]
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    tot = torch.zeros(B, T, T, device=DEV)
    for L, hs in MATCHERS.items():
        at = H[L].attn
        dummy = torch.zeros(1, T, 9, 128, device=DEV)
        cos_t, sin_t = at.rotary(dummy)
        def pipe(lin):
            z = F.rms_norm(lin(x).view(B, T, 9, 128), (128,))
            return are(z, cos_t, sin_t)
        q = pipe(at.c_q); k = pipe(at.c_k); q2 = pipe(at.c_q2); k2 = pipe(at.c_k2)
        for h in hs:
            s1 = torch.einsum('bqd,bkd->bqk', q[:, :, h].float(), k[:, :, h].float()) / 128.0
            s2 = torch.einsum('bqd,bkd->bqk', q2[:, :, h].float(), k2[:, :, h].float()) / 128.0
            tot += s1 * s2
    ar = torch.arange(T, device=DEV)
    far = (ar[:, None] - ar[None, :]) > WIN
    neg = (-tot).masked_fill(~far.unsqueeze(0), float('-inf'))
    val, arg = neg.max(dim=-1)
    off = ar[None, :] - arg
    val = torch.where(torch.isfinite(val), val, torch.zeros_like(val))
    return val, off


@torch.no_grad()
def fetch_payload(idx, succ_pos):
    at = H[8].attn
    B = idx.shape[0]
    succ_tok = torch.gather(idx, 1, succ_pos.clamp(0, T - 1))
    x = F.rms_norm(m.transformer.wte(succ_tok), (D,))
    v1s = H[0].attn.c_v(x).view(B, T, 9, 128)
    y = torch.zeros(B, T, 9, 128, device=DEV, dtype=v1s.dtype)
    lam = float(at.lamb)
    for h in (3, 4):
        y[:, :, h, :] = lam * v1s[:, :, h, :]
    return at.c_proj(y.reshape(B, T, D))


@torch.no_grad()
def fwd_v4(idx, vmeans, mode, axis_d=None, axis_vals=None, payload=None, pscale=0.0):
    """mode 'skeleton': closure_route with quad live. mode 'standin': quad read-masked +
    route-only values; axis/payload injections applied if given."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        if axis_d is not None and 4 <= L <= 8:
            cur = (x.float() * axis_d).sum(-1)
            x = x + ((axis_vals[L] - cur).unsqueeze(-1) * axis_d).to(x.dtype)
        if payload is not None and L == 8:
            pn = payload.float() / payload.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            srms = x.float().norm(dim=-1, keepdim=True)
            x = x + (pscale * srms * pn).to(x.dtype)
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        if mode == 'standin' and L in QUAD:
            msk = torch.stack([MASK_W if h in QUAD[L] else FULL for h in range(9)], 0)
            pat = pat.masked_fill(~msk.unsqueeze(0), 0.0)
        else:
            pat = pat.masked_fill(~FULL, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        v1v = v1.view_as(v)
        vv = (1 - at.lamb) * v + at.lamb * v1v
        # route mask: heads not in keep-set get mean fresh values + live v1 route
        keep = CLOSURE33 | {(2, 5)}
        if mode == 'standin':
            keep = keep - QUADSET
        sel = torch.tensor([(L, h) not in keep for h in range(9)], device=DEV).view(1, 1, 9, 1)
        vfixed = (1 - at.lamb) * vmeans[L].view(1, 1, 9, 128).to(vv.dtype) + at.lamb * v1v
        vv = torch.where(sel, vfixed, vv)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    global MASK_W, FULL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    FIT = cl.fineweb_rows(NFIT)[:, :T + 1].contiguous().clone()
    FIT[:, 128:256] = FIT[:, 0:128]
    REP = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    REP[:, 128:256] = REP[:, 0:128]
    MEANR = cl.fineweb_rows(NMEAN)[:, :T + 1].contiguous()

    # per-head fresh-value means
    vcaps = {L: [] for L in range(18)}; hooks = []
    for L in range(18):
        def mkv(L):
            def h(mod, args, out):
                vcaps[L].append(out.detach().float().view(out.shape[0], -1, 9, 128).mean((0, 1)))
                return out
            return h
        hooks.append(H[L].attn.c_v.register_forward_hook(mkv(L)))
    for i in range(0, NMEAN, 4):
        fwd_plain(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for h in hooks:
        h.remove()
    vmeans = {L: torch.stack(v).mean(0).to(DEV) for L, v in vcaps.items()}

    d = fit_axis(FIT)
    # affine fit: capture axis component in the FULL model at blocks 4-8 on FIT rows
    A = {}; Bc = {}
    caps_fit = {L: [] for L in range(4, 9)}; scores_fit = []
    for i in range(0, NFIT, 4):
        idx = FIT[i:i + 4, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for L, blk in enumerate(H):
            if 4 <= L <= 8:
                caps_fit[L].append((x.float() * d).sum(-1))
            x, v1 = blk(x, v1, x0)
        sc, _ = weight_scores_max(idx)
        scores_fit.append(sc)
    S = torch.cat(scores_fit)[:, QSTART:].reshape(-1)
    for L in range(4, 9):
        Y = torch.cat(caps_fit[L])[:, QSTART:].reshape(-1)
        sm, ym = S.mean(), Y.mean()
        beta = float(((S - sm) * (Y - ym)).sum() / ((S - sm) ** 2).sum().clamp_min(1e-9))
        A[L] = beta; Bc[L] = float(ym - beta * sm)

    # pscale grid inside the skeleton
    qp = torch.arange(QSTART, T, device=DEV)
    idxf = FIT[:8, :-1].to(DEV).contiguous(); tgtf = FIT[:8, 1:].to(DEV).contiguous()
    scf, offf = weight_scores_max(idxf)
    valsf = {L: A[L] * scf + Bc[L] for L in A}
    succf = (torch.arange(T, device=DEV)[None, :] - offf + 1).clamp(0, T - 1)
    pf = fetch_payload(idxf, succf)
    best_ps, best_ce = 0.0, 1e9
    for ps in (0.0, 0.02, 0.05, 0.1, 0.15, 0.25):
        lo = fwd_v4(idxf, vmeans, 'standin', d, valsf, pf, ps).float()
        cec = float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]), tgtf[:, qp].reshape(-1)))
        print(f"pscale {ps}: fit CE {cec:.4f}", flush=True)
        if cec < best_ce:
            best_ce, best_ps = cec, ps
    print(f"chosen pscale {best_ps}", flush=True)

    ce = {c: 0.0 for c in ('full', 'skeleton', 'skel_mask', 'skel_axis', 'skel_standin', 'skel_wrong')}
    n = 0
    for i in range(0, NR, 4):
        bb = REP[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        sc, off = weight_scores_max(idx)
        vals = {L: A[L] * sc + Bc[L] for L in A}
        pos = torch.arange(T, device=DEV)[None, :]
        pl = fetch_payload(idx, (pos - off + 1).clamp(0, T - 1))
        pl2 = fetch_payload(idx, (pos - off + 2).clamp(0, T - 1))
        outs = {'full': fwd_plain(idx),
                'skeleton': fwd_v4(idx, vmeans, 'skeleton'),
                'skel_mask': fwd_v4(idx, vmeans, 'standin'),
                'skel_axis': fwd_v4(idx, vmeans, 'standin', d, vals, None, 0.0),
                'skel_standin': fwd_v4(idx, vmeans, 'standin', d, vals, pl, best_ps),
                'skel_wrong': fwd_v4(idx, vmeans, 'standin', d, vals, pl2, best_ps)}
        for c, lo in outs.items():
            lo = lo.float()
            ce[c] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                           tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    gap = CE['skel_mask'] - CE['skeleton']
    rec = {c: round((CE['skel_mask'] - CE[c]) / max(gap, 1e-6), 4)
           for c in ('skel_axis', 'skel_standin', 'skel_wrong')}
    d_skel = CE['skeleton'] - CE['full']
    d_stand = CE['skel_standin'] - CE['full']
    pa = rec['skel_standin'] >= rec['skel_axis'] + 0.05
    pb2 = rec['skel_wrong'] <= rec['skel_axis'] + 0.05
    pc2 = best_ps <= 0.15
    out = {'n_rows': NR, 'pscale': best_ps, 'ce': CE, 'quad_gap_in_skeleton': round(gap, 4),
           'recovery': rec, 'dmg_vs_full': {'skeleton': round(d_skel, 4), 'standin': round(d_stand, 4)},
           'pred_a_calibrated_transfers': bool(pa), 'pred_b_wrong_flat': bool(pb2),
           'pred_c_small_alpha': bool(pc2),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"quad gap {gap:.4f} | recovery {rec} | dmg skel {d_skel:.4f} standin {d_stand:.4f}")
    print(f"pred_a calibrated {pa} | pred_b wrong-flat {pb2} | pred_c alpha {pc2} (alpha {best_ps})")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
