# circuit_trunk: SUB-LAYER FEEDER SEARCH (S1516: whole early layers are a shared
# trunk; selectivity needs head grain). Circuit: semicolon (5-head ensemble).
# Stage 1: each of the 63 early heads (layers 0-6) exact SINGLY on top of
# bg+ensemble, scored by class-recovery gain (NR=120 screening). Stage 2: greedy
# over the top-10 (accept if class gain >= .02 and class gain >= global gain),
# verified at NR=960.
# Registered predictions:
#   pred_a <= 10 exact early heads recover >= .40 of the semicolon class gap.
#   pred_b the found feeder set is SELECTIVE: class recovery >= 1.3x global.
#   pred_c >= 6 of the top-10 single-head gains sit in layers 0-2.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_trunk_results.json'
NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENSEMBLE = None  # filled in-script from the weights-only score (top-8)
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
ENS_C = {}   # filled after the ensemble is chosen


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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    MEANR = cl.fineweb_rows(24, skip=80)[:, :T + 1].contiguous()

    # offset-averaged patterns for the extraction background
    ACC = {L: torch.zeros(9, T, T) for L in range(18)}
    nb = 0
    for i in range(0, 24, 4):
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
    KERNS = {}
    for L in range(18):
        mp = (ACC[L] / nb).to(DEV)
        kern = torch.zeros_like(mp)
        for d_ in range(T):
            idxs = torch.arange(d_, T)
            kern[:, idxs, idxs - d_] = mp[:, idxs, idxs - d_].mean(1).unsqueeze(1)
        KERNS[L] = kern
    print("patterns cached", flush=True)

    # mlp0 interaction subspace (top-8, RMS-whitened composed block-1 reads)
    FR = cl.fineweb_rows(480, skip=80)[:, :T + 1].contiguous()
    a1 = torch.zeros(HD, device=DEV); a2 = torch.zeros(HD, device=DEV); n0 = 0
    for i in range(0, 480, 8):
        idx = FR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
        blk = H[0]
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        pat = block_pat(blk.attn, xin, idx.shape[0])
        v = blk.attn.c_v(xin).view(-1, T, 9, 128)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(v.dtype), v)
        xx = xm + blk.attn.c_proj(y.reshape(-1, T, D))
        z = F.rms_norm(xx, (D,))
        h = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float()).reshape(-1, HD)
        a1 += h.sum(0); a2 += (h * h).sum(0); n0 += h.shape[0]
    MU = a1 / n0
    RMS = (a2 / n0).clamp_min(1e-12).sqrt()
    at1 = H[1].attn
    Wd0 = H[0].mlp.Down.weight.float().to(DEV)
    STACK = torch.cat([at1.c_q.weight.float().to(DEV) @ Wd0,
                       at1.c_k.weight.float().to(DEV) @ Wd0,
                       at1.c_q2.weight.float().to(DEV) @ Wd0,
                       at1.c_k2.weight.float().to(DEV) @ Wd0,
                       at1.c_v.weight.float().to(DEV) @ Wd0,
                       H[1].mlp.Left.weight.float().to(DEV) @ Wd0,
                       H[1].mlp.Right.weight.float().to(DEV) @ Wd0], 0)
    _, _, Vt = torch.linalg.svd(STACK * RMS.unsqueeze(0), full_matrices=False)
    W8 = Vt[:8]
    print("subspace built", flush=True)

    import tiktoken, re
    ENC = tiktoken.get_encoding('gpt2')
    def rx(pat):
        v = torch.zeros(50257, dtype=torch.bool)
        for t in range(50257):
            if re.match(pat, ENC.decode([t])):
                v[t] = True
        return v
    CIRCUITS = {
        'question': {'mask': rx(r'^\?$| \?$'),
                     'heads': [(10, 5), (12, 6), (15, 6), (15, 1), (9, 7)]},
        'semicolon': {'mask': rx(r'^;$'),
                      'heads': [(12, 6), (13, 3), (15, 1), (10, 5), (13, 8)]},
        'close_paren': {'mask': rx(r'^\)|^ ?\)$'),
                        'heads': [(13, 8), (15, 1), (7, 2), (12, 6), (16, 8)]},
    }
    # NOTE: semicolon/close_paren head lists from the screen results json
    import json as _j
    scr1 = _j.load(open(PT + 'circuit_screen_results.json'))['res']
    scr2 = _j.load(open(PT + 'circuit_screen2_results.json'))['res']
    def parse(hs):
        return [(int(s.split('.')[0]), int(s.split('.')[1])) for s in hs]
    CIRCUITS['question']['heads'] = parse(scr1['question']['W']['heads'])
    CIRCUITS['close_paren']['heads'] = parse(scr1['close_paren']['W']['heads'])
    CIRCUITS['semicolon']['heads'] = parse(scr2['semicolon']['W']['heads'])
    global ENSEMBLE
    # rank-32 whitened QK background for ALL layers (extraction arms)
    CR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    XACC = {L: torch.zeros(D, D, device=DEV) for L in range(18)}
    ncov = 0
    for i in range(0, 96, 8):
        idx = CR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1_ = None
        for L, blk in enumerate(H):
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            Xf = xin.float().reshape(-1, D)
            XACC[L] += Xf.T @ Xf
            x, v1_ = blk(x, v1_, x0)
        ncov += idx.shape[0] * T
    def trunc_perhead(W, r_, Wh, Whi):
        Wf = W.float().to(DEV).view(9, 128, D)
        out = torch.zeros_like(Wf)
        for h in range(9):
            U, S, Vt = torch.linalg.svd(Wf[h] @ Wh, full_matrices=False)
            out[h] = ((U[:, :r_] * S[:r_]) @ Vt[:r_]) @ Whi
        return out.view(9 * 128, D)
    TWALL = {}
    for L in range(18):
        Sg = XACC[L] / ncov
        ev, Vv = torch.linalg.eigh(Sg)
        ev = ev.clamp_min(1e-6)
        Wh = Vv @ torch.diag(ev.sqrt()) @ Vv.T
        Whi = Vv @ torch.diag(ev.rsqrt()) @ Vv.T
        at = H[L].attn
        TWALL[L] = {'q': trunc_perhead(at.c_q.weight, 32, Wh, Whi),
                    'k': trunc_perhead(at.c_k.weight, 32, Wh, Whi),
                    'q2': trunc_perhead(at.c_q2.weight, 32, Wh, Whi),
                    'k2': trunc_perhead(at.c_k2.weight, 32, Wh, Whi)}
        print(f"bg maps L{L}", flush=True)

    @torch.no_grad()
    def fwd(idx, rm_ens=False, rm_sub=False, bg=False, ens_exact=False,
            exact_layers=frozenset(), exact_heads=frozenset()):
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            if bg and L not in exact_layers:
                TW = TWALL[L]
                qp = (xin.float() @ TW['q'].T).view(B, T, 9, 128)
                kp = (xin.float() @ TW['k'].T).view(B, T, 9, 128)
                q2p = (xin.float() @ TW['q2'].T).view(B, T, 9, 128)
                k2p = (xin.float() @ TW['k2'].T).view(B, T, 9, 128)
                keep = [hh for hh in range(9)
                        if (ens_exact and L in ENSEMBLE and hh in ENSEMBLE[L])
                        or (L, hh) in exact_heads]
                if keep:
                    qf = at.c_q(xin).view(B, T, 9, 128).float()
                    kf = at.c_k(xin).view(B, T, 9, 128).float()
                    q2f = at.c_q2(xin).view(B, T, 9, 128).float()
                    k2f = at.c_k2(xin).view(B, T, 9, 128).float()
                    for hh in keep:
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
            if rm_ens and L in ENSEMBLE:
                y = y.clone()
                for hh in ENSEMBLE[L]:
                    y[:, :, hh, :] = ENS_C[(L, hh)].to(y.dtype)
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)

    def ce_run(**kw):
        s_ = 0.0; n_ = 0; sc = 0.0; nc = 0
        tsum = torch.zeros(50257); tn = torch.zeros(50257)
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx, **kw).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cls = CAPSET.to(DEV)[tg] & mk
            s_ += float(ce[mk & ~cls].sum()); n_ += int((mk & ~cls).sum())
            sc += float(ce[cls].sum()); nc += int(cls.sum())
            tgf = tg.cpu().reshape(-1)
            cef = (ce * (mk & cls).float()).cpu().reshape(-1)
            mkf = (mk & cls).cpu().reshape(-1).float()
            tsum.index_add_(0, tgf, cef)
            tn.index_add_(0, tgf, mkf)
        return {'global': s_ / max(n_, 1), 'cls': sc / max(nc, 1)}, tsum, tn

    spec = CIRCUITS['semicolon']
    ENSEMBLE = {}
    for (L, hh) in spec['heads']:
        ENSEMBLE.setdefault(L, []).append(hh)
    CAPSET = spec['mask']
    SCR = EVR[:120]

    def ce_run(rows, **kw):
        s_ = 0.0; n_ = 0; sc = 0.0; nc = 0
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx, **kw).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = CAPSET.to(DEV)[tg] & mk
            s_ += float(ce[mk & ~cm].sum()); n_ += int((mk & ~cm).sum())
            sc += float(ce[cm].sum()); nc += int(cm.sum())
        return s_ / max(n_, 1), sc / max(nc, 1)

    g_c, c_c = ce_run(SCR)
    g_b, c_b = ce_run(SCR, bg=True, ens_exact=True)
    print(f"screen clean cls {c_c:.4f} bg_ens cls {c_b:.4f}", flush=True)

    singles = {}
    for L in range(7):
        for hh in range(9):
            g1, c1 = ce_run(SCR, bg=True, ens_exact=True,
                            exact_heads=frozenset({(L, hh)}))
            singles[(L, hh)] = c_b - c1          # class-CE reduction
        print(f"layer {L} singles done", flush=True)
    top10 = sorted(singles, key=lambda k2: -singles[k2])[:10]
    print("top10:", [(f'{L}.{h}', round(singles[(L, h)], 4))
                     for L, h in top10], flush=True)

    cur = []
    best_c = c_b
    for hd in top10:
        trial = cur + [hd]
        g1, c1 = ce_run(SCR, bg=True, ens_exact=True,
                        exact_heads=frozenset(trial))
        if (best_c - c1) >= 0.02 and (best_c - c1) >= (g_b - g1) - (g_b - g1) * 0 \
           and (best_c - c1) > 0:
            cur = trial; best_c = c1
    print("feeder set:", [f'{L}.{h}' for L, h in cur], flush=True)

    gV0, cV0 = ce_run(EVR)
    gVb, cVb = ce_run(EVR, bg=True, ens_exact=True)
    gVf, cVf = ce_run(EVR, bg=True, ens_exact=True, exact_heads=frozenset(cur))
    rec_cls = (cVb - cVf) / max(cVb - cV0, 1e-6)
    rec_gl = (gVb - gVf) / max(gVb - gV0, 1e-6)
    n_early = sum(1 for (L, h) in top10 if L <= 2)
    pa = len(cur) <= 10 and rec_cls >= 0.40
    pb = rec_cls >= 1.3 * max(rec_gl, 1e-6)
    pc = n_early >= 6
    out = {'top10_singles': [[f'{L}.{h}', round(singles[(L, h)], 4)]
                             for L, h in top10],
           'feeder_set': [f'{L}.{h}' for L, h in cur],
           'verified': {'rec_cls': round(rec_cls, 4), 'rec_global': round(rec_gl, 4)},
           'pred_a_10heads_40': bool(pa), 'pred_b_selective_13': bool(pb),
           'pred_c_top10_early_6': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(out['verified'])
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
