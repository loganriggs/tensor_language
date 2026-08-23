"""DECONFOUND of §1126 (registered there): hold the TARGET fixed at the top-64 API coords of C14; widen only
the INPUTS: K_in = 64 | 256 (stream coords + matching attention sums) | RAW-512 (rank-512 PCA of the raw L5
residual + raw attention sums, no content-basis restriction at all). If wider inputs gain nothing on the fixed
top-64 target, the missing ~30% is not linearly present in the L5 state + attention at ANY width — the scratch
is computed through, not carried (the §1118 frame, now at simulation level); if RAW-512 gains materially, the
off-coordinate share was just width.

REGISTERED PREDICTIONS:
  (0) SANITY: K_in=64 reproduces §1124 (+attn 0.705 on the top-64 target).
  (a) NOT-CARRIED: RAW-512 inputs gain < 0.05 over K_in=64 -> no linear window on the L5 stream carries the
      missing information; construction simulation requires running the nonlinear steps (final boundary);
  (b) WIDTH AFTER ALL: RAW-512 (+attn) >= 0.80 -> §1125/§1126 were width artifacts; report plainly."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'coord_direct_kin_results.json'
NSEQ = 192; SEQ = 256; BAND = list(range(5, 15)); EMB32 = 32
KS = [64, 128, 256]
H = m.transformer.h
SPANS = [(5, 14)]


def fit_linear(X, Y):
    A = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)
    Wl = torch.linalg.lstsq(A, Y).solution
    return lambda Z: torch.cat([Z, torch.ones(Z.shape[0], 1, device=Z.device)], 1) @ Wl


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); nb = NSEQ; ntr = int(0.7*nb); T = SEQ - 1

    caps = {L: [] for L in BAND}; capsA = {L: [] for L in BAND}
    hsA = []
    for L in BAND:
        def mkA(L):
            def h(mo, i_, o_):
                y = o_[0] if isinstance(o_, tuple) else o_
                capsA[L].append(y.detach().float().reshape(-1, D))
            return h
        hsA.append(H[L].attn.register_forward_hook(mkA(L)))
    idsL = []
    for i in range(0, nb, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1))
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for Li, blk in enumerate(H):
            x, v1 = blk(x, v1, x0)
            if Li in BAND: caps[Li].append(x.detach().float().reshape(-1, D))
    for h in hsA: h.remove()
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))

    devsum = None; xbars = {}
    for L in BAND:
        X = torch.cat(caps[L], 0); caps[L] = X
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
        xbars[L] = xb/cn.clamp_min(1).unsqueeze(1)
        if L in (8, 10, 12):
            dv = X - xbars[L][tok]
            devsum = dv if devsum is None else devsum + dv
    dev = devsum/3; dev = dev - dev.mean(0)
    _, _, VtF = torch.linalg.svd(dev, full_matrices=False)
    del dev, devsum
    A_raw = {L: torch.cat(capsA[L], 0) for L in BAND}
    for L in BAND: capsA[L] = None
    _, _, Vte = torch.linalg.svd(m.transformer.wte.weight.float()[torch.unique(tok)], full_matrices=False)
    TF = (m.transformer.wte(tok).float() @ Vte[:EMB32].T.contiguous()).view(nb, T, EMB32)

    trm = torch.zeros(nb, dtype=torch.bool); trm[:ntr] = True
    ev = ~trm
    def flat(x, mask): return x[mask].reshape(-1, x.shape[-1])
    def cosR2(pred, true):
        return (round(float(F.cosine_similarity(pred, true, dim=-1).mean()), 3),
                round(1 - float(((pred-true)**2).sum()/((true - true.mean(0))**2).sum()), 3))

    # FIXED TARGET: top-64 API coords of C14
    Uc64 = VtF[:64].T.contiguous()
    Y = ((caps[14] - xbars[14][tok]) @ Uc64).view(nb, T, 64)
    # raw-residual 512 basis for the widest input
    Xr5 = caps[5] - xbars[5][tok]
    _, _, Vr = torch.linalg.svd(Xr5[:40000] - Xr5[:40000].mean(0), full_matrices=False)
    U512 = Vr[:512].T.contiguous()
    res = {}
    for name, Kin, U in [('K64', 64, Uc64), ('K256', 256, VtF[:256].T.contiguous()), ('RAW512', 512, U512)]:
        C5 = (Xr5 @ U).view(nb, T, Kin)
        attn_sum = sum((A_raw[L2] @ U) for L2 in range(6, 15)).view(nb, T, Kin)
        cs = C5.cumsum(1); pool = cs/torch.arange(1, T+1, device=DEV).view(1, T, 1).float()
        Xa = torch.cat([C5, TF, pool, attn_sum], -1)
        fa = fit_linear(flat(Xa, trm), flat(Y, trm))
        ra = cosR2(fa(flat(Xa, ev)), flat(Y, ev))
        res[name] = {'plus_attn': ra}
        print(f"{name}: +attn on fixed top-64 target {ra}", flush=True)
    out = {'per_input': res}
    gain = res['RAW512']['plus_attn'][1] - res['K64']['plus_attn'][1]
    out['raw512_gain'] = round(gain, 3)
    out['pred_a_not_carried'] = bool(gain < 0.05)
    out['pred_b_width_after_all'] = bool(res['RAW512']['plus_attn'][1] >= 0.80)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"RAW512 gain {gain:+.3f} | pred_a not-carried {out['pred_a_not_carried']} | pred_b width-after-all {out['pred_b_width_after_all']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
