"""BOUNDARY-PRICING registered in §1125: fixed-64-coord construction simulation caps at R²≈0.70 with oracle
attention; the missing ~30% is off-coordinate (§1125: quadratics inert). Is it basis WIDTH (finite: R² climbs
with K) or basis DRIFT/positionality (flat in K)? Same direct + attn-sum fits, span L5→L14, at K = 64/128/256
(basis, coords, attention coords, and pool all rebuilt per K).

REGISTERED PREDICTIONS:
  (0) SANITY: K=64 reproduces §1124 (direct 0.534, +attn 0.705); R² non-decreasing in K.
  (a) WIDTH: +attn reaches >= 0.82 at K=256 -> the off-coordinate share is just basis width; the construction
      is simulable in a wide-enough fixed basis (finite object, closable);
  (b) DRIFT: if +attn gains < 0.05 from K=64 to 256, no fixed basis captures the inputs — the construction
      reads a DRIFTING frame (per-layer local bases required, §1095) — report plainly."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'coord_direct_k_results.json'
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

    res = {}
    for K in KS:
        Uc = VtF[:K].T.contiguous()
        C5 = ((caps[5] - xbars[5][tok]) @ Uc).view(nb, T, K)
        C14 = ((caps[14] - xbars[14][tok]) @ Uc).view(nb, T, K)
        attn_sum = sum((A_raw[L2] @ Uc) for L2 in range(6, 15)).view(nb, T, K)
        cs = C5.cumsum(1); pool = cs/torch.arange(1, T+1, device=DEV).view(1, T, 1).float()
        Xb = torch.cat([C5, TF, pool], -1)
        Xa = torch.cat([C5, TF, pool, attn_sum], -1)
        fb = fit_linear(flat(Xb, trm), flat(C14, trm))
        fa = fit_linear(flat(Xa, trm), flat(C14, trm))
        rb = cosR2(fb(flat(Xb, ev)), flat(C14, ev))
        ra = cosR2(fa(flat(Xa, ev)), flat(C14, ev))
        res[str(K)] = {'direct': rb, 'plus_attn': ra}
        print(f"K={K}: direct {rb} | +attn {ra}", flush=True)
    out = {'per_k': res, 'reference_K64_1124': {'direct': [0.638, 0.534], 'plus_attn': [0.738, 0.705]}}
    gain = res['256']['plus_attn'][1] - res['64']['plus_attn'][1]
    out['k_gain_64_to_256'] = round(gain, 3)
    out['pred_a_width'] = bool(res['256']['plus_attn'][1] >= 0.82)
    out['pred_b_drift'] = bool(gain < 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"K-gain (plus_attn, 64->256): {gain:+.3f} | pred_a width {out['pred_a_width']} | pred_b drift {out['pred_b_drift']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
