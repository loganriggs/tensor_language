"""CLOSURE test registered in §1124: the construction budget = linear transport 53% + attention 17% +
a front-loaded ~30% remainder hypothesized to be the §1041 dev×dev multiplication WRITTEN IN COORD SPACE.
Add coordinate-QUADRATIC features to the direct maps: quad(c) = {c_i*c_j : i<=j, i,j in top-16 coords}
(136 features) — the bilinear form's coordinate shadow. Conditions per span: direct | +attn-sum |
+quad | +attn-sum+quad; also a QUADRATIC NULL (products of 16 random-rotated coords — same capacity,
wrong axes... rotated within the 64-dim coord space).

REGISTERED PREDICTIONS:
  (0) SANITY: base conditions reproduce §1124 (L5→L14 direct 0.534, +attn 0.705); quad never hurts.
  (a) QUADRATIC CLOSES IT: L5→L14 with +attn-sum+quad reaches R² >= 0.85 -> construction FULLY characterized:
      linear + attention injections + coordinate-quadratic (the §1041 bilinear form in coords); the rotated-
      quad null should trail the true-axes quad by >= 0.05 (axis-specific, not generic capacity);
  (b) OFF-COORDINATE: if +attn+quad < 0.78, the remaining ~30% is NOT expressible in the 64 coords at all
      (off-coordinate D-space paths carry it) — then K must widen and the '64-coord content object' framing
      has a stated boundary (report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'coord_quadratic_results.json'
NSEQ = 192; SEQ = 256; K = 64; BAND = list(range(5, 15)); EMB32 = 32
H = m.transformer.h
SPANS = [(5, 9), (5, 14)]


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
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    Uc = Vt[:K].T.contiguous(); del dev, devsum

    C = {L: ((caps[L] - xbars[L][tok]) @ Uc).view(nb, T, K) for L in BAND}
    A = {L: ((torch.cat(capsA[L], 0)) @ Uc).view(nb, T, K) for L in BAND}
    for L in BAND: caps[L] = None; capsA[L] = None
    _, _, Vte = torch.linalg.svd(m.transformer.wte.weight.float()[torch.unique(tok)], full_matrices=False)
    TF = (m.transformer.wte(tok).float() @ Vte[:EMB32].T.contiguous()).view(nb, T, EMB32)
    def cpool(Cl):
        cs = Cl.cumsum(1)
        return cs/torch.arange(1, T+1, device=DEV).view(1, T, 1).float()

    trm = torch.zeros(nb, dtype=torch.bool); trm[:ntr] = True
    ev = ~trm
    def flat(x, mask): return x[mask].reshape(-1, x.shape[-1])
    def cosR2(pred, true):
        return (round(float(F.cosine_similarity(pred, true, dim=-1).mean()), 3),
                round(1 - float(((pred-true)**2).sum()/((true - true.mean(0))**2).sum()), 3))

    def quad_feats(Cl, P16):
        z = Cl @ P16                                        # ..., 16
        iu = torch.triu_indices(16, 16)
        q = z.unsqueeze(-1) * z.unsqueeze(-2)               # ..., 16, 16
        return q[..., iu[0], iu[1]]                          # ..., 136
    # true axes: top-16 coords; null: random rotation within the 64-dim coord space
    P_true = torch.zeros(K, 16, device=DEV); P_true[:16, :16] = torch.eye(16, device=DEV)
    g2 = torch.Generator(device=DEV).manual_seed(11)
    P_rot = torch.linalg.qr(torch.randn(K, K, generator=g2, device=DEV))[0][:, :16]

    res = {}
    for (Ls, Le) in SPANS:
        attn_sum = sum(A[L2] for L2 in range(Ls+1, Le+1))
        Qt = quad_feats(C[Ls], P_true); Qr = quad_feats(C[Ls], P_rot)
        base = [C[Ls], TF, cpool(C[Ls])]
        conds = {'direct': base,
                 'plus_attn': base + [attn_sum],
                 'plus_quad': base + [Qt],
                 'plus_attn_quad': base + [attn_sum, Qt],
                 'plus_attn_quadrot': base + [attn_sum, Qr]}
        Y = C[Le]
        row = {}
        for nm, parts in conds.items():
            Xc = torch.cat(parts, -1)
            f = fit_linear(flat(Xc, trm), flat(Y, trm))
            row[nm] = cosR2(f(flat(Xc, ev)), flat(Y, ev))
        res[f'L{Ls}_to_L{Le}'] = row
        print(f"L{Ls}->L{Le}: " + " | ".join(f"{nm} {v}" for nm, v in row.items()), flush=True)

    d = res['L5_to_L14']
    out = {'spans': res, 'reference_1124': {'direct': [0.638, 0.534], 'plus_attn': [0.738, 0.705]}}
    out['pred_a_quadratic_closes'] = bool(d['plus_attn_quad'][1] >= 0.85
                                          and d['plus_attn_quad'][1] - d['plus_attn_quadrot'][1] >= 0.05)
    out['pred_b_off_coordinate'] = bool(d['plus_attn_quad'][1] < 0.78)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a quadratic-closes {out['pred_a_quadratic_closes']} | pred_b off-coordinate {out['pred_b_off_coordinate']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
