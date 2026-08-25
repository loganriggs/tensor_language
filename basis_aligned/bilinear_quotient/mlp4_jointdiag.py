# mlp4_jointdiag: JOINTLY-DEFINED FEATURES VIA SIMULTANEOUS DIAGONALIZATION (user
# directive 2026-08-25: decompositions should be defined by DOWNSTREAM computation,
# jointly — not per-module variance). Take mlp4's interaction matrices B_a (a = top-64
# output dirs, restricted to the 128-dim [attn4-native-64; mlp3-native-64] subspace)
# and find ONE rotation Q minimizing total off-diagonal mass across all 64 (Jacobi
# sweeps). The resulting directions are features defined by what mlp4 MULTIPLIES.
# Metrics: diagonal-mass fraction before/after; causal check: squares-only stand-in
# (128 features = q_i^2 coords in the joint-diag basis) on the lin5 residual vs a
# matched-capacity (F=128) random-pair control. Fit skip=80, EVAL HELD OUT skip=7000.
#
# Registered predictions:
#   pred_a joint diagonalization concentrates: diagonal mass fraction >= 2x the
#          unrotated basis's.
#   pred_b the joint-diag squares stand-in gains >= .02 recovery over lin5 (vs the
#          .004 native-squares gain of §1432).
#   pred_c joint-diag squares beat the matched-F random-pair control.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp4_jointdiag_results.json'
NFIT = 960; NEV = 960
R = 64
H = m.transformer.h
STAND = {'mode': None, 'tensor': None}
CAP = {'on': False, 'store': None}


def cap_hook_for(name):
    def hook(mod, args, output):
        if CAP['on']:
            CAP['store'][name].append(output.detach().float().cpu())
        return None
    return hook


def m4_hook(mod, args, output):
    if STAND['mode'] is None:
        return None
    return STAND['tensor'].to(output.dtype)


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FITR = cl.fineweb_rows(NFIT, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NEV, skip=7000)[:, :T + 1].contiguous()

    # ---- native input bases (attn4 c_proj out; mlp3 Down) and mlp4 tensor slices
    cp4 = H[4].attn.c_proj.weight.float().to(DEV)
    Ua4 = torch.linalg.svd(cp4, full_matrices=False)[0][:, :R]
    Dw3 = H[3].mlp.Down.weight.float().to(DEV)
    Um3 = torch.linalg.svd(Dw3, full_matrices=False)[0][:, :R]
    S = torch.cat([Ua4, Um3], 1)                                # [D, 128]
    L4 = H[4].mlp.Left.weight.float().to(DEV)
    R4 = H[4].mlp.Right.weight.float().to(DEV)
    D4 = H[4].mlp.Down.weight.float().to(DEV)
    Ug4 = torch.linalg.svd(D4, full_matrices=False)[0][:, :R]
    A = L4 @ S; Bm = R4 @ S                                     # [4608, 128]
    Q4 = Ug4.T @ D4                                             # [R, 4608]
    Bs = torch.einsum('ah,hi,hj->aij', Q4, A, Bm)               # [R, 128, 128]
    Bs = 0.5 * (Bs + Bs.transpose(1, 2))

    def diag_frac(Bt):
        d = (torch.diagonal(Bt, dim1=1, dim2=2) ** 2).sum()
        return float(d / (Bt ** 2).sum())

    f0 = diag_frac(Bs)
    print(f"diag frac before: {f0:.4f}", flush=True)

    # ---- Jacobi joint diagonalization (Cardoso-style sweeps)
    n = 128
    Qrot = torch.eye(n, device=DEV)
    Bw = Bs.clone()
    for sweep in range(6):
        off_before = float((Bw ** 2).sum() - (torch.diagonal(Bw, dim1=1, dim2=2) ** 2).sum())
        for i in range(n - 1):
            for j in range(i + 1, n):
                bii = Bw[:, i, i]; bjj = Bw[:, j, j]; bij = Bw[:, i, j]
                # Cardoso closed-form angle for symmetric joint diagonalization
                g1 = (bii - bjj); g2 = 2 * bij
                ton = float((g1 * g1 - g2 * g2).sum())
                toff = float(2 * (g1 * g2).sum())
                theta = 0.25 * torch.atan2(torch.tensor(toff), torch.tensor(ton + 1e-30))
                c = float(torch.cos(theta)); s_ = float(torch.sin(theta))
                if abs(s_) < 1e-9:
                    continue
                Gi = Bw[:, :, i].clone(); Gj = Bw[:, :, j].clone()
                Bw[:, :, i] = c * Gi + s_ * Gj
                Bw[:, :, j] = -s_ * Gi + c * Gj
                Ri_ = Bw[:, i, :].clone(); Rj = Bw[:, j, :].clone()
                Bw[:, i, :] = c * Ri_ + s_ * Rj
                Bw[:, j, :] = -s_ * Ri_ + c * Rj
                qi = Qrot[:, i].clone(); qj = Qrot[:, j].clone()
                Qrot[:, i] = c * qi + s_ * qj
                Qrot[:, j] = -s_ * qi + c * qj
        f_now = diag_frac(Bw)
        off_now = float((Bw ** 2).sum() - (torch.diagonal(Bw, dim1=1, dim2=2) ** 2).sum())
        print(f"sweep {sweep + 1}: diag frac {f_now:.4f}", flush=True)
        if off_before - off_now < 1e-6 * off_before:
            break
    f1 = diag_frac(Bw)
    FEATS = (S @ Qrot)                                          # [D->? no: 1152x128 dirs]

    # ---- causal check: capture, ridge lin-all baseline, squares in joint basis
    hooks = [H[L].mlp.register_forward_hook(cap_hook_for(nm))
             for L, nm in ((0, 'm0'), (1, 'm1'), (2, 'm2'), (3, 'm3'), (4, 'm4'))]
    hooks.append(H[4].attn.c_proj.register_forward_hook(cap_hook_for('a4')))
    CAP['on'] = True; CAP['store'] = {nm: [] for nm in ('m0', 'm1', 'm2', 'm3', 'm4', 'a4')}
    for i in range(0, NFIT, 8):
        fwd(FITR[i:i + 8, :-1].to(DEV).contiguous())
    CAP['on'] = False
    FT = {nm: torch.cat(v) for nm, v in CAP['store'].items()}
    print("fit capture done", flush=True)

    X5f = torch.cat([FT['a4'], FT['m0'], FT['m1'], FT['m2'], FT['m3']], -1).reshape(-1, 5 * D)
    Yf = FT['m4'].reshape(-1, D)

    def ridge(X, Y, lamf=0.01):
        Xg = X.to(DEV); Yg = Y.to(DEV)
        xm = Xg.mean(0); ym = Yg.mean(0)
        Xc = Xg - xm; Yc = Yg - ym
        XtX = Xc.T @ Xc
        lam = lamf * float(torch.diagonal(XtX).mean())
        W = torch.linalg.solve(XtX + lam * torch.eye(X.shape[1], device=DEV), Xc.T @ Yc)
        return W, xm, ym
    W5, xm5, ym5 = ridge(X5f, Yf)
    gmean = Yf.mean(0).to(DEV)

    # joint-space coordinates: z = ([a4; m3-part of stream input]) — features act on the
    # residual-sum input; approximate with component outputs projected by FEATS' two halves
    Fa4 = FEATS[:, :]                                           # rotated dirs in D-space? No:
    # FEATS columns live in D-space via S@Qrot; z = (a4_out + m3_out ... ) — the tensor
    # was built on S-projected input; use x_in ≈ a4 + m3 contributions (registered approx)
    def zcoords(Ea4, Em3):
        Xin = (Ea4 + Em3).to(DEV)                               # [N, D]
        return Xin @ FEATS                                      # [N, 128]

    def quadfit(feat_fn, Fn, seed):
        gq = torch.Generator().manual_seed(seed)
        pi = torch.randint(0, 128, (Fn,), generator=gq).to(DEV)
        pj = torch.randint(0, 128, (Fn,), generator=gq).to(DEV)
        XtX = torch.zeros(Fn, Fn, device=DEV); XtY = torch.zeros(Fn, D, device=DEV)
        xs = torch.zeros(Fn, device=DEV); ys = torch.zeros(D, device=DEV); nn = 0
        A4 = FT['a4'].reshape(-1, D); M3 = FT['m3'].reshape(-1, D)
        NPOS = A4.shape[0]
        for i in range(0, NPOS, 16384):
            z = feat_fn(A4[i:i + 16384], M3[i:i + 16384])
            q = z[:, pi] * z[:, pj] if Fn != 128 else z * z
            resid = (Yf[i:i + 16384].to(DEV) - (ym5 + (X5f[i:i + 16384].to(DEV) - xm5) @ W5))
            XtX += q.T @ q; XtY += q.T @ resid
            xs += q.sum(0); ys += resid.sum(0); nn += q.shape[0]
        XtX -= torch.outer(xs, xs) / nn; XtY -= torch.outer(xs, ys) / nn
        lam = 0.05 * float(torch.diagonal(XtX).mean())
        Wq = torch.linalg.solve(XtX + lam * torch.eye(Fn, device=DEV), XtY)
        return (Wq, xs / nn, ys / nn, pi, pj)

    QD = quadfit(zcoords, 128, 7)          # squares (Fn==128 -> z*z path)
    gq2 = torch.Generator().manual_seed(77)
    RND = torch.linalg.qr(torch.randn(D, 128, generator=gq2))[0].to(DEV)
    def zrand(Ea4, Em3):
        return ((Ea4 + Em3).to(DEV)) @ RND
    QR = quadfit(zrand, 129, 9)            # matched-capacity random pairs (F=129~128)

    mh = H[4].mlp.register_forward_hook(m4_hook)

    def ce_run(mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode is None:
                STAND['mode'] = None
            else:
                CAP['on'] = True
                CAP['store'] = {nm: [] for nm in ('m0', 'm1', 'm2', 'm3', 'm4', 'a4')}
                STAND['mode'] = None
                fwd(idx)
                CAP['on'] = False
                E = {nm: torch.cat(v) for nm, v in CAP['store'].items()}
                Bz = idx.shape[0]
                Xe5 = torch.cat([E['a4'], E['m0'], E['m1'], E['m2'], E['m3']],
                                -1).reshape(-1, 5 * D).to(DEV)
                lin5p = ym5 + (Xe5 - xm5) @ W5
                if mode == 'mean':
                    st = gmean.expand(Bz, T, D)
                elif mode == 'lin5':
                    st = lin5p.view(Bz, T, D)
                else:
                    A4e = E['a4'].reshape(-1, D); M3e = E['m3'].reshape(-1, D)
                    Wq, qm, rm, pi, pj = QD if mode == 'jd_squares' else QR
                    z = zcoords(A4e, M3e) if mode == 'jd_squares' else zrand(A4e, M3e)
                    q = z * z if mode == 'jd_squares' else z[:, pi] * z[:, pj]
                    st = (lin5p + rm + (q - qm) @ Wq).view(Bz, T, D)
                STAND['mode'] = 'on'
                STAND['tensor'] = st
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        STAND['mode'] = None
        return s_ / max(n_, 1)

    res = {}
    for mode in (None, 'mean', 'lin5', 'jd_squares', 'rand_pairs'):
        key = str(mode)
        res[key] = round(ce_run(mode if mode != 'rand_pairs' else 'rand_pairs'), 4)
        print(f"{mode}: {res[key]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res,
                   'diag_frac': {'before': round(f0, 4), 'after': round(f1, 4)}},
                  open(OUT, 'w'), indent=1)
    mh.remove()
    for hk in hooks:
        hk.remove()

    stake = res['mean'] - res['None']
    rec = lambda a: (res['mean'] - res[a]) / max(stake, 1e-6)
    r5, rj, rr = rec('lin5'), rec('jd_squares'), rec('rand_pairs')
    pa = f1 >= 2.0 * max(f0, 1e-6)
    pb = (rj - r5) >= 0.02
    pc = rj > rr
    out = {'diag_frac_before': round(f0, 4), 'diag_frac_after': round(f1, 4),
           'ce': res, 'stake': round(stake, 4),
           'recovery': {'lin5': round(r5, 4), 'jd_squares': round(rj, 4),
                        'rand_pairs': round(rr, 4)},
           'pred_a_diag_2x': bool(pa), 'pred_b_jd_gains_02': bool(pb),
           'pred_c_jd_beats_rand': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"diag {f0:.4f}->{f1:.4f} | lin5 {r5:.3f} jd {rj:.3f} rand {rr:.3f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
