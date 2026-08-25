# mlp4_weight_tensor: READ THE LITERAL COMPUTATION FROM THE WEIGHTS (user directive:
# "which features from previous layers interact, solely from the weights"). mlp4 is
# PURE bilinear (gated=False, verified): out = Down(Left(x) ⊙ Right(x)) + b — the whole
# layer IS the third-order tensor T[k,:,:] = Left^T diag(Down[k]) Right (symmetrized).
# Restrict it to NAMED upstream features: f = top-64 PCs each of the mlp0/mlp2/mlp3
# contributions as seen at mlp4's input (192 features), output side g = top-64 PCs of
# mlp4's output. Interaction tensor I[a,i,j] = Σ_h (g_a·Down_h)(Left_h·f_i)(Right_h·f_j).
# Assumption registered: rms_norm denominator treated as constant (= its fit-row mean) —
# the §105-style calibration; features are unit vectors in the NORMED input space.
# Causal check: projected-bilinear stand-in x̂ = m̄ + Π(x_norm − m̄) (Π = 192-feature
# projector), out = mlp4(x̂ · scale), held-out CE recovery vs the live/mean stake.
#
# Registered predictions:
#   pred_a >= 50% of squared interaction mass lies in blocks involving mlp0 features.
#   pred_b interactions are SPARSE: the top 5% of |I| entries carry >= 50% of the
#          squared mass.
#   pred_c the 192-feature projected bilinear stand-in recovers >= .55 of mlp4's
#          held-out stake (the restricted tensor is causally sufficient).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp4_weight_tensor_results.json'
NFIT = 480; NEV = 960
R = 64
H = m.transformer.h
STAND = {'mode': None, 'proj': None, 'mean_in': None}
CAP = {'on': False, 'store': None}


def cap_hook_for(name):
    def hook(mod, args, output):
        if CAP['on']:
            if name == 'm4in':
                CAP['store'][name].append(args[0].detach().float().cpu())
            else:
                CAP['store'][name].append(output.detach().float().cpu())
        return None
    return hook


def mlp4_pre_hook(mod, args):
    if STAND['mode'] is None:
        return None
    x = args[0]
    xc = x.float() - STAND['mean_in']
    xproj = STAND['mean_in'] + xc @ STAND['proj']
    return (xproj.to(x.dtype),)


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

    hooks = [H[L].mlp.register_forward_hook(cap_hook_for(n))
             for L, n in ((0, 'm0'), (2, 'm2'), (3, 'm3'), (4, 'm4'))]
    hooks.append(H[4].mlp.register_forward_pre_hook(
        lambda mod, args: CAP['store']['m4in'].append(args[0].detach().float().cpu())
        if CAP['on'] else None))
    CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'm2', 'm3', 'm4', 'm4in')}
    for i in range(0, NFIT, 8):
        fwd(FITR[i:i + 8, :-1].to(DEV).contiguous())
    CAP['on'] = False
    FT = {n: torch.cat(v) for n, v in CAP['store'].items()}
    print("fit capture done", flush=True)

    # rms calibration: components live pre-norm; the normed input is x/rms. Scale each
    # component contribution by 1/mean_rms to express features in normed coordinates.
    # mean_rms from the ratio of pre-norm stream to normed input is not directly
    # captured; use the normed-input variance structure directly for the OUTPUT basis
    # and scale component features into normed space via the mean rms of the L4 stream.
    # Approximation registered in the docstring.
    # Estimate rms: capture pre-norm stream indirectly — mlp4 input IS the normed
    # stream; component outputs are pre-norm. Estimate scale s by regressing:
    # normed_input ≈ (sum of contributions + rest)/rms; use s = 1/median rms proxy via
    # norm matching: scale each feature basis to unit norm in normed space (PCs of the
    # SCALED contributions). Since PCs are direction-only, the per-component scale
    # cancels — only the PROJECTOR matters, so no rms estimate is needed for Π.
    feats = []
    fam = []
    for n in ('m0', 'm2', 'm3'):
        Xc = FT[n].reshape(-1, D)
        Xc = Xc - Xc.mean(0)
        # PCs via covariance eigh on GPU
        C = (Xc.T @ Xc).to(DEV) / Xc.shape[0]
        evals, evecs = torch.linalg.eigh(C)
        Fb = evecs[:, -R:].flip(-1)                      # [D, R]
        feats.append(Fb)
        fam += [n] * R
        print(f"{n} PCs done (top eval {float(evals[-1]):.3f})", flush=True)
    Fall = torch.cat(feats, 1)                            # [D, 192]
    # orthonormalize the union (QR) so the projector is well-defined
    Q, _ = torch.linalg.qr(Fall)
    PROJ = Q @ Q.T                                        # [D, D]

    Yc = FT['m4'].reshape(-1, D)
    Yc = Yc - Yc.mean(0)
    Cy = (Yc.T @ Yc).to(DEV) / Yc.shape[0]
    eva, eve = torch.linalg.eigh(Cy)
    G = eve[:, -R:].flip(-1)                              # [D, R] output basis
    m4in_mean = FT['m4in'].reshape(-1, D).mean(0).to(DEV)

    mlp = H[4].mlp
    Lw = mlp.Left.weight.float()                          # [4608, D]
    Rw = mlp.Right.weight.float()
    Dw = mlp.Down.weight.float()                          # [D, 4608]
    A = Lw @ Fall                                         # [4608, 192]
    B = Rw @ Fall
    Qo = (G.T @ Dw)                                       # [R, 4608]
    I = torch.einsum('ah,hi,hj->aij', Qo, A, B)           # [R, 192, 192]
    I = 0.5 * (I + I.transpose(1, 2))
    print("interaction tensor built", flush=True)

    M2 = (I ** 2).sum(0)                                  # [192, 192] squared mass
    total = float(M2.sum())
    blocks = {}
    for bi, ni in ((0, 'm0'), (1, 'm2'), (2, 'm3')):
        for bj, nj in ((0, 'm0'), (1, 'm2'), (2, 'm3')):
            if bj < bi:
                continue
            sl_i = slice(bi * R, (bi + 1) * R); sl_j = slice(bj * R, (bj + 1) * R)
            mass = float(M2[sl_i, sl_j].sum())
            if bi != bj:
                mass += float(M2[sl_j, sl_i].sum())
            blocks[f'{ni}x{nj}'] = round(mass / total, 4)
    print("block mass:", blocks, flush=True)
    flat = M2.flatten()
    k5 = max(1, int(0.05 * flat.numel()))
    top5_mass = float(flat.topk(k5).values.sum()) / total
    m0_mass = blocks['m0xm0'] + blocks['m0xm2'] + blocks['m0xm3']

    for hk in hooks:
        hk.remove()
    ph = H[4].mlp.register_forward_pre_hook(mlp4_pre_hook)

    def ce_run(mode):
        STAND['mode'] = mode
        STAND['proj'] = PROJ; STAND['mean_in'] = m4in_mean
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            m_ = torch.ones_like(tg, dtype=torch.bool); m_[:, :64] = False
            s_ += float(ce[m_].sum()); n_ += int(m_.sum())
        STAND['mode'] = None
        return s_ / max(n_, 1)

    live = ce_run(None)
    projb = ce_run('proj')
    ph.remove()
    # mean arm: reuse the known mean-ablation via output hook
    gmean = FT['m4'].reshape(-1, D).mean(0).to(DEV)
    oh = H[4].mlp.register_forward_hook(
        lambda mod, a, o: gmean.expand_as(o).to(o.dtype))
    meanc = ce_run(None)
    oh.remove()
    stake = meanc - live
    rec = (meanc - projb) / max(stake, 1e-6)
    print(f"live {live:.4f} mean {meanc:.4f} proj {projb:.4f} rec {rec:.4f}", flush=True)

    top_pairs = []
    fl = M2.clone(); fl = torch.triu(fl)
    v, ix = fl.flatten().topk(20)
    names = fam
    for val, ii in zip(v.tolist(), ix.tolist()):
        i2, j2 = ii // 192, ii % 192
        top_pairs.append([f'{names[i2]}#{i2 % R}', f'{names[j2]}#{j2 % R}',
                          round(val / total, 4)])

    pa = m0_mass >= 0.50
    pb = top5_mass >= 0.50
    pc = rec >= 0.55
    out = {'block_mass': blocks, 'm0_involved_mass': round(m0_mass, 4),
           'top5pct_mass': round(top5_mass, 4), 'top_pairs': top_pairs,
           'ce': {'live': round(live, 4), 'mean': round(meanc, 4),
                  'proj_bilinear': round(projb, 4)},
           'stake': round(stake, 4), 'proj_recovery': round(rec, 4),
           'pred_a_m0_mass_50': bool(pa), 'pred_b_sparse_50': bool(pb),
           'pred_c_causal_55': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} ({m0_mass:.3f}) | pred_b {pb} ({top5_mass:.3f}) | pred_c {pc} ({rec:.3f})")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
