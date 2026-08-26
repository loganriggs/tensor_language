# plank_corr_mlp0: TRAINED CORRECTION FOR THE mlp0 PLANK (the S1489 move —
# the one method that has been reliably additive — applied to the board's top
# items). Stand-in = tier-2000 token table + ridge on [attn0] + a TRAINABLE
# rank-64 correction on the same inputs, trained vs full-model CE with ONLY
# mlp0 replaced (solo context, everything else live). 400 steps, Adam 3e-3,
# correction init zero (the regime where CE-training works, S1533).
#
# Registered predictions:
#   pred_a corrected fid_opt >= 0.945 (plank alone: .932).
#   pred_b held-out CE gain from the correction >= 0.01.
#   pred_c the gain generalizes: skip=5000 gain >= .5x the skip=7000 gain.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'plank_corr_mlp0_results.json'
NR = 960; NTR = 480; STEPS = 400
H = m.transformer.h
ML = 0
CAP = {'on': False, 'a': None, 'm0': None}
STATE = {'mode': None, 'T': None, 'W': None, 'xm': None, 'ym': None,
         'corr': None, 'toks': None}


def a_hook(mod, args, output):
    CAP['a'] = output.float()
    return None


def m0_hook(mod, args, output):
    if ML == 1:
        CAP['m0'] = output.float()
    return None


def stand_hook(mod, args, output):
    if STATE['mode'] is None:
        return None
    B = output.shape[0]
    if ML == 1:
        X = torch.cat([CAP['a'], CAP['m0']], -1).reshape(-1, 2 * D)
    else:
        X = CAP['a'].reshape(-1, D)
    st = STATE['T'][STATE['toks']].view(B, T, D) \
        + (STATE['ym'] + (X - STATE['xm']) @ STATE['W']).view(B, T, D)
    if STATE['corr'] is not None:
        b_, U_, V_ = STATE['corr']
        st = st + (b_ + ((X - STATE['xm']) @ V_) @ U_.T).view(B, T, D)
    return st.to(output.dtype)


def fwd(idx):
    STATE['toks'] = idx.reshape(-1)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    for p in m.parameters():
        p.requires_grad_(False)
    TRR = cl.fineweb_rows(NTR, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    EV5 = cl.fineweb_rows(NR, skip=5000)[:, :T + 1].contiguous()
    hks = [H[ML].attn.c_proj.register_forward_hook(a_hook),
           H[ML].mlp.register_forward_hook(stand_hook)]
    if ML == 1:
        hks.append(H[0].mlp.register_forward_hook(m0_hook))

    # fit the plank on clean captures
    with torch.no_grad():
        outs = []; ains = []; m0s = []; toks = []
        capm = H[ML].mlp.register_forward_hook(
            lambda mo, a, o: outs.append(o.detach().float().cpu()))
        STATE['mode'] = None
        for i in range(0, NTR, 8):
            idx = TRR[i:i + 8, :-1].to(DEV).contiguous()
            fwd(idx)
            ains.append(CAP['a'].cpu())
            if ML == 1:
                m0s.append(CAP['m0'].cpu())
            toks.append(TRR[i:i + 8, :-1].reshape(-1))
        capm.remove()
        Y = torch.cat(outs).reshape(-1, D)
        A = torch.cat(ains).reshape(-1, D)
        TK = torch.cat(toks)
        if ML == 1:
            X = torch.cat([torch.cat(ains), torch.cat(m0s)], -1).reshape(-1, 2 * D)
        else:
            X = A
        tsum = torch.zeros(50257, D); tcnt = torch.zeros(50257)
        tsum.index_add_(0, TK, Y); tcnt.index_add_(0, TK, torch.ones(TK.shape[0]))
        gm = Y.mean(0)
        TAB = torch.where(tcnt.unsqueeze(1) > 0,
                          tsum / tcnt.clamp_min(1).unsqueeze(1), gm.unsqueeze(0))
        Tc = (TAB - gm).to(DEV)
        U_, S_, V_ = torch.svd_lowrank(Tc, q=96, niter=4)
        base = gm.to(DEV) + (U_[:, :64] * S_[:64]) @ V_[:, :64].T
        keep = tcnt.argsort(descending=True)[:2000]
        TIER = base.clone(); TIER[keep] = TAB[keep].to(DEV)
        RESY = (Y - TAB[TK]).to(DEV)
        Xg = X.to(DEV)
        xm_ = Xg.mean(0); ym_ = RESY.mean(0)
        Xc = Xg - xm_
        XtX = Xc.T @ Xc
        lam = 0.01 * float(torch.diagonal(XtX).mean())
        W = torch.linalg.solve(XtX + lam * torch.eye(X.shape[1], device=DEV),
                               Xc.T @ (RESY - ym_))
        STATE.update({'T': TIER, 'W': W, 'xm': xm_, 'ym': ym_})
    print("plank fit", flush=True)

    def ce_eval(rows):
        s_ = 0.0; n_ = 0
        with torch.no_grad():
            for i in range(0, NR, 8):
                bb = rows[i:i + 8].to(DEV)
                idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
                lo = fwd(idx).float()
                ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]),
                                     tg.reshape(-1),
                                     reduction='none').view(tg.shape)
                mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
                s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    STATE['mode'] = None
    clean = ce_eval(EVR)
    STATE['mode'] = 'on'
    ce_plank = ce_eval(EVR)
    print(f"clean {clean:.4f} plank {ce_plank:.4f}", flush=True)

    IND = 2 * D if ML == 1 else D
    b_ = torch.nn.Parameter(torch.zeros(D, device=DEV))
    U_p = torch.nn.Parameter(torch.zeros(D, 64, device=DEV))
    V_p = torch.nn.Parameter(torch.randn(IND, 64, device=DEV) * 0.001)
    opt = torch.optim.Adam([b_, U_p, V_p], lr=3e-3)
    g = torch.Generator().manual_seed(5)
    STATE['corr'] = (b_, U_p, V_p)
    for step in range(STEPS):
        sel = torch.randint(0, NTR, (8,), generator=g)
        bb = TRR[sel].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
        lo = fwd(idx).float()
        ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                             reduction='none').view(tg.shape)
        loss = ce[:, 64:].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"  step {step} loss {float(loss):.4f}", flush=True)
    STATE['corr'] = tuple(t.detach() for t in (b_, U_p, V_p))
    ce_corr = ce_eval(EVR)
    ce_corr5 = ce_eval(EV5)
    STATE['corr'] = None
    ce_plank5 = ce_eval(EV5)
    STATE['mode'] = None
    clean5 = ce_eval(EV5)

    sw = json.load(open(PT + 'optimal_ablation_all_results.json'))['results'][f'mlp{ML}']
    fid_p = (sw['ce_opt'] - ce_plank) / max(sw['ce_opt'] - clean, 1e-6)
    fid_c = (sw['ce_opt'] - ce_corr) / max(sw['ce_opt'] - clean, 1e-6)
    gain = ce_plank - ce_corr
    gain5 = ce_plank5 - ce_corr5
    pa = fid_c >= 0.945
    pb = gain >= 0.01
    pc = gain5 >= 0.5 * gain
    out = {'clean': round(clean, 4), 'ce_plank': round(ce_plank, 4),
           'ce_corrected': round(ce_corr, 4),
           'fid_plank': round(fid_p, 4), 'fid_corrected': round(fid_c, 4),
           'gain_7000': round(gain, 4), 'gain_5000': round(gain5, 4),
           'pred_a_fid': bool(pa), 'pred_b_gain': bool(pb),
           'pred_c_generalizes': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(out)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
