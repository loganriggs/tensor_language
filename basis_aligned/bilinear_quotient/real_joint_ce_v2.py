"""REAL JOINT CE v2 -- RECONSTRUCTION-ANCHORED (fixes 758's collapse). 758 showed
pure-CE training of the substituted SAEs walks away from weight-faithfulness
(reconstruction R2 -> negative, eval CE-recovery 0.92 -> 0.59). Fix: keep the MSE
action-reconstruction term as an ANCHOR:
    loss = CE + lam_rec*(MSE(Down_0) + MSE(Left_1)) + lam_e*||norm(E_L1)@norm(D_D0)||_1
The anchor holds R2 up (faithful), CE nudges toward loss-relevance, the edge penalty
sparsifies the wiring. Warm-start from the independent MSE fit; log curves.

Conditions (all warm-started from indep):
  anchored     -- CE + MSE anchor + edge   (THE FIX)
  no_anchor    -- CE + edge, lam_rec=0      (reproduces 758 collapse; control)
Report eval CE-recovery, per-layer R2, in-degree over steps + figure.

REGISTERED PREDICTIONS:
  (0) SANITY: indep warm-start CE-rec ~0.9;
  (a) ANCHOR HOLDS FAITHFULNESS + SPARSIFIES: anchored keeps R2_Down0 > 0.6 and
      eval CE-recovery within 0.08 of indep THROUGHOUT, while in-degree drops
      >= 40% from the edge penalty -- a faithful, CE-aware, SPARSE-wired joint;
  (b) CONTROL: no_anchor reproduces 758 (R2 -> < 0, CE-rec falls well below indep);
  NULL: n/a (control IS the null)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/rspd'); sys.path.insert(0, '/workspace/tensor_language')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'real_joint_ce_v2_results.json'; FIG = PT + 'real_joint_ce_v2.png'
NFIT = 48; NEVAL = 24; P = 512; K = 32; STEPS = 500; EVAL_EVERY = 25; LAM_E = 0.03; LAM_REC = 1.0
SUB = {'d0': None, 'l1': None}
BLUE, RED, GREEN, ORANGE = '#3987e5', '#e34948', '#2e8b57', '#e08a1e'


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def hook_d0(mo, i_, o_):
    return o_ if SUB['d0'] is None else SUB['d0'](i_[0].reshape(-1, HID)).reshape(o_.shape).to(o_.dtype)


def hook_l1(mo, i_, o_):
    return o_ if SUB['l1'] is None else SUB['l1'](i_[0].reshape(-1, D)).reshape(o_.shape).to(o_.dtype)


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def ce_on(rows, n):
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1)))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


@torch.no_grad()
def capture(rows, n, module, dim):
    cap = []
    h = module.register_forward_hook(lambda mo, i_, o_: cap.append(i_[0].detach().float().reshape(-1, dim)))
    for i in range(0, n, 4): forward_logits(rows[i:i+4, :257].to(DEV)[:, :-1].contiguous())
    h.remove(); return torch.cat(cap, 0)


def sub_fn(Dm, Em, b): return lambda g: topk(g @ Em.T, K) @ Dm.T + b
def r2(recon, Y): return float(1 - ((Y-recon)**2).sum()/((Y-Y.mean(0))**2).sum())


def in_degree(EL, D0, thr=0.2):
    C = F.normalize(EL, dim=1) @ F.normalize(D0, dim=0)
    return float((C.abs() > thr*C.abs().max(0).values.clamp_min(1e-9)).float().sum(0).mean())


def train_indep(Xin, Ytrue, din, dout, seed):
    torch.manual_seed(seed)
    Dm = (torch.randn(dout, P, device=DEV)/np.sqrt(dout)).requires_grad_(True)
    Em = (torch.randn(P, din, device=DEV)/np.sqrt(din)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True); opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(600):
        z = topk(Xin @ Em.T, K); loss = F.mse_loss(z @ Dm.T + b, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
    return Dm.detach(), Em.detach(), b.detach()


def run(fit, nfit, g0, r1, Y0, YL, lam_rec, lam_e, init0, initL, ce_abl, ben, ev):
    D0 = init0[0].clone().requires_grad_(True); E0 = init0[1].clone().requires_grad_(True); b0 = init0[2].clone().requires_grad_(True)
    DL = initL[0].clone().requires_grad_(True); EL = initL[1].clone().requires_grad_(True); bL = initL[2].clone().requires_grad_(True)
    opt = torch.optim.Adam([D0, E0, b0, DL, EL, bL], lr=1e-3)
    cur = {'step': [], 'ce_rec': [], 'r2_d0': [], 'r2_l1': [], 'in_deg': []}
    for s in range(STEPS + 1):
        if s % EVAL_EVERY == 0:
            SUB['d0'] = sub_fn(D0.detach(), E0.detach(), b0.detach()); SUB['l1'] = sub_fn(DL.detach(), EL.detach(), bL.detach())
            with torch.no_grad():
                ce = ce_on(ev, NEVAL)
                rec0 = topk(g0 @ E0.T, K) @ D0.T + b0; recL = topk(r1 @ EL.T, K) @ DL.T + bL
            SUB['d0'] = SUB['l1'] = None
            cur['step'].append(s); cur['ce_rec'].append(round(float((ce_abl-ce)/max(ben,1e-6)), 4))
            cur['r2_d0'].append(round(r2(rec0, Y0), 4)); cur['r2_l1'].append(round(r2(recL, YL), 4))
            cur['in_deg'].append(round(in_degree(EL.detach(), D0.detach()), 1))
        if s == STEPS: break
        SUB['d0'] = sub_fn(D0, E0, b0); SUB['l1'] = sub_fn(DL, EL, bL)
        bb = fit[(s % (nfit//4))*4:(s % (nfit//4))*4+4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        ce = F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1))
        mse = F.mse_loss(topk(g0 @ E0.T, K) @ D0.T + b0, Y0) + F.mse_loss(topk(r1 @ EL.T, K) @ DL.T + bL, YL)
        C = F.normalize(EL, dim=1) @ F.normalize(D0, dim=0)
        (ce + lam_rec*mse + lam_e*C.abs().mean()).backward(); opt.step(); opt.zero_grad()
        SUB['d0'] = SUB['l1'] = None
    return cur


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL); fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    W0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV)
    WL = m.transformer.h[1].mlp.Left.weight.data.float().to(DEV)
    h0 = m.transformer.h[0].mlp.Down.register_forward_hook(hook_d0)
    hL = m.transformer.h[1].mlp.Left.register_forward_hook(hook_l1)

    g0 = capture(fit, NFIT, m.transformer.h[0].mlp.Down, HID); r1 = capture(fit, NFIT, m.transformer.h[1].mlp.Left, D)
    Y0, YL = g0 @ W0.T, r1 @ WL.T

    SUB['d0'] = SUB['l1'] = None
    with torch.no_grad(): ce_full = ce_on(ev, NEVAL)
    SUB['d0'] = lambda g: torch.zeros(g.shape[0], D, device=DEV); SUB['l1'] = lambda g: torch.zeros(g.shape[0], HID, device=DEV)
    with torch.no_grad(): ce_abl = ce_on(ev, NEVAL)
    SUB['d0'] = SUB['l1'] = None; ben = ce_abl - ce_full
    with torch.enable_grad():
        init0 = train_indep(g0, Y0, HID, D, 1); initL = train_indep(r1, YL, D, HID, 3)
    SUB['d0'] = sub_fn(*init0); SUB['l1'] = sub_fn(*initL)
    with torch.no_grad(): indep_rec = float((ce_abl - ce_on(ev, NEVAL))/max(ben, 1e-6))
    SUB['d0'] = SUB['l1'] = None
    print(f'CE_full {ce_full:.3f} benefit {ben:.3f} | indep CE-rec {indep_rec:.3f}', flush=True)

    with torch.enable_grad():
        anc = run(fit, NFIT, g0, r1, Y0, YL, LAM_REC, LAM_E, init0, initL, ce_abl, ben, ev)
        print(f'anchored  end CE-rec {anc["ce_rec"][-1]} R2d0 {anc["r2_d0"][-1]} in-deg {anc["in_deg"][-1]}', flush=True)
        noa = run(fit, NFIT, g0, r1, Y0, YL, 0.0, LAM_E, init0, initL, ce_abl, ben, ev)
        print(f'no_anchor end CE-rec {noa["ce_rec"][-1]} R2d0 {noa["r2_d0"][-1]} in-deg {noa["in_deg"][-1]}', flush=True)
    h0.remove(); hL.remove()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6)); fig.patch.set_facecolor(SURFACE)
    for ax in axs: ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.6, zorder=0)
    st = anc['step']
    axs[0].axhline(indep_rec, color=MUTED, lw=1.5, ls='--', label=f'indep {indep_rec:.2f}')
    axs[0].plot(st, anc['ce_rec'], '-o', color=GREEN, ms=4, label='anchored (CE+MSE+edge)')
    axs[0].plot(st, noa['ce_rec'], '-o', color=RED, ms=4, label='no anchor (CE+edge)')
    axs[0].set_title('eval CE-recovery vs step', color=INK, fontsize=12, loc='left')
    axs[0].set_xlabel('step'); axs[0].set_ylabel('CE-recovery'); axs[0].legend(fontsize=9)
    axs[1].plot(st, anc['r2_d0'], '-o', color=GREEN, ms=4, label='anchored Down_0 R²')
    axs[1].plot(st, anc['r2_l1'], '-s', color=GREEN, ms=4, mfc='none', label='anchored Left_1 R²')
    axs[1].plot(st, noa['r2_d0'], '-o', color=RED, ms=4, label='no-anchor Down_0 R²')
    axs[1].plot(st, noa['r2_l1'], '-s', color=RED, ms=4, mfc='none', label='no-anchor Left_1 R²')
    axs[1].axhline(0, color=SECONDARY, lw=0.8)
    axs[1].set_title('per-layer reconstruction R² (anchor holds it up)', color=INK, fontsize=12, loc='left')
    axs[1].set_xlabel('step'); axs[1].set_ylabel('action R²'); axs[1].legend(fontsize=8.5)
    axs[2].plot(st, anc['in_deg'], '-o', color=GREEN, ms=4, label='anchored')
    axs[2].plot(st, noa['in_deg'], '-o', color=RED, ms=4, label='no anchor')
    axs[2].set_title('coupling in-degree (edge penalty sparsifies)', color=INK, fontsize=12, loc='left')
    axs[2].set_xlabel('step'); axs[2].set_ylabel('mean strong targets / source'); axs[2].legend(fontsize=9)
    for ax in axs:
        for s_ in ['top', 'right']: ax.spines[s_].set_visible(False)
        for s_ in ['left', 'bottom']: ax.spines[s_].set_color(SECONDARY)
    fig.suptitle('Reconstruction-anchored joint CE training — faithful + CE-aware + sparse-wired',
                 fontsize=13.5, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG, dpi=150, facecolor=SURFACE); print('wrote', FIG, flush=True)

    pa = (min(anc['r2_d0']) > 0.6 and min(anc['ce_rec']) >= indep_rec - 0.08
          and anc['in_deg'][-1] <= 0.6*anc['in_deg'][0])
    pb = noa['r2_d0'][-1] < 0.0 and noa['ce_rec'][-1] < indep_rec - 0.1
    out = {'indep_ce_rec': round(indep_rec, 4), 'lam_rec': LAM_REC, 'lam_e': LAM_E,
           'anchored': anc, 'no_anchor': noa, 'benefit': round(ben, 4),
           'pred_a': bool(pa), 'pred_b': bool(pb), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) anchor faithful+sparse: {pa}; (b) control collapses: {pb}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
