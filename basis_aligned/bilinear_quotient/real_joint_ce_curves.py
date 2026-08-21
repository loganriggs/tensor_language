"""TRAINING CURVES for the real-model CE-trained joint composition (diagnose the
0.908->0.605 drop: undertraining, or CE fine-tune damaging a good init?). Logs,
over steps: eval CE-recovery (held out), per-layer reconstruction R2 (Down_0,
Left_1), coupling in-degree, and train CE. Warm-starts joint from the independent
MSE solution (CE-rec ~0.9). Conditions:
  indep        -- reference (MSE per layer, no CE): flat lines.
  joint_e0     -- CE fine-tune, no edge, NFIT=16 (the real-run config).
  joint_e      -- CE fine-tune + edge penalty, NFIT=16.
  joint_e0_big -- CE fine-tune, no edge, NFIT=48 (overfitting test: more data ->
                  does eval CE-recovery hold?).
Produces a 4-panel figure (palette-matched) + the curve arrays as JSON.

REGISTERED PREDICTIONS:
  (0) if the drop is OVERFITTING: joint_e0 eval CE-recovery FALLS below its warm-
      start 0.9 as train CE falls, and joint_e0_big (more data) holds HIGHER;
  (a) if UNDERTRAINING: eval CE-recovery still RISING at the step budget (it is not
      plateaued) -- more steps would help;
  (b) edge penalty drops in-degree sharply with eval CE-recovery tracking joint_e0
      (sparser wiring, ~no CE cost);
  NULL: reconstruction R2 of each layer stays high if CE-training is not damaging
      the weight fit (if R2 falls, CE-training is trading weight-faithfulness for
      fit-set CE = overfitting)."""
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
OUT = PT + 'real_joint_ce_curves_results.json'; FIG = PT + 'real_joint_ce_curves.png'
NEVAL = 24; P = 512; K = 32; STEPS = 500; EVAL_EVERY = 25; LAM_E = 0.03
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


def run_joint(fit, nfit, g0, r1, Y0, YL, lam, init0, initL, ce_abl, ben, ev):
    D0 = init0[0].clone().requires_grad_(True); E0 = init0[1].clone().requires_grad_(True); b0 = init0[2].clone().requires_grad_(True)
    DL = initL[0].clone().requires_grad_(True); EL = initL[1].clone().requires_grad_(True); bL = initL[2].clone().requires_grad_(True)
    opt = torch.optim.Adam([D0, E0, b0, DL, EL, bL], lr=1e-3)
    cur = {'step': [], 'ce_rec': [], 'r2_d0': [], 'r2_l1': [], 'in_deg': [], 'train_ce': []}
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
        C = F.normalize(EL, dim=1) @ F.normalize(D0, dim=0)
        (ce + lam*C.abs().mean()).backward(); opt.step(); opt.zero_grad()
        cur['train_ce'].append(round(float(ce), 4))
        SUB['d0'] = SUB['l1'] = None
    return cur


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(48 + NEVAL); ev = rows[48:48+NEVAL]
    W0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV)
    WL = m.transformer.h[1].mlp.Left.weight.data.float().to(DEV)
    h0 = m.transformer.h[0].mlp.Down.register_forward_hook(hook_d0)
    hL = m.transformer.h[1].mlp.Left.register_forward_hook(hook_l1)

    g0_big = capture(rows[:48], 48, m.transformer.h[0].mlp.Down, HID)
    r1_big = capture(rows[:48], 48, m.transformer.h[1].mlp.Left, D)
    # per-token counts differ; index the first 16-row worth for the small conditions
    n16 = g0_big.shape[0] * 16 // 48
    g0, r1 = g0_big[:n16], r1_big[:n16]
    Y0, YL = g0 @ W0.T, r1 @ WL.T; Y0b, YLb = g0_big @ W0.T, r1_big @ WL.T

    SUB['d0'] = SUB['l1'] = None
    with torch.no_grad(): ce_full = ce_on(ev, NEVAL)
    SUB['d0'] = lambda g: torch.zeros(g.shape[0], D, device=DEV); SUB['l1'] = lambda g: torch.zeros(g.shape[0], HID, device=DEV)
    with torch.no_grad(): ce_abl = ce_on(ev, NEVAL)
    SUB['d0'] = SUB['l1'] = None; ben = ce_abl - ce_full
    print(f'CE_full {ce_full:.3f} benefit {ben:.3f}', flush=True)

    with torch.enable_grad():
        init0 = train_indep(g0, Y0, HID, D, 1); initL = train_indep(r1, YL, D, HID, 3)
        init0b = train_indep(g0_big, Y0b, HID, D, 1); initLb = train_indep(r1_big, YLb, D, HID, 3)
    SUB['d0'] = sub_fn(*init0); SUB['l1'] = sub_fn(*initL)
    with torch.no_grad(): indep_rec = float((ce_abl - ce_on(ev, NEVAL))/max(ben, 1e-6))
    SUB['d0'] = SUB['l1'] = None
    print(f'indep CE-rec {indep_rec:.3f}', flush=True)

    with torch.enable_grad():
        c_e0 = run_joint(rows[:16], 16, g0, r1, Y0, YL, 0.0, init0, initL, ce_abl, ben, ev)
        print(f'joint_e0 end CE-rec {c_e0["ce_rec"][-1]}  in-deg {c_e0["in_deg"][-1]}', flush=True)
        c_e = run_joint(rows[:16], 16, g0, r1, Y0, YL, LAM_E, init0, initL, ce_abl, ben, ev)
        print(f'joint_e  end CE-rec {c_e["ce_rec"][-1]}  in-deg {c_e["in_deg"][-1]}', flush=True)
        c_big = run_joint(rows[:48], 48, g0_big, r1_big, Y0b, YLb, 0.0, init0b, initLb, ce_abl, ben, ev)
        print(f'joint_e0_big end CE-rec {c_big["ce_rec"][-1]}  in-deg {c_big["in_deg"][-1]}', flush=True)
    h0.remove(); hL.remove()

    # ---- figure ----
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8)); fig.patch.set_facecolor(SURFACE)
    for ax in axs.flat: ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.6, zorder=0)
    st = c_e0['step']
    A = axs[0, 0]
    A.axhline(indep_rec, color=MUTED, lw=1.5, ls='--', label=f'indep (MSE) {indep_rec:.2f}')
    A.plot(st, c_e0['ce_rec'], '-o', color=BLUE, ms=4, label='joint CE, no edge (16 rows)')
    A.plot(st, c_e['ce_rec'], '-o', color=GREEN, ms=4, label='joint CE + edge (16 rows)')
    A.plot(st, c_big['ce_rec'], '-o', color=ORANGE, ms=4, label='joint CE, no edge (48 rows)')
    A.set_title('eval CE-recovery vs step (held-out)', color=INK, fontsize=12, loc='left')
    A.set_xlabel('step', fontsize=10); A.set_ylabel('CE-recovery', fontsize=10); A.legend(fontsize=8.5, framealpha=0.9)
    B = axs[0, 1]
    B.plot(st, c_e0['r2_d0'], '-o', color=BLUE, ms=4, label='Down_0 R² (no edge)')
    B.plot(st, c_e0['r2_l1'], '-s', color=BLUE, ms=4, mfc='none', label='Left_1 R² (no edge)')
    B.plot(st, c_e['r2_d0'], '-o', color=GREEN, ms=4, label='Down_0 R² (edge)')
    B.plot(st, c_e['r2_l1'], '-s', color=GREEN, ms=4, mfc='none', label='Left_1 R² (edge)')
    B.set_title('per-layer reconstruction R² vs step\n(falls => CE-training damages weight fit)', color=INK, fontsize=12, loc='left')
    B.set_xlabel('step', fontsize=10); B.set_ylabel('action R²', fontsize=10); B.legend(fontsize=8.5)
    C = axs[1, 0]
    C.plot(st, c_e0['in_deg'], '-o', color=BLUE, ms=4, label='no edge')
    C.plot(st, c_e['in_deg'], '-o', color=GREEN, ms=4, label='+ edge penalty')
    C.set_title('coupling in-degree vs step (wiring density)', color=INK, fontsize=12, loc='left')
    C.set_xlabel('step', fontsize=10); C.set_ylabel('mean strong targets / source atom', fontsize=10); C.legend(fontsize=9)
    Dp = axs[1, 1]
    Dp.plot(range(len(c_e0['train_ce'])), c_e0['train_ce'], color=BLUE, lw=1, alpha=0.8, label='no edge (16)')
    Dp.plot(range(len(c_big['train_ce'])), c_big['train_ce'], color=ORANGE, lw=1, alpha=0.8, label='no edge (48)')
    Dp.set_title('train CE vs step (fit set)', color=INK, fontsize=12, loc='left')
    Dp.set_xlabel('step', fontsize=10); Dp.set_ylabel('train CE (nats)', fontsize=10); Dp.legend(fontsize=9)
    for ax in axs.flat:
        for s_ in ['top', 'right']: ax.spines[s_].set_visible(False)
        for s_ in ['left', 'bottom']: ax.spines[s_].set_color(SECONDARY)
    fig.suptitle('Weight-action joint CE training — is the 0.91→0.61 drop undertraining or overfitting?',
                 fontsize=13.5, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIG, dpi=150, facecolor=SURFACE)
    print('wrote', FIG, flush=True)

    out = {'indep_ce_rec': round(indep_rec, 4), 'joint_e0': c_e0, 'joint_e': c_e, 'joint_e0_big': c_big,
           'benefit': round(ben, 4), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1); print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
