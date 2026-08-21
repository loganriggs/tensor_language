"""CONVERGENCE / TWO-KINDS-OF-FAITHFULNESS test (user: "the no-edge 48 looks
undertrained"). 758 stated pure-CE training DAMAGES the SAE, but its CE-recovery
curves were still declining at 500 steps and NOT converged -- so the endpoint was
provisional. Settle it: run pure-CE (no reconstruction anchor, no edge) at 48 and
128 rows for 1000 steps, tracking BOTH eval CE-recovery AND reconstruction R2,
against the anchored reference. Distinguishes:
  * UNDERTRAINING (user's hypothesis): pure-CE CE-recovery keeps CLIMBING with more
    data+steps toward indep -- 758's low numbers were just too-few-data/steps;
  * TWO KINDS OF FAITHFULNESS: pure-CE reaches decent CE-recovery (LOSS-faithful)
    but reconstruction R2 stays collapsed/negative (WEIGHT-unfaithful) -- there are
    two distinct fidelities and CE alone buys only one;
  * DAMAGE (758's claim): pure-CE CE-recovery plateaus BELOW indep and R2 negative.

Warm-start from the independent MSE fit (CE-rec ~0.94) so we watch where CE walks it.

REGISTERED PREDICTIONS:
  (0) SANITY: indep warm-start CE-rec ~0.9;
  (a) WEIGHT-UNFAITHFUL regardless of data: pure-CE reconstruction R2_Down0 ends < 0.3
      at BOTH 48 and 128 rows (CE has no reconstruction term -> abandons it; more data
      does not restore weight-faithfulness);
  (b) MORE DATA HELPS CE-RECOVERY: pure-CE 128-row final CE-recovery > 48-row (less
      overfitting), quantifying how much of 758's drop was data-limited vs fundamental;
  (c) anchored reference holds BOTH (R2 > 0.6 and CE-rec >= indep-0.05);
  NULL: n/a (anchored is the positive control)."""
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
OUT = PT + 'real_joint_ce_converge_results.json'; FIG = PT + 'real_joint_ce_converge.png'
NEVAL = 24; P = 512; K = 32; STEPS = 1000; EVAL_EVERY = 50
BLUE, RED, GREEN, ORANGE, PURPLE = '#3987e5', '#e34948', '#2e8b57', '#e08a1e', '#7b4ea8'
SUB = {'d0': None, 'l1': None}


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


def train_indep(Xin, Ytrue, din, dout, seed):
    torch.manual_seed(seed)
    Dm = (torch.randn(dout, P, device=DEV)/np.sqrt(dout)).requires_grad_(True)
    Em = (torch.randn(P, din, device=DEV)/np.sqrt(din)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True); opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(700):
        z = topk(Xin @ Em.T, K); loss = F.mse_loss(z @ Dm.T + b, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
    return Dm.detach(), Em.detach(), b.detach()


def run(fit, nfit, g0, r1, Y0, YL, lam_rec, init0, initL, ce_abl, ben, ev):
    D0 = init0[0].clone().requires_grad_(True); E0 = init0[1].clone().requires_grad_(True); b0 = init0[2].clone().requires_grad_(True)
    DL = initL[0].clone().requires_grad_(True); EL = initL[1].clone().requires_grad_(True); bL = initL[2].clone().requires_grad_(True)
    opt = torch.optim.Adam([D0, E0, b0, DL, EL, bL], lr=1e-3)
    cur = {'step': [], 'ce_rec': [], 'r2_d0': []}
    for s in range(STEPS + 1):
        if s % EVAL_EVERY == 0:
            SUB['d0'] = sub_fn(D0.detach(), E0.detach(), b0.detach()); SUB['l1'] = sub_fn(DL.detach(), EL.detach(), bL.detach())
            with torch.no_grad():
                ce = ce_on(ev, NEVAL); rec0 = topk(g0 @ E0.T, K) @ D0.T + b0
            SUB['d0'] = SUB['l1'] = None
            cur['step'].append(s); cur['ce_rec'].append(round(float((ce_abl-ce)/max(ben,1e-6)), 4))
            cur['r2_d0'].append(round(r2(rec0, Y0), 4))
        if s == STEPS: break
        SUB['d0'] = sub_fn(D0, E0, b0); SUB['l1'] = sub_fn(DL, EL, bL)
        bb = fit[(s % (nfit//4))*4:(s % (nfit//4))*4+4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        ce = F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1))
        loss = ce
        if lam_rec > 0:
            loss = loss + lam_rec*(F.mse_loss(topk(g0 @ E0.T, K) @ D0.T + b0, Y0) + F.mse_loss(topk(r1 @ EL.T, K) @ DL.T + bL, YL))
        loss.backward(); opt.step(); opt.zero_grad()
        SUB['d0'] = SUB['l1'] = None
    return cur


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    NBIG = 128; rows = cl.fineweb_rows(NBIG + NEVAL); ev = rows[NBIG:NBIG+NEVAL]
    W0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV)
    WL = m.transformer.h[1].mlp.Left.weight.data.float().to(DEV)
    h0 = m.transformer.h[0].mlp.Down.register_forward_hook(hook_d0)
    hL = m.transformer.h[1].mlp.Left.register_forward_hook(hook_l1)
    g0b = capture(rows[:NBIG], NBIG, m.transformer.h[0].mlp.Down, HID)
    r1b = capture(rows[:NBIG], NBIG, m.transformer.h[1].mlp.Left, D)
    Y0b, YLb = g0b @ W0.T, r1b @ WL.T
    n48 = g0b.shape[0]*48//NBIG; g048, r148 = g0b[:n48], r1b[:n48]; Y048, YL48 = g048 @ W0.T, r148 @ WL.T

    SUB['d0'] = SUB['l1'] = None
    with torch.no_grad(): ce_full = ce_on(ev, NEVAL)
    SUB['d0'] = lambda g: torch.zeros(g.shape[0], D, device=DEV); SUB['l1'] = lambda g: torch.zeros(g.shape[0], HID, device=DEV)
    with torch.no_grad(): ce_abl = ce_on(ev, NEVAL)
    SUB['d0'] = SUB['l1'] = None; ben = ce_abl - ce_full
    with torch.enable_grad():
        i0b = train_indep(g0b, Y0b, HID, D, 1); iLb = train_indep(r1b, YLb, D, HID, 3)
        i048 = train_indep(g048, Y048, HID, D, 1); iL48 = train_indep(r148, YL48, D, HID, 3)
    SUB['d0'] = sub_fn(*i0b); SUB['l1'] = sub_fn(*iLb)
    with torch.no_grad(): indep_rec = float((ce_abl - ce_on(ev, NEVAL))/max(ben, 1e-6))
    SUB['d0'] = SUB['l1'] = None
    print(f'benefit {ben:.3f} | indep CE-rec {indep_rec:.3f}', flush=True)

    with torch.enable_grad():
        pure48 = run(rows[:48], 48, g048, r148, Y048, YL48, 0.0, i048, iL48, ce_abl, ben, ev)
        print(f'pure-CE 48  end CE-rec {pure48["ce_rec"][-1]}  R2 {pure48["r2_d0"][-1]}', flush=True)
        pure128 = run(rows[:NBIG], NBIG, g0b, r1b, Y0b, YLb, 0.0, i0b, iLb, ce_abl, ben, ev)
        print(f'pure-CE 128 end CE-rec {pure128["ce_rec"][-1]}  R2 {pure128["r2_d0"][-1]}', flush=True)
        anc128 = run(rows[:NBIG], NBIG, g0b, r1b, Y0b, YLb, 1.0, i0b, iLb, ce_abl, ben, ev)
        print(f'anchored128 end CE-rec {anc128["ce_rec"][-1]}  R2 {anc128["r2_d0"][-1]}', flush=True)
    h0.remove(); hL.remove()

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8)); fig.patch.set_facecolor(SURFACE)
    for ax in axs: ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.6)
    st = pure128['step']
    axs[0].axhline(indep_rec, color=MUTED, ls='--', lw=1.5, label=f'indep {indep_rec:.2f}')
    axs[0].plot(pure48['step'], pure48['ce_rec'], '-o', color=ORANGE, ms=3, label='pure CE, 48 rows')
    axs[0].plot(st, pure128['ce_rec'], '-o', color=RED, ms=3, label='pure CE, 128 rows')
    axs[0].plot(st, anc128['ce_rec'], '-o', color=GREEN, ms=3, label='anchored, 128 rows')
    axs[0].set_title('eval CE-recovery vs step (1000 steps)', color=INK, fontsize=12, loc='left')
    axs[0].set_xlabel('step'); axs[0].set_ylabel('CE-recovery  (1 = as good as real component)'); axs[0].legend(fontsize=9, loc='center right')
    axs[0].annotate('better\n(more faithful)', xy=(0.045, 0.97), xytext=(0.045, 0.72), xycoords='axes fraction',
                    ha='center', fontsize=8.5, color=GREEN, fontweight='bold',
                    arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=1.8))
    axs[1].axhline(0, color=SECONDARY, lw=0.8)
    axs[1].plot(pure48['step'], pure48['r2_d0'], '-o', color=ORANGE, ms=3, label='pure CE, 48')
    axs[1].plot(st, pure128['r2_d0'], '-o', color=RED, ms=3, label='pure CE, 128')
    axs[1].plot(st, anc128['r2_d0'], '-o', color=GREEN, ms=3, label='anchored, 128')
    axs[1].set_title('Down_0 reconstruction R² vs step\n(does more data restore weight-faithfulness?)', color=INK, fontsize=12, loc='left')
    axs[1].set_xlabel('step'); axs[1].set_ylabel('action R²  (1 = perfect weight reconstruction)'); axs[1].legend(fontsize=9, loc='center right')
    axs[1].annotate('better\n(weight-faithful)', xy=(0.045, 0.97), xytext=(0.045, 0.72), xycoords='axes fraction',
                    ha='center', fontsize=8.5, color=GREEN, fontweight='bold',
                    arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=1.8))
    for ax in axs:
        for s_ in ['top', 'right']: ax.spines[s_].set_visible(False)
        for s_ in ['left', 'bottom']: ax.spines[s_].set_color(SECONDARY)
    fig.suptitle('Undertraining or two kinds of faithfulness? Pure-CE at 1000 steps, 48 vs 128 rows',
                 fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG, dpi=150, facecolor=SURFACE); print('wrote', FIG, flush=True)

    pa = pure48['r2_d0'][-1] < 0.3 and pure128['r2_d0'][-1] < 0.3
    pb = pure128['ce_rec'][-1] > pure48['ce_rec'][-1]
    pc = min(anc128['r2_d0']) > 0.6 and anc128['ce_rec'][-1] >= indep_rec - 0.05
    out = {'indep_ce_rec': round(indep_rec, 4), 'benefit': round(ben, 4),
           'pure_ce_48': pure48, 'pure_ce_128': pure128, 'anchored_128': anc128,
           'pred_a_weight_unfaithful': bool(pa), 'pred_b_more_data_helps_ce': bool(pb),
           'pred_c_anchor_holds_both': bool(pc), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) pure-CE weight-unfaithful both sizes (R2<0.3): {pa}; (b) more data helps CE-rec: {pb}; '
          f'(c) anchor holds both: {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
