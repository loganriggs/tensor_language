"""EDGE CAUSALITY (phase-3 opener: is the sparse wiring CAUSAL, not just a sparse
fit?). Train the reconstruction-anchored joint (Down_0, Left_1) as in 759, get the
weight-only coupling C = norm(E_L1) @ norm(D_D0) (P2 x P1). Then test, causally:
  (A) GRAPH PREDICTS DOWNSTREAM EFFECT: knock out source atom i (zero its Down_0
      code), run the model, measure each TARGET atom j's pre-activation change
      Delta z2_pre[j]. If the weight coupling is causal, Delta z2_pre correlates
      with -C[:,i] (removing source i's drive lowers exactly the targets C says it
      drives). Compare to a random wrong-source column (null).
  (B) STRONG-COUPLING ATOMS MATTER MORE FOR CE: knocking out high-out-degree source
      atoms raises CE more than low-out-degree ones.

REGISTERED PREDICTIONS:
  (0) SANITY: knockouts change downstream target activations (nonzero delta);
  (a) CAUSAL WIRING: mean corr( Delta z2_pre , -C[:,i] ) over tested sources >= 0.4
      and >> the wrong-source null (< 0.1); the coupling predicts the causal
      downstream effect;
  (b) high-coupling source atoms raise CE more than low-coupling (ratio > 1.5);
  NULL: wrong-source-column corr ~ 0."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd'); sys.path.insert(0, '/workspace/tensor_language')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'edge_causality_results.json'
NFIT = 48; NEVAL = 24; P = 512; K = 32; STEPS = 350; LAM_E = 0.03; LAM_REC = 1.0
SUB = {'d0': None, 'l1': None}; KNOCK = {'mask': None}


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


def sub_d0(Dm, Em, b):
    # Down_0 substitution with optional source-atom knockout mask
    def fn(g):
        z = topk(g @ Em.T, K)
        if KNOCK['mask'] is not None: z = z * KNOCK['mask']
        return z @ Dm.T + b
    return fn


def sub_l1(Dm, Em, b): return lambda g: topk(g @ Em.T, K) @ Dm.T + b


def train_indep(Xin, Ytrue, din, dout, seed):
    torch.manual_seed(seed)
    Dm = (torch.randn(dout, P, device=DEV)/np.sqrt(dout)).requires_grad_(True)
    Em = (torch.randn(P, din, device=DEV)/np.sqrt(din)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True); opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(600):
        z = topk(Xin @ Em.T, K); loss = F.mse_loss(z @ Dm.T + b, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
    return Dm.detach(), Em.detach(), b.detach()


def train_anchored(fit, g0, r1, Y0, YL, init0, initL):
    D0 = init0[0].clone().requires_grad_(True); E0 = init0[1].clone().requires_grad_(True); b0 = init0[2].clone().requires_grad_(True)
    DL = initL[0].clone().requires_grad_(True); EL = initL[1].clone().requires_grad_(True); bL = initL[2].clone().requires_grad_(True)
    opt = torch.optim.Adam([D0, E0, b0, DL, EL, bL], lr=1e-3)
    for s in range(STEPS):
        SUB['d0'] = sub_d0(D0, E0, b0); SUB['l1'] = sub_l1(DL, EL, bL); KNOCK['mask'] = None
        bb = fit[(s % (NFIT//4))*4:(s % (NFIT//4))*4+4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        ce = F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1))
        mse = F.mse_loss(topk(g0 @ E0.T, K) @ D0.T + b0, Y0) + F.mse_loss(topk(r1 @ EL.T, K) @ DL.T + bL, YL)
        C = F.normalize(EL, dim=1) @ F.normalize(D0, dim=0)
        (ce + LAM_REC*mse + LAM_E*C.abs().mean()).backward(); opt.step(); opt.zero_grad()
    SUB['d0'] = SUB['l1'] = None
    return (D0.detach(), E0.detach(), b0.detach()), (DL.detach(), EL.detach(), bL.detach())


@torch.no_grad()
def capture_z2pre(rows, n, EL):
    # Left_1 pre-topk codes = EL @ (Left_1 input), under current SUB/KNOCK
    r1 = capture(rows, n, m.transformer.h[1].mlp.Left, D)
    return r1 @ EL.T


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL); fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    W0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV)
    WL = m.transformer.h[1].mlp.Left.weight.data.float().to(DEV)
    h0 = m.transformer.h[0].mlp.Down.register_forward_hook(hook_d0)
    hL = m.transformer.h[1].mlp.Left.register_forward_hook(hook_l1)
    g0 = capture(fit, NFIT, m.transformer.h[0].mlp.Down, HID); r1 = capture(fit, NFIT, m.transformer.h[1].mlp.Left, D)
    Y0, YL = g0 @ W0.T, r1 @ WL.T
    with torch.enable_grad():
        init0 = train_indep(g0, Y0, HID, D, 1); initL = train_indep(r1, YL, D, HID, 3)
        (D0, E0, b0), (DL, EL, bL) = train_anchored(fit, g0, r1, Y0, YL, init0, initL)
    C = (F.normalize(EL, dim=1) @ F.normalize(D0, dim=0))       # (P2, P1)
    outdeg = (C.abs() > 0.2*C.abs().max(0).values.clamp_min(1e-9)).float().sum(0)   # per source atom
    # USAGE of each source atom on eval (must FIRE to have any causal effect -- the
    # 15:42 all-zeros bug knocked high-coupling but INACTIVE atoms; select by usage).
    g0_ev = capture(ev, NEVAL, m.transformer.h[0].mlp.Down, HID)
    usage = (topk(g0_ev @ E0.T, K) > 1e-6).float().mean(0)      # (P1,) fraction of tokens active
    active = (usage > 0).nonzero(as_tuple=True)[0]
    hi_out_usage = float(usage[torch.argsort(-outdeg)[:16]].mean())
    print(f'trained. mean out-deg {outdeg.mean():.1f} | mean usage {usage.mean():.3f} | '
          f'usage of top-16 out-deg atoms {hi_out_usage:.3f} (bug check: low=inactive)', flush=True)

    # baseline z2_pre (no knockout)
    SUB['d0'] = sub_d0(D0, E0, b0); SUB['l1'] = sub_l1(DL, EL, bL); KNOCK['mask'] = None
    z2_base = capture_z2pre(ev, NEVAL, EL).mean(0)              # (P2,)

    # TEST A: knock out top-USAGE (active) source atoms; correlate downstream delta with -C[:,i]
    top_src = active[torch.argsort(-usage[active])[:12]]
    corrs, nulls = [], []
    g = torch.Generator(device=DEV).manual_seed(0)
    for i in top_src.tolist():
        mask = torch.ones(P, device=DEV); mask[i] = 0.0; KNOCK['mask'] = mask
        z2_k = capture_z2pre(ev, NEVAL, EL).mean(0)
        delta = z2_k - z2_base                                   # (P2,)
        pred = -C[:, i]                                          # graph prediction of the sign/size
        d = delta - delta.mean(); p = pred - pred.mean()
        corrs.append(float((d*p).sum()/(d.norm()*p.norm()).clamp_min(1e-9)))
        j = int(torch.randint(0, P, (1,), generator=g, device=DEV)); pr = -C[:, j]; pr = pr - pr.mean()
        nulls.append(float((d*pr).sum()/(d.norm()*pr.norm()).clamp_min(1e-9)))
    KNOCK['mask'] = None
    mean_corr = float(np.mean(corrs)); mean_null = float(np.mean(nulls))
    print(f'(A) graph-predicts-effect: mean corr {mean_corr:.3f}  wrong-source null {mean_null:.3f}', flush=True)

    # TEST B: among ACTIVE atoms, CE increase knocking high vs low out-degree (matched usage band)
    def ce_knock(atoms):
        mask = torch.ones(P, device=DEV); mask[atoms] = 0.0; KNOCK['mask'] = mask
        c = ce_on(ev, NEVAL); KNOCK['mask'] = None; return c
    SUB['d0'] = sub_d0(D0, E0, b0); SUB['l1'] = sub_l1(DL, EL, bL)
    ce0 = ce_on(ev, NEVAL)
    act_sorted = active[torch.argsort(-usage[active])[:64]]      # 64 most-used active atoms
    od = outdeg[act_sorted]
    hi = act_sorted[torch.argsort(-od)[:16]].tolist(); lo = act_sorted[torch.argsort(od)[:16]].tolist()
    dce_hi = ce_knock(hi) - ce0; dce_lo = ce_knock(lo) - ce0
    print(f'(B) dCE knock high-coupling {dce_hi:.4f} vs low {dce_lo:.4f}  ratio {dce_hi/max(dce_lo,1e-6):.2f}', flush=True)
    h0.remove(); hL.remove()

    p0 = abs(mean_corr) > 0.05
    pa = mean_corr >= 0.4 and mean_corr - mean_null >= 0.3
    pb = dce_hi > 1.5*max(dce_lo, 1e-6)
    null_ok = abs(mean_null) < 0.1
    out = {'mean_outdeg': round(float(outdeg.mean()), 2), 'mean_usage': round(float(usage.mean()), 4),
           'top_outdeg_atom_usage': round(hi_out_usage, 4), 'testA_mean_corr': round(mean_corr, 4),
           'testA_wrong_src_null': round(mean_null, 4), 'per_source_corr': [round(c, 3) for c in corrs],
           'ce0': round(ce0, 4), 'dce_high_coupling': round(dce_hi, 4), 'dce_low_coupling': round(dce_lo, 4),
           'pred_0': bool(p0), 'pred_a': bool(pa), 'pred_b': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) wiring CAUSAL (graph predicts downstream, corr>=0.4 & >>null): {pa}; '
          f'(b) strong-coupling atoms hurt CE more: {pb}; NULL wrong-src~0: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
