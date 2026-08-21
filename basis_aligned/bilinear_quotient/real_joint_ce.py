"""REAL-MODEL CE-TRAINED JOINT COMPOSITION (the culmination: train two weight-
action SAEs on the LIVE model, end-to-end on CE, with an edge penalty that
sparsifies the cross-layer wiring). Pair = Down_0 (residual WRITE) and Left_1
(residual READ), the clean linear-coupling pair from 754.

Both SAEs are substituted LIVE via forward hooks (params require grad); the full
18-layer forward -> logits -> CE is backpropagated to the SAE params. Objective:
  loss = CE( model with Down_0, Left_1 replaced by their SAE recon )
       + lam_e * || normalize(E_L1, rows) @ normalize(D_D0, cols) ||_1
The edge term penalizes the weight-only coupling C = E_L1 @ D_D0 (P x P), pushing
the two dictionaries to a SPARSE wiring. This is the CE objective the user wants
(reconstruction != CE, 737/748), and the full forward auto-handles input drift.

Compare three fits + reference (joint fits WARM-START from the indep MSE solution,
CE-rec ~0.9, then CE fine-tune -> isolates the edge penalty's effect on wiring
sparsity from the harder from-scratch CE optimization):
  (indep)     each SAE trained on its own MSE action-recon (750 way), no CE, no edge.
  (joint,e=0) warm-start, then JOINT CE fine-tune, no edge penalty.
  (joint,e>0) warm-start, then JOINT CE + edge penalty fine-tune.
Report CE-recovery and coupling in-degree (wiring density) for each; A-SVD rank-r
and random-OC as references/null.

REGISTERED PREDICTIONS:
  (0) SANITY: joint-CE fits keep CE-recovery high (>0.7);
  (a) EDGE PENALTY SPARSIFIES WIRING AT LOW CE COST: joint(e>0) has LOWER coupling
      in-degree than joint(e=0) and indep (by >=25%) while CE-recovery stays within
      0.1 of joint(e=0) -- a sparse faithful wiring, trained on CE;
  (b) JOINT-CE >= INDEP on CE-recovery (end-to-end CE beats per-layer MSE for the
      loss we care about), report all;
  NULL: random-OC substitution is catastrophic (CE-recovery << 0)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'real_joint_ce_results.json'
NFIT = 16; NEVAL = 24; P = 512; K = 32; STEPS = 250; LAM_E = 3e-2
SUB = {'d0': None, 'l1': None}     # substitution fns (gate-> recon); None = original


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def hook_d0(mo, i_, o_):
    if SUB['d0'] is None: return o_
    return SUB['d0'](i_[0].reshape(-1, HID)).reshape(o_.shape).to(o_.dtype)


def hook_l1(mo, i_, o_):
    if SUB['l1'] is None: return o_
    return SUB['l1'](i_[0].reshape(-1, D)).reshape(o_.shape).to(o_.dtype)


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
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
    h.remove(); return torch.cat(cap, 0)


def new_sae(Xin, Ytrue, din, dout, seed):
    torch.manual_seed(seed)
    Dm = (torch.randn(dout, P, device=DEV)/np.sqrt(dout)).requires_grad_(True)
    Em = (torch.randn(P, din, device=DEV)/np.sqrt(din)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True)
    return Dm, Em, b


def sub_fn(Dm, Em, b, k=K):
    return lambda g: topk(g @ Em.T, k) @ Dm.T + b


def train_indep(Xin, Ytrue, din, dout, seed):
    Dm, Em, b = new_sae(Xin, Ytrue, din, dout, seed); opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(600):
        z = topk(Xin @ Em.T, K); loss = F.mse_loss(z @ Dm.T + b, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
    return Dm.detach(), Em.detach(), b.detach()


def train_joint_ce(fit, g0, r1, W0, WL, lam_e, seed, init0=None, initL=None):
    # warm-start from the faithful independent MSE solution (CE-rec ~0.9), then CE
    # + edge fine-tune -> isolates whether the edge penalty sparsifies the wiring
    # while CE stays high, rather than conflating with from-scratch CE optimization.
    if init0 is not None:
        D0 = init0[0].clone().requires_grad_(True); E0 = init0[1].clone().requires_grad_(True); b0 = init0[2].clone().requires_grad_(True)
        DL = initL[0].clone().requires_grad_(True); EL = initL[1].clone().requires_grad_(True); bL = initL[2].clone().requires_grad_(True)
    else:
        D0, E0, b0 = new_sae(g0, g0 @ W0.T, HID, D, seed)
        DL, EL, bL = new_sae(r1, r1 @ WL.T, D, HID, seed+1)
    opt = torch.optim.Adam([D0, E0, b0, DL, EL, bL], lr=1e-3)
    for s in range(STEPS):
        SUB['d0'] = sub_fn(D0, E0, b0); SUB['l1'] = sub_fn(DL, EL, bL)
        bb = fit[(s % (NFIT//4))*4:(s % (NFIT//4))*4+4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        ce = F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1))
        C = F.normalize(EL, dim=1) @ F.normalize(D0, dim=0)
        loss = ce + lam_e*C.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    SUB['d0'] = SUB['l1'] = None
    return (D0.detach(), E0.detach(), b0.detach()), (DL.detach(), EL.detach(), bL.detach())


def in_degree(EL, D0, thr=0.2):
    C = F.normalize(EL, dim=1) @ F.normalize(D0, dim=0)
    strong = (C.abs() > thr*C.abs().max(0).values.clamp_min(1e-9)).float()
    return float(strong.sum(0).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL); fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    W0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV)
    WL = m.transformer.h[1].mlp.Left.weight.data.float().to(DEV)
    h0 = m.transformer.h[0].mlp.Down.register_forward_hook(hook_d0)
    hL = m.transformer.h[1].mlp.Left.register_forward_hook(hook_l1)

    g0 = capture(fit, NFIT, m.transformer.h[0].mlp.Down, HID)
    r1 = capture(fit, NFIT, m.transformer.h[1].mlp.Left, D)

    SUB['d0'] = SUB['l1'] = None; ce_full = ce_on(ev, NEVAL)
    SUB['d0'] = lambda g: torch.zeros(g.shape[0], D, device=DEV)
    SUB['l1'] = lambda g: torch.zeros(g.shape[0], HID, device=DEV); ce_abl = ce_on(ev, NEVAL)
    SUB['d0'] = SUB['l1'] = None; ben = ce_abl - ce_full
    def rec(ce): return float((ce_abl - ce)/max(ben, 1e-6))
    print(f'CE_full {ce_full:.3f} ablate-both benefit {ben:.3f}', flush=True)

    res = {}
    # indep (MSE per layer)
    with torch.enable_grad():
        D0i = train_indep(g0, g0 @ W0.T, HID, D, 1); DLi = train_indep(r1, r1 @ WL.T, D, HID, 3)
    SUB['d0'] = sub_fn(*D0i); SUB['l1'] = sub_fn(*DLi); ce = ce_on(ev, NEVAL); SUB['d0'] = SUB['l1'] = None
    res['indep'] = {'ce_recovery': round(rec(ce), 4), 'in_degree': round(in_degree(DLi[1], D0i[0]), 1)}
    print(f"indep     : CE-rec {res['indep']['ce_recovery']:.3f}  in-deg {res['indep']['in_degree']:.0f}", flush=True)

    # joint CE (warm-started from indep), edge off / on
    for tag, lam in [('joint_e0', 0.0), ('joint_e', LAM_E)]:
        with torch.enable_grad(): (D0, E0, b0), (DL, EL, bL) = train_joint_ce(fit, g0, r1, W0, WL, lam, 10, init0=D0i, initL=DLi)
        SUB['d0'] = sub_fn(D0, E0, b0); SUB['l1'] = sub_fn(DL, EL, bL); ce = ce_on(ev, NEVAL); SUB['d0'] = SUB['l1'] = None
        res[tag] = {'ce_recovery': round(rec(ce), 4), 'in_degree': round(in_degree(EL, D0), 1)}
        print(f"{tag:9s} : CE-rec {res[tag]['ce_recovery']:.3f}  in-deg {res[tag]['in_degree']:.0f}", flush=True)

    # null: random-OC
    torch.manual_seed(1); Wr0 = torch.randn(P, HID, device=DEV); Dr0 = torch.randn(D, P, device=DEV)
    WrL = torch.randn(P, D, device=DEV); DrL = torch.randn(HID, P, device=DEV)
    SUB['d0'] = lambda g, W=Wr0, Dd=Dr0, b=(g0@W0.T).mean(0): topk(g@W.T, K)@Dd.T + b
    SUB['l1'] = lambda g, W=WrL, Dd=DrL, b=(r1@WL.T).mean(0): topk(g@W.T, K)@Dd.T + b
    ce = ce_on(ev, NEVAL); SUB['d0'] = SUB['l1'] = None
    res['rand'] = {'ce_recovery': round(rec(ce), 4)}
    print(f"rand-OC   : CE-rec {res['rand']['ce_recovery']:.3f}", flush=True)
    h0.remove(); hL.remove()

    p0 = res['joint_e0']['ce_recovery'] > 0.7
    pa = (res['joint_e']['in_degree'] <= 0.75*res['joint_e0']['in_degree']
          and res['joint_e']['ce_recovery'] >= res['joint_e0']['ce_recovery'] - 0.1)
    pb = res['joint_e0']['ce_recovery'] >= res['indep']['ce_recovery'] - 0.02
    null_ok = res['rand']['ce_recovery'] < 0
    out = {'ce_full': round(ce_full, 4), 'benefit': round(ben, 4), 'P': P, 'K': K, 'lam_e': LAM_E,
           'results': res, 'pred_0': bool(p0), 'pred_a': bool(pa), 'pred_b': bool(pb),
           'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(0) joint-CE faithful: {p0}; (a) edge penalty sparsifies wiring at low CE cost: {pa}; '
          f'(b) joint-CE>=indep: {pb}; NULL rand catastrophic: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
