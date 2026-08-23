"""AIRTIGHT confirmation of the §992 reconciliation on ONE instrument: at the front, the bilinear MLP's interaction is
LOAD-BEARING (deleting it hurts, §992) yet the output is reproducible by a BEST-FIT linear map (§941). The two facts
turn on TWO DIFFERENT linear surrogates for the MLP output; measure all of them side-by-side on the same CE eval.

Three surrogates replace the MLP output at a layer; report CE cost vs baseline and the linear-recoverable fraction
frac = 1 - cost/cost_meanablate (§941's definition):
  (A) BEST-FIT linear map  out ~= W x + c  (ridge-fit input->output on a calibration set; §941's method)
  (B) TAYLOR const+linear  Down(a*Rx + b*Lx - a*b) + bias  (drops the interaction u*w; §992's surrogate)
  (C) MEAN-ABLATE          output := its calibration mean  (the frac denominator)
  (NULL) best-fit linear map fit on SHUFFLED input (should recover ~0 -> frac ~0).

REGISTERED PREDICTIONS:
  (0) NULL: shuffled-input best-fit linear recovers ~0 (frac <~0.1) -> the best-fit map is genuine, not leakage.
  (a) FRONT (L0,L1): frac_bestfit is HIGH (>~0.8, replicating §941) while frac_taylor is MUCH LOWER (and may be
      NEGATIVE, i.e. worse than mean-ablate) -> the front interaction is load-bearing (Taylor drop hurts) YET
      best-fit-linearly reproducible (best-fit absorbs the interaction's linearly-shaped part). This is the direct
      proof of the §992 reconciliation;
  (b) MIDDLE (L8,L11): both surrogates have SMALL absolute cost (each middle MLP is low-stakes, redundant band
      §940), and frac_bestfit is LOWER than at the front (~0.4, §941 middle) -> genuinely multiplicative but
      low-stakes;
  (c) report per-layer cost_bestfit, cost_taylor, cost_meanablate, frac_bestfit, frac_taylor."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bestfit_vs_taylor_linear_results.json'
NCAL = 96; NEVAL = 160; SEQ = 256
LAYERS = [0, 1, 8, 11]; FRONT = [0, 1]; MIDDLE = [8, 11]
RIDGE = 10.0
AB = {}; WLIN = {}; WNULL = {}; MEANOUT = {}
MODE = {'kind': None, 'layer': None}   # kind in {bestfit, taylor, meanablate, null}


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook_factory(L):
    mlp = m.transformer.h[L].mlp
    def h(mo, i_, o_):
        if MODE['layer'] != L or MODE['kind'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_).float()
        k = MODE['kind']
        if k == 'meanablate':
            return MEANOUT[L].to(o_.dtype).expand_as(o_)
        if k == 'taylor':
            Lx = mlp.Left(x); Rx = mlp.Right(x); a, b = AB[L]
            bias = mlp.Down.bias if mlp.Down.bias is not None else 0
            return (F.linear(a*Rx + b*Lx - a*b, mlp.Down.weight) + bias).to(o_.dtype)
        W = WLIN[L] if k == 'bestfit' else WNULL[L]
        x1 = torch.cat([x.reshape(-1, D), torch.ones(x.reshape(-1, D).shape[0], 1, device=DEV)], 1)
        return (x1 @ W).reshape(o_.shape).to(o_.dtype)
    return h


@torch.no_grad()
def fit_surrogates(blocks):
    # gather (input, output, Left, Right) per layer; fit best-fit W, null W (shuffled input), a/b, mean-out
    Xs = {L: [] for L in LAYERS}; Ys = {L: [] for L in LAYERS}; As = {L: [] for L in LAYERS}; Bs = {L: [] for L in LAYERS}
    caps = {}; hs = []
    for L in LAYERS:
        mlp = m.transformer.h[L].mlp
        def mk(L, mlp):
            def h(mo, i_, o_):
                x = (i_[0] if isinstance(i_, tuple) else i_).float()
                caps[L] = (x.reshape(-1, D).detach(), o_.float().reshape(-1, o_.shape[-1]).detach(),
                           mlp.Left(x).float().reshape(-1, mlp.Left.weight.shape[0]).detach(),
                           mlp.Right(x).float().reshape(-1, mlp.Right.weight.shape[0]).detach())
            return h
        hs.append(mlp.register_forward_hook(mk(L, mlp)))
    for i in range(0, blocks.shape[0], 8):
        forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
        for L in LAYERS:
            x, o, lx, rx = caps[L]; n = x.shape[0]; idx = torch.randperm(n, device=DEV)[:400]
            Xs[L].append(x[idx].cpu()); Ys[L].append(o[idx].cpu()); As[L].append(lx[idx].cpu()); Bs[L].append(rx[idx].cpu())
    for h in hs: h.remove()
    for L in LAYERS:
        X = torch.cat(Xs[L], 0).to(DEV); Y = torch.cat(Ys[L], 0).to(DEV)
        Lx = torch.cat(As[L], 0).to(DEV); Rx = torch.cat(Bs[L], 0).to(DEV)
        AB[L] = (Lx.mean(0, keepdim=True), Rx.mean(0, keepdim=True)); MEANOUT[L] = Y.mean(0, keepdim=True)
        X1 = torch.cat([X, torch.ones(X.shape[0], 1, device=DEV)], 1)
        A = X1.T @ X1 + RIDGE*torch.eye(D+1, device=DEV)
        WLIN[L] = torch.linalg.solve(A, X1.T @ Y)
        Xsh = X[torch.randperm(X.shape[0], device=DEV)]  # shuffled input null
        Xsh1 = torch.cat([Xsh, torch.ones(Xsh.shape[0], 1, device=DEV)], 1)
        WNULL[L] = torch.linalg.solve(Xsh1.T @ Xsh1 + RIDGE*torch.eye(D+1, device=DEV), Xsh1.T @ Y)
        del X, Y, Lx, Rx


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1)
        lp_tok = lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf]
        tot += float(-lp_tok.sum()); n += tf.shape[0]
    return tot/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NCAL + NEVAL)
    calib = rows[:NCAL, :SEQ].contiguous(); blocks = rows[NCAL:NCAL+NEVAL, :SEQ].contiguous()
    fit_surrogates(calib)
    hooks = [m.transformer.h[L].mlp.register_forward_hook(sub_hook_factory(L)) for L in LAYERS]
    MODE['kind'] = None; base = ce(blocks); print(f"baseline CE {base:.4f}", flush=True)
    out = {'baseline': round(base, 4), 'layers': {}}
    for L in LAYERS:
        r = {}
        for kind in ['meanablate', 'bestfit', 'taylor', 'null']:
            MODE['kind'] = kind; MODE['layer'] = L
            r[f'cost_{kind}'] = round(ce(blocks) - base, 4)
        MODE['kind'] = None
        ma = max(r['cost_meanablate'], 1e-6)
        r['frac_bestfit'] = round(1 - r['cost_bestfit']/ma, 3)
        r['frac_taylor'] = round(1 - r['cost_taylor']/ma, 3)
        r['frac_null'] = round(1 - r['cost_null']/ma, 3)
        out['layers'][str(L)] = r
        print(f"L{L:>2}: meanabl {r['cost_meanablate']} bestfit {r['cost_bestfit']} taylor {r['cost_taylor']} null {r['cost_null']} | frac bestfit {r['frac_bestfit']} taylor {r['frac_taylor']} null {r['frac_null']}", flush=True)
    for h in hooks: h.remove()
    fb = float(np.mean([out['layers'][str(L)]['frac_bestfit'] for L in FRONT]))
    ft = float(np.mean([out['layers'][str(L)]['frac_taylor'] for L in FRONT]))
    fn = float(np.mean([out['layers'][str(L)]['frac_null'] for L in FRONT]))
    mb = float(np.mean([out['layers'][str(L)]['frac_bestfit'] for L in MIDDLE]))
    out['front_frac_bestfit'] = round(fb, 3); out['front_frac_taylor'] = round(ft, 3); out['front_frac_null'] = round(fn, 3)
    out['mid_frac_bestfit'] = round(mb, 3)
    out['pred_0_null_zero'] = bool(fn < 0.1)
    out['pred_a_front_bestfit_gg_taylor'] = bool(fb > 0.8 and fb > ft + 0.3)
    out['pred_b_mid_lower_bestfit'] = bool(mb < fb)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"FRONT frac bestfit {fb:.3f} >> taylor {ft:.3f} (null {fn:.3f}) | MID bestfit {mb:.3f}", flush=True)
    print(f"pred_0 null~0 {out['pred_0_null_zero']} | pred_a front bestfit>>taylor {out['pred_a_front_bestfit_gg_taylor']} | pred_b mid<front {out['pred_b_mid_lower_bestfit']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
