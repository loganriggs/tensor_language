"""WEIGHT-ACTION TOP-K SAE across the EARLY LAYERS (750 did only layer 1;
user goal: "understand the first few layers entirely"). Fit the weight-action
top-k SAE on mlp.Down of layers 0..5 and report CE-recovery vs k per layer,
against the A-SVD rank-k reference. Question: is the learned overcomplete sparse
dictionary uniformly the right decomposition across the early stack, or do some
layers resist it (denser / higher effective rank)?

Method per layer L (as 750): reconstruct Down_L's action W@gate with
D@topk_k(E@gate); substitute at the Down output via hook; measure CE-recovery
(CE_ablate - CE_recon)/(CE_ablate - CE_full). Compare to A-SVD rank-k (dense,
faithful) and random-overcomplete top-k (null).

REGISTERED PREDICTIONS:
  (0) SANITY: CE-recovery rises with k in every layer;
  (a) UNIFORM WIN: the weight-action top-k SAE beats A-SVD rank-k at low k
      (k=8) in EVERY early layer (CE-recovery gap > 0.3 at k=8), i.e. the
      overcomplete-sparse decomposition is the right family across the stack,
      not a layer-1 special case; report CE-recovery(k) per layer;
  (b) HETEROGENEITY: report which layers reach >=0.85 CE-recovery soonest (small
      k) vs need large k -- a per-layer "effective sparse rank" profile;
  NULL: random-overcomplete top-k is far worse than the learned SAE in every layer."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'weight_action_multilayer_results.json'
NFIT = 96; NEVAL = 48; P = 2048; KS = [8, 32, 64]; STEPS = 1000
LAYERS = [0, 2, 3, 4, 5]        # layer 1 already in 750; cover the rest of the early stack
REPL = {'fn': None, 'layer': 1}


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    return A, B


def make_hook(L):
    def down_hook(mo, i_, o_):
        if REPL['fn'] is None or REPL['layer'] != L: return o_
        gate = i_[0].float().reshape(-1, HID)
        return REPL['fn'](gate).reshape(o_.shape).to(o_.dtype)
    return down_hook


@torch.no_grad()
def capture_gate(rows, n, L):
    cap = []
    h = m.transformer.h[L].mlp.Down.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


@torch.no_grad()
def forward_ce(rows, n):
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='mean'))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


def train(Wg, Ytrue, k, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    Dm = (torch.randn(D, P, device=DEV)/np.sqrt(D)).requires_grad_(True)
    Em = (torch.randn(P, HID, device=DEV)/np.sqrt(HID)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(steps):
        z = topk(Wg @ Em.T, k); recon = z @ Dm.T + b
        loss = F.mse_loss(recon, Ytrue); opt.zero_grad(); loss.backward(); opt.step()
    Dm = Dm.detach(); Em = Em.detach(); b = b.detach()
    def fn(gate): return topk(gate @ Em.T, k) @ Dm.T + b
    with torch.no_grad():
        r2 = float(1 - ((Ytrue - fn(Wg))**2).sum()/((Ytrue - Ytrue.mean(0))**2).sum())
    return fn, r2


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    hooks = [m.transformer.h[L].mlp.Down.register_forward_hook(make_hook(L)) for L in LAYERS]

    per_layer = {}
    for L in LAYERS:
        W = m.transformer.h[L].mlp.Down.weight.data.float().to(DEV)
        Wg = capture_gate(fit, NFIT, L); Ytrue = Wg @ W.T
        REPL['layer'] = L
        REPL['fn'] = None; ce_full = forward_ce(ev, NEVAL)
        REPL['fn'] = lambda g: torch.zeros(g.shape[0], D, device=DEV); ce_abl = forward_ce(ev, NEVAL); REPL['fn'] = None
        ben = ce_abl - ce_full
        A, B = asvd_fast(W, Wg)
        res = {'wa_topk': {}, 'asvd': {}, 'rand': {}, 'benefit': round(float(ben), 4)}
        for k in KS:
            with torch.enable_grad(): fn, r2 = train(Wg, Ytrue, k)
            REPL['fn'] = fn; ce = forward_ce(ev, NEVAL); REPL['fn'] = None
            res['wa_topk'][k] = {'out_r2': round(r2, 4), 'ce_recovery': round(float((ce_abl - ce)/max(ben, 1e-6)), 4)}
            REPL['fn'] = (lambda A=A, B=B, k=k: (lambda g: g @ (A[:, :k]@B[:k, :]).T))()
            ce_s = forward_ce(ev, NEVAL); REPL['fn'] = None
            res['asvd'][k] = round(float((ce_abl - ce_s)/max(ben, 1e-6)), 4)
            torch.manual_seed(1); Wr = torch.randn(P, HID, device=DEV); Dr = torch.randn(D, P, device=DEV)
            REPL['fn'] = (lambda Wr=Wr, Dr=Dr, k=k, b=Ytrue.mean(0): (lambda g: topk(g@Wr.T, k)@Dr.T + b))()
            ce_r = forward_ce(ev, NEVAL); REPL['fn'] = None
            res['rand'][k] = round(float((ce_abl - ce_r)/max(ben, 1e-6)), 4)
        per_layer[str(L)] = res
        print(f'L{L}: benefit {ben:.3f} | ' + '  '.join(
            f'k{k} WA {res["wa_topk"][k]["ce_recovery"]:.2f}/ASVD {res["asvd"][k]:.2f}/rnd {res["rand"][k]:.2f}' for k in KS), flush=True)
    for h in hooks: h.remove()

    pa = all(per_layer[str(L)]['wa_topk'][8]['ce_recovery'] - per_layer[str(L)]['asvd'][8] > 0.3 for L in LAYERS)
    null_ok = all(per_layer[str(L)]['rand'][k] < per_layer[str(L)]['wa_topk'][k]['ce_recovery'] for L in LAYERS for k in KS)
    out = {'layers': LAYERS, 'ks': KS, 'P': P, 'per_layer': per_layer,
           'pred_a': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) WA-topk beats A-SVD by >0.3 at k=8 in EVERY early layer: {pa}; NULL rand<WA all: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
