"""WEIGHT-ACTION TOP-K SAE (corrected 749: HARD top-k, not soft L1). Learn
D (Dout x P), E (P x Din) to reconstruct mlp1.Down's ACTION (its output
W@gate) with a SPARSE input-driven code: recon = D @ topk_k(E @ gate).
Minimize ||W@gate - D@topk(E@gate)||^2. This is the weight-action analog of
the 748 activation SAE (encoder is the linear map E of the GATE, tied to the
weight; sparsity ENFORCED by top-k), which should extract the sparse
structure the soft-L1 version (749) could not. Faithfulness = output R^2 +
CE-recovery when the sparse recon replaces Down's output.

Sweep k; compare to A-SVD rank-k (dense) and random.

REGISTERED PREDICTIONS:
  (0) SANITY: reconstruction R^2 rises with k;
  (a) HARD TOP-K WORKS: the weight-action top-k SAE achieves sparse codes
      (k atoms/datapoint) with HIGH CE-recovery at small k (>= 0.7 at k=32),
      unlike the soft-L1 749 (which stayed dense / broke faithfulness), and
      beats A-SVD rank-k at low k (which is catastrophic, 737/748);
  (b) report output-R^2 + CE-recovery per k, vs A-SVD;
  NULL: random-overcomplete top-k is far worse (win from learning)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'weight_action_topk_results.json'
NFIT = 96; NEVAL = 48; P = 2048; KS = [8, 32, 64]; STEPS = 1200
REPL = {'fn': None}


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    return A, B


def down_hook(mo, i_, o_):
    if REPL['fn'] is None: return o_
    gate = i_[0].float().reshape(-1, HID)
    return REPL['fn'](gate).reshape(o_.shape).to(o_.dtype)


@torch.no_grad()
def capture_gate(rows, n):
    cap = []
    h = m.transformer.h[LAYER].mlp.Down.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


@torch.no_grad()
def forward_ce(rows, n):
    s=0.0; nn=0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        s += float(F.nll_loss(lp.reshape(-1,lp.shape[-1]), tgt.reshape(-1), reduction='mean'))*idx.shape[0]; nn+=idx.shape[0]
    return s/nn


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def train(Wg, Ytrue, k, steps=STEPS, seed=0):
    # Wg: (N, HID) gate; Ytrue: (N, D) = gate @ W.T (the weight's action)
    torch.manual_seed(seed)
    Dm = (torch.randn(D, P, device=DEV)/np.sqrt(D)).requires_grad_(True)
    Em = (torch.randn(P, HID, device=DEV)/np.sqrt(HID)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(steps):
        z = topk(Wg @ Em.T, k); recon = z @ Dm.T + b
        loss = F.mse_loss(recon, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
    Dm=Dm.detach(); Em=Em.detach(); b=b.detach()
    def fn(gate): return topk(gate @ Em.T, k) @ Dm.T + b
    with torch.no_grad():
        recon = fn(Wg); r2 = float(1 - ((Ytrue-recon)**2).sum()/((Ytrue-Ytrue.mean(0))**2).sum())
    return fn, r2


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    W = m.transformer.h[LAYER].mlp.Down.weight.data.float().to(DEV)
    Wg = capture_gate(fit, NFIT); Ytrue = Wg @ W.T          # the weight's action (output, no bias)

    h = m.transformer.h[LAYER].mlp.Down.register_forward_hook(down_hook)
    REPL['fn'] = None; ce_full = forward_ce(ev, NEVAL)
    REPL['fn'] = lambda g: torch.zeros(g.shape[0], D, device=DEV); ce_abl = forward_ce(ev, NEVAL); REPL['fn']=None
    ben = ce_abl - ce_full
    A, B = asvd_fast(W, Wg)
    print(f'CE_full {ce_full:.3f} benefit {ben:.3f}', flush=True)

    res = {'wa_topk': {}, 'asvd': {}, 'rand': {}}
    for k in KS:
        with torch.enable_grad(): fn, r2 = train(Wg, Ytrue, k)
        REPL['fn'] = fn; ce = forward_ce(ev, NEVAL); REPL['fn'] = None
        res['wa_topk'][k] = {'out_r2': round(r2,4), 'ce_recovery': round(float((ce_abl-ce)/max(ben,1e-6)),4)}
        # A-SVD rank-k (dense faithful)
        REPL['fn'] = (lambda A=A,B=B,k=k: (lambda g: g @ (A[:,:k]@B[:k,:]).T))()
        ce_s = forward_ce(ev, NEVAL); REPL['fn']=None
        res['asvd'][k] = round(float((ce_abl-ce_s)/max(ben,1e-6)),4)
        # random-OC top-k
        torch.manual_seed(1); Wr = torch.randn(P, HID, device=DEV); Dr = torch.randn(D, P, device=DEV)
        REPL['fn'] = (lambda Wr=Wr,Dr=Dr,k=k,b=Ytrue.mean(0): (lambda g: topk(g@Wr.T,k)@Dr.T + b))()
        ce_r = forward_ce(ev, NEVAL); REPL['fn']=None
        res['rand'][k] = round(float((ce_abl-ce_r)/max(ben,1e-6)),4)
        print(f'k={k:3d}: WA-topk out-R2 {r2:.3f} CE-rec {res["wa_topk"][k]["ce_recovery"]:.3f} | '
              f'A-SVD CE-rec {res["asvd"][k]:.3f} | rand {res["rand"][k]:.3f}', flush=True)
    h.remove()

    pa = res['wa_topk'][32]['ce_recovery'] >= 0.7
    null_ok = all(res['rand'][k] < res['wa_topk'][k]['ce_recovery'] for k in KS)
    print(f'\n(a) WA-topk sparse+CE-faithful (k=32 >=0.7): {pa}; NULL rand<WA: {null_ok}', flush=True)
    out = {'ce_full': round(ce_full,4), 'benefit': round(ben,4), 'P': P, 'ks': KS, 'results': res,
           'pred_a': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
