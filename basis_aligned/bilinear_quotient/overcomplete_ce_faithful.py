"""OVERCOMPLETE CE FAITHFUL (the CE version of 742's sparse frontier; also
corrects a phantom "queued" claim in 742 -- this script was never created
then). Train a top-k SAE on mlp1's OUTPUT, substitute its reconstruction
back into the LIVE model, and measure CE recovery -- vs substituting the SVD
rank-k reconstruction. This measures FAITHFULNESS in CE terms (does the
sparse per-datapoint reconstruction preserve the model's loss?), not just L2.

CE-recovery(k) = (CE_ablate - CE_recon)/(CE_ablate - CE_full), for the SAE
reconstruction and the SVD rank-k reconstruction, across k. Faithful = ->1.

REGISTERED PREDICTIONS:
  (0) SANITY: full-output (no replacement) reproduces baseline; ablate is the
      floor;
  (a) SAE MORE CE-FAITHFUL than SVD at same k: the top-k SAE reconstruction
      recovers MORE CE than SVD rank-k at small k (the L2 win of 742 carries
      to CE), for k in {8,32}; report CE-recovery(k) for both;
  (b) report CE-recovery for SAE / SVD across k;
  NULL: random-overcomplete top-k reconstruction is much less CE-faithful."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'overcomplete_ce_faithful_results.json'
NFIT = 128; NEVAL = 48; P = 512; KS = [8, 32, 64]; STEPS = 600
REPL = {'fn': None}   # callable o->o_recon, or 'ablate', or None


def hook(mo, i_, o_):
    if REPL['fn'] is None: return o_
    if REPL['fn'] == 'ablate': return torch.zeros_like(o_)
    flat = o_.float().reshape(-1, D); rec = REPL['fn'](flat)
    return rec.reshape(o_.shape).to(o_.dtype)


@torch.no_grad()
def capture_out(rows, n):
    cap = []
    h = m.transformer.h[LAYER].mlp.register_forward_hook(lambda mo,i_,o_: cap.append(o_.detach().float().reshape(-1,D)))
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
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='mean'))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


def topk_encode(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def train_sae(Otr, k, P, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    We = (torch.randn(D, P, device=DEV)/np.sqrt(D)).requires_grad_(True)
    Wd = (torch.randn(P, D, device=DEV)/np.sqrt(P)).requires_grad_(True)
    b = Otr.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([We, Wd, b], lr=2e-3)
    for s in range(steps):
        z = topk_encode((Otr-b)@We, k); loss = F.mse_loss(z@Wd+b, Otr)
        opt.zero_grad(); loss.backward(); opt.step()
    We=We.detach(); Wd=Wd.detach(); b=b.detach()
    def recon(flat): z = topk_encode((flat-b)@We, k); return z@Wd + b
    return recon


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    Otr = capture_out(fit, NFIT)

    h = m.transformer.h[LAYER].mlp.register_forward_hook(hook)
    REPL['fn'] = None; ce_full = forward_ce(ev, NEVAL)
    REPL['fn'] = 'ablate'; ce_abl = forward_ce(ev, NEVAL); REPL['fn'] = None
    ben = ce_abl - ce_full
    print(f'CE_full {ce_full:.3f}  CE_ablate {ce_abl:.3f}  benefit {ben:.3f}', flush=True)

    mu = Otr.mean(0); res = {'sae': {}, 'svd': {}, 'rand': {}}
    for k in KS:
        with torch.enable_grad(): sae_recon = train_sae(Otr, k, P)
        V = torch.linalg.svd(Otr-mu, full_matrices=False)[2][:k]
        def svd_recon(flat, V=V): return (flat-mu)@V.T@V + mu
        torch.manual_seed(1); Wr = torch.randn(P, D, device=DEV); Wr = Wr/Wr.norm(dim=1,keepdim=True)
        def rand_recon(flat, Wr=Wr): z = topk_encode((flat-mu)@Wr.T, k); return z@Wr + mu
        for tag, fn in [('sae', sae_recon), ('svd', svd_recon), ('rand', rand_recon)]:
            REPL['fn'] = fn; ce = forward_ce(ev, NEVAL); REPL['fn'] = None
            res[tag][k] = round(float((ce_abl - ce)/max(ben,1e-6)),4)
        print(f'k={k:3d}: CE-recovery  SAE {res["sae"][k]:.3f}  SVD {res["svd"][k]:.3f}  '
              f'rand {res["rand"][k]:.3f}', flush=True)
    h.remove()

    pa = all(res['sae'][k] >= res['svd'][k] for k in KS[:2])
    null_ok = all(res['rand'][k] < res['sae'][k] for k in KS)
    print(f'\n(a) SAE >= SVD CE-recovery at small k: {pa}; NULL rand<SAE: {null_ok}', flush=True)
    out = {'ce_full': round(ce_full,4), 'benefit': round(ben,4), 'P': P, 'ks': KS, 'ce_recovery': res,
           'pred_a': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
