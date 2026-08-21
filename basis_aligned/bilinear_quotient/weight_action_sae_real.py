"""WEIGHT-ACTION SAE on the REAL mlp1.Down (747 validated the method on a
toy; apply to a real layer). Factor W = mlp1.Down (1152 x 4608) as D@E
overcomplete (P=2048 > rank 1152), minimizing ||W - D@E||_F^2 + lambda *
mean_i ||E x_i||_1 over the real gate data X. Measure the FAITHFULNESS-vs-
SPARSITY frontier (sweep lambda): weight-recon error ||W-DE||/||W||, code
sparsity (effective L0), and CE-faithfulness (substitute D@E as Down.weight,
measure CE recovery). Compare to A-SVD (dense, faithful) as the reference.

Question: does a REAL layer weight admit a FAITHFUL SPARSE OVERCOMPLETE
decomposition (low ||W-DE|| AND sparse codes AND CE preserved), or is real
structure too dense for sparsity without a faithfulness cost?

REGISTERED PREDICTIONS:
  (0) SANITY: lambda=0 reaches low weight-recon error (D@E can fit W since
      P>rank) but DENSE codes;
  (a) FRONTIER: increasing lambda makes codes SPARSER (lower L0) at rising
      weight-recon error / falling CE-recovery -- a real tradeoff. Report the
      frontier. Register the OPEN question of whether a useful point exists
      (sparse codes with CE-recovery still high);
  (b) report ||W-DE||, code-L0, CE-recovery per lambda + A-SVD reference;
  NULL: lambda=0 codes are dense (L0 >> a few)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'weight_action_sae_real_results.json'
NFIT = 48; NEVAL = 48; P = 2048; STEPS = 1500; LAMS = [0.0, 1e-3, 1e-2]


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    return A, B


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
def forward_ce(rows, n, Wsub=None):
    mod = m.transformer.h[LAYER].mlp.Down; orig = mod.weight.data
    if Wsub is not None: mod.weight.data = (torch.zeros_like(orig) if Wsub=='ablate' else Wsub.to(orig.dtype))
    s=0.0; nn=0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        s += float(F.nll_loss(lp.reshape(-1,lp.shape[-1]), tgt.reshape(-1), reduction='mean'))*idx.shape[0]; nn+=idx.shape[0]
    if Wsub is not None: mod.weight.data = orig
    return s/nn


def eff_l0(codes, thresh=0.05):
    a = codes.abs(); mx = a.max(1, keepdim=True).values.clamp_min(1e-9)
    return float((a > thresh*mx).float().sum(1).mean())


def train_wa(W, Xs, lam, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    Dm = (torch.randn(D, P, device=DEV)/np.sqrt(D)).requires_grad_(True)
    Em = (torch.randn(P, HID, device=DEV)/np.sqrt(HID)).requires_grad_(True)
    opt = torch.optim.Adam([Dm, Em], lr=5e-3)
    for s in range(steps):
        wloss = F.mse_loss(Dm@Em, W)
        codes = Xs @ Em.T
        loss = wloss + lam*codes.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        faith = float((W - Dm@Em).norm()/W.norm()); codes = Xs @ Em.T
    return Dm.detach(), Em.detach(), faith, eff_l0(codes)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    W = m.transformer.h[LAYER].mlp.Down.weight.data.float().to(DEV)
    Xg = capture_gate(fit, NFIT)
    Xs = Xg[:6000]                     # subset for the sparsity term
    ce_full = forward_ce(ev, NEVAL); ce_abl = forward_ce(ev, NEVAL, 'ablate'); ben = ce_abl - ce_full
    print(f'CE_full {ce_full:.3f} benefit {ben:.3f}', flush=True)

    res = {}
    for lam in LAMS:
        with torch.enable_grad(): Dm, Em, faith, l0 = train_wa(W, Xs, lam)
        cer = float((ce_abl - forward_ce(ev, NEVAL, Dm@Em))/max(ben,1e-6))
        res[str(lam)] = {'weight_recon_err': round(faith,4), 'code_l0': round(l0,2), 'ce_recovery': round(cer,4)}
        print(f'lambda={lam}: ||W-DE||/||W|| {faith:.3f}  code-L0 {l0:.1f}/{P}  CE-recovery {cer:.3f}', flush=True)
    # A-SVD reference (full-rank faithful, dense)
    A, B = asvd_fast(W, Xg.to(DEV))
    cer_asvd = float((ce_abl - forward_ce(ev, NEVAL, A@B))/max(ben,1e-6))
    print(f'A-SVD full-rank: CE-recovery {cer_asvd:.3f} (dense, faithful reference)', flush=True)

    p0 = res['0.0']['weight_recon_err'] < 0.2 and res['0.0']['code_l0'] > 50
    null_ok = res['0.0']['code_l0'] > 4*res[str(LAMS[-1])]['code_l0']
    out = {'ce_full': round(ce_full,4), 'benefit': round(ben,4), 'P': P, 'lambdas': LAMS,
           'frontier': res, 'asvd_ce_recovery': round(cer_asvd,4),
           'pred_0': bool(p0), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'\n(0) lambda=0 fits W + dense: {p0}; NULL sparsity reduces L0: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
