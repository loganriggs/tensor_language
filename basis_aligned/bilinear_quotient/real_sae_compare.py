"""REAL SAE COMPARE (user: track against the sparser mlp0 + attention). Apply
the validated overcomplete top-k SAE (743-744) to REAL layer OUTPUTS and
compare to SVD rank-k, across layers of different functional rank:
  mlp0 (low-rank r80=8), mlp1 (high-rank r80=128), mlp16 (rank-1),
  block1.attn.c_proj (attention output).
The SAE-vs-SVD R^2 GAP at fixed k measures HIDDEN SPARSE/OVERCOMPLETE
structure: large gap = the layer packs more sparse structure than its SVD
rank suggests; small gap = genuinely low-rank/dense (SVD already captures
it). Usage-Gini of the learned dictionary = how concentrated atom usage is.

REGISTERED PREDICTIONS:
  (0) SANITY: SVD rank-k R^2 orders by the layer's known rank;
  (a) GAP VARIES: report the SAE-vs-SVD R^2 gap per layer at k=8. Register
      the expectation that the HIGH-rank layer (mlp1) has a LARGER SAE
      advantage (more hidden sparse structure) than the genuinely LOW-rank
      layers (mlp0 r80=8, mlp16 rank-1) where SVD-8 already captures most;
  (b) report R^2_sae / R^2_svd / gap / usage-Gini per layer;
  NULL: random-overcomplete top-k does not beat SVD for any layer (win from
      learning)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'real_sae_compare_results.json'
NFIT = 128; NEVAL = 48; P = 512; K = 8; STEPS = 600
COMPONENTS = [('mlp0', 0, 'mlp'), ('mlp1', 1, 'mlp'), ('mlp16', 16, 'mlp'),
              ('block1.attn', 1, 'attn')]


def mod_of(layer, kind):
    blk = m.transformer.h[layer]
    return blk.mlp if kind == 'mlp' else blk.attn.c_proj


@torch.no_grad()
def capture_out(rows, n, layer, kind):
    cap = []
    h = mod_of(layer, kind).register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


def topk_encode(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def train_sae(Otr, Oev, k, P, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    We = (torch.randn(D, P, device=DEV)/np.sqrt(D)).requires_grad_(True)
    Wd = (torch.randn(P, D, device=DEV)/np.sqrt(P)).requires_grad_(True)
    b = Otr.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([We, Wd, b], lr=2e-3)
    for s in range(steps):
        z = topk_encode((Otr-b)@We, k); loss = F.mse_loss(z@Wd + b, Otr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = topk_encode((Oev-b)@We, k); recon = z@Wd + b
        r2 = 1 - ((Oev-recon)**2).sum()/((Oev-Oev.mean(0))**2).sum()
        ztr = topk_encode((Otr-b)@We, k); usage = (ztr.abs()>1e-6).float().mean(0).cpu().numpy()
        pu = np.sort(usage[usage>0]/usage.sum()); gini = float(1-2*np.sum(pu.cumsum())/(pu.sum()*len(pu))+1/len(pu)) if len(pu) else 0
    return float(r2), gini


def r2_svd(Otr, Oev, k):
    with torch.no_grad():
        V = torch.linalg.svd(Otr-Otr.mean(0), full_matrices=False)[2][:k]; mu = Otr.mean(0)
        recon = (Oev-mu)@V.T@V + mu
        return float(1-((Oev-recon)**2).sum()/((Oev-Oev.mean(0))**2).sum())


def r2_rand(Otr, Oev, k, P, seed=1):
    with torch.no_grad():
        torch.manual_seed(seed); Wd = torch.randn(P, D, device=DEV); Wd = Wd/Wd.norm(dim=1,keepdim=True)
        mu = Otr.mean(0); z = topk_encode((Oev-mu)@Wd.T, k); recon = z@Wd + mu
        return float(1-((Oev-recon)**2).sum()/((Oev-Oev.mean(0))**2).sum())


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    res = {}
    for name, L, kind in COMPONENTS:
        Otr = capture_out(rows[:NFIT], NFIT, L, kind); Oev = capture_out(rows[NFIT:NFIT+NEVAL], NEVAL, L, kind)
        with torch.enable_grad():
            r2s, gini = train_sae(Otr, Oev, K, P)
        r2v = r2_svd(Otr, Oev, K); r2r = r2_rand(Otr, Oev, K, P)
        res[name] = {'r2_sae': round(r2s,4), 'r2_svd': round(r2v,4), 'gap': round(r2s-r2v,4),
                     'r2_rand': round(r2r,4), 'usage_gini': round(gini,3)}
        print(f'{name:12s}: SAE {r2s:.3f}  SVD {r2v:.3f}  gap {r2s-r2v:+.3f}  rand {r2r:.3f}  '
              f'usage-gini {gini:.2f}', flush=True)
    gaps = {n: res[n]['gap'] for n in res}
    print(f'\nSAE-vs-SVD gaps: {gaps}', flush=True)
    null_ok = all(res[n]['r2_rand'] < res[n]['r2_sae'] for n in res)
    out = {'k': K, 'P': P, 'components': res, 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'NULL random<SAE all: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
