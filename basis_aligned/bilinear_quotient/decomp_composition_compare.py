"""DECOMP COMPOSITION COMPARE (user: run pair-ablation IDENTICALLY on A-SVD
and plain weight-SVD; measure which basis is "better" on downstream metrics
beyond reconstruction). For mlp1, take M components from each of two
decompositions of Down:
  A-SVD (data-conditioned)  vs  weight-SVD (plain SVD of W).
Ablate each single component (project its output direction out of the mlp
output) -> per-datapoint CE damage d_i. Ablate random PAIRS -> d_ij.
Compute three practical metrics for EACH basis:
  1. COMPOSABILITY (interaction sparsity): I(i,j) = ||d_ij - d_i - d_j|| /
     (||d_i||+||d_j||). Mean over pairs. LOWER = more additive/independent
     = easier to compose/edit components separately.
  2. MONOSEMANTICITY (damage concentration): participation ratio of each
     component's |damage| over datapoints, mean over components. LOWER =
     each component hits a more concentrated token set = more interpretable.
  3. per-component damage MAGNITUDE (are components individually meaningful).
Report both bases side by side. (Efficiency/r80 is the separate baseline.)

This is exploratory (the user: orthogonal SVD may not be optimal; we could
later OPTIMIZE a basis for the chosen metric). Goal: quantify where A-SVD
and weight-SVD stand on composability + monosemanticity, not just variance.

REGISTERED PREDICTIONS:
  (0) SANITY: both bases reconstruct the layer at full rank; single-comp
      damages non-trivial in aggregate;
  (a) REPORT (no strong prior): the interaction and concentration metrics
      for A-SVD vs weight-SVD. Register the expectation that A-SVD (data-
      aligned) has LOWER per-component damage concentration (its directions
      are used broadly) while weight-SVD's top comps may be more concentrated
      -- i.e. the reconstruction-optimal basis is not automatically the most
      monosemantic; state which wins each metric;
  NULL: shuffling the component-to-damage assignment removes any metric
      difference (the metrics reflect the basis, not the datapoints)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'decomp_composition_compare_results.json'
NFIT = 96; NDATA = 24        # ~6k datapoints
M = 20; NPAIRS = 40
PROJ = {'d': None}


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps); B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


def hook(mo, i_, o_):
    P = PROJ['d']
    if P is None: return o_
    of = o_.float(); return (of - (of @ P) @ P.T).to(o_.dtype)


@torch.no_grad()
def capture_gate(rows, n):
    cap = []
    h = m.transformer.h[LAYER].mlp.Down.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID).cpu()))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


@torch.no_grad()
def per_tok_ce(rows, n):
    ce = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        ce.append(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='none').cpu())
    return torch.cat(ce).numpy()


def participation(v):
    a = np.abs(v); s = a.sum(); s2 = (a**2).sum()
    return float((s*s)/(s2+1e-12)/len(v))     # in (0,1]; low = concentrated


@torch.no_grad()
def analyze(basis, base, ev):
    # basis: (D, M) orthonormal-ish output directions
    d_single = np.zeros((M, len(base)), dtype=np.float32)
    for i in range(M):
        PROJ['d'] = basis[:, i:i+1]; d_single[i] = per_tok_ce(ev, NDATA) - base; PROJ['d'] = None
    rng = np.random.default_rng(0)
    inter = []
    for _ in range(NPAIRS):
        i, j = rng.choice(M, size=2, replace=False)
        PROJ['d'] = torch.linalg.qr(basis[:, [i, j]])[0]      # orthonormal 2-subspace
        dij = per_tok_ce(ev, NDATA) - base; PROJ['d'] = None
        num = np.linalg.norm(dij - d_single[i] - d_single[j])
        den = np.linalg.norm(d_single[i]) + np.linalg.norm(d_single[j]) + 1e-9
        inter.append(num/den)
    conc = np.mean([participation(d_single[i]) for i in range(M)])
    mag = float(np.mean([np.linalg.norm(d_single[i]) for i in range(M)]))
    return {'interaction': round(float(np.mean(inter)),4), 'concentration': round(float(conc),4),
            'mean_damage_mag': round(mag,4)}


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    allrows = cl.fineweb_rows(NFIT + NDATA)
    fit, ev = allrows[:NFIT], allrows[NFIT:NFIT+NDATA]
    W = m.transformer.h[LAYER].mlp.Down.weight.data.float().to(DEV)
    X = capture_gate(fit, NFIT).to(DEV)
    A, _ = asvd_fast(W, X); A = A[:, :M] / A[:, :M].norm(dim=0, keepdim=True)
    Uw, Sw, Vhw = torch.linalg.svd(W); Wsvd = Uw[:, :M]      # weight-SVD output dirs (already orthonormal)

    h = m.transformer.h[LAYER].mlp.register_forward_hook(hook)
    PROJ['d'] = None; base = per_tok_ce(ev, NDATA)
    res_asvd = analyze(A, base, ev)
    res_wsvd = analyze(Wsvd, base, ev)
    h.remove()

    print(f'metric            A-SVD      weight-SVD', flush=True)
    for k in ['interaction', 'concentration', 'mean_damage_mag']:
        print(f'  {k:16s}{res_asvd[k]:8.4f}   {res_wsvd[k]:8.4f}', flush=True)
    print('\ninterpretation: lower interaction = more composable; lower concentration = more monosemantic', flush=True)
    verdict = {
        'composability_winner': 'A-SVD' if res_asvd['interaction'] < res_wsvd['interaction'] else 'weight-SVD',
        'monosemanticity_winner': 'A-SVD' if res_asvd['concentration'] < res_wsvd['concentration'] else 'weight-SVD'}
    print(f'composability winner: {verdict["composability_winner"]}; '
          f'monosemanticity winner: {verdict["monosemanticity_winner"]}', flush=True)

    out = {'layer': LAYER, 'M': M, 'n_datapoints': len(base), 'asvd': res_asvd,
           'weight_svd': res_wsvd, 'verdict': verdict, 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
