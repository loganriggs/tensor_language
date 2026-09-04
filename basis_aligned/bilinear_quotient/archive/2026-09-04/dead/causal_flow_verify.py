"""CAUSAL FLOW VERIFY (verify the 723 composition map is CAUSAL, not just
geometric). The write->read overlap says layer i's output COULD be read by
block j. Test causally: project OUT layer i's top-r WRITE subspace from the
residual right after block i, and measure how much block j's GATE (its
encoder input) changes -- vs projecting out a RANDOM same-rank subspace. If
the geometric flow is causal, the write-aligned ablation perturbs the
downstream gate MORE than random.

Tested paths (strong in 723): mlp0->blk1, mlp5->blk6, mlp15->blk16,
mlp15->blk17, mlp1->blk17. Metric: ||Delta gate_j|| / ||gate_j|| (relative
downstream gate perturbation), write-subspace vs random-subspace ablation.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating a random subspace perturbs the downstream gate a
      little (baseline);
  (a) CAUSAL: the WRITE-subspace ablation perturbs block j's gate
      substantially MORE than the random-subspace ablation (ratio >= 1.5)
      for the strong adjacent paths -- the composition is causal, not just
      geometric;
  (b) report the perturbation ratio (write/random) per path;
  NULL: for a NON-path (backward / far, low geometric overlap), the write
      vs random ratio is ~1 (no privileged causal flow)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608; R = 16
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'causal_flow_verify_results.json'
NFIT = 64; NEVAL = 32
PATHS = [(0, 1), (5, 6), (15, 16), (15, 17), (1, 17), (0, 17), (10, 3)]  # last two: hub / backward null


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps); B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


PROJ = {'layer': None, 'P': None}   # project OUT span(P) from residual after block `layer`


@torch.no_grad()
def capture_gate(rows, n, jlayer):
    """forward, optionally project-out PROJ['P'] after block PROJ['layer'],
    capture block jlayer's gate (Down input)."""
    gate = []
    h = m.transformer.h[jlayer].mlp.Down.register_forward_pre_hook(
        lambda mo, inp: gate.append(inp[0].detach().float().reshape(-1, HID)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            if PROJ['layer'] == li and PROJ['P'] is not None:
                P = PROJ['P']                       # (D, r) orthonormal
                x = x - (x @ P) @ P.T
    h.remove()
    return torch.cat(gate, 0)


@torch.no_grad()
def capture_write_gate(rows, n, ilayer):
    """gate of layer i (for its A-SVD write basis)."""
    g = []
    h = m.transformer.h[ilayer].mlp.Down.register_forward_pre_hook(
        lambda mo, inp: g.append(inp[0].detach().float().reshape(-1, HID).cpu()))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(g, 0)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT); ev = cl.fineweb_rows(NEVAL)

    # write bases per source layer
    Wbasis = {}
    for i in set(p[0] for p in PATHS):
        gi = capture_write_gate(rows, NFIT, i).to(DEV)
        A, _ = asvd_fast(m.transformer.h[i].mlp.Down.weight.data.float().to(DEV), gi)
        Q, _ = torch.linalg.qr(A[:, :R]); Wbasis[i] = Q
        del gi
    g = torch.Generator().manual_seed(0)
    Rand = {i: torch.linalg.qr(torch.randn(D, R, generator=g))[0].to(DEV) for i in Wbasis}

    results = []
    for (i, j) in PATHS:
        # baseline gate_j (no projection)
        PROJ['layer'] = None; PROJ['P'] = None
        g0 = capture_gate(ev, NEVAL, j)
        # write-subspace ablation
        PROJ['layer'] = i; PROJ['P'] = Wbasis[i]
        gw = capture_gate(ev, NEVAL, j)
        dw = float((gw - g0).norm() / g0.norm().clamp_min(1e-9))
        # random-subspace ablation
        PROJ['P'] = Rand[i]
        gr = capture_gate(ev, NEVAL, j)
        dr = float((gr - g0).norm() / g0.norm().clamp_min(1e-9))
        PROJ['layer'] = None; PROJ['P'] = None
        ratio = dw / max(dr, 1e-9)
        results.append({'i': i, 'j': j, 'write_pert': round(dw, 4), 'rand_pert': round(dr, 4),
                        'ratio': round(ratio, 2)})
        print(f'mlp{i:2d} write ablated -> blk{j:2d} gate change: write {dw:.3f}  '
              f'random {dr:.3f}  ratio {ratio:.2f}', flush=True)

    adj = [r for r in results if r['j'] == r['i'] + 1]
    causal = all(r['ratio'] >= 1.5 for r in adj)
    backward = [r for r in results if r['j'] < r['i']]
    null_ok = all(r['ratio'] < 1.5 for r in backward) if backward else True
    print(f'\n(a) adjacent paths causal (ratio>=1.5): {causal}', flush=True)
    print(f'NULL backward/non-path ratio ~1: {null_ok}', flush=True)

    out = {'paths': results, 'pred_a_causal': bool(causal), 'null_ok': bool(null_ok),
           'R': R, 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
