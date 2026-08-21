"""READ/WRITE SUBSPACE OVERLAP (user Q, offline batch). For each MLP, does
it WRITE into the same residual subspace it READS from, or are they mostly
ORTHOGONAL? This is the read!=write theme (619-622, 676) at the subspace
level across depth.

Definitions (all in residual space, D=1152):
  WRITE subspace = top-r output directions of Down (activation-conditioned
    A-SVD on 64k tokens): span(A[:, :r]).
  READ subspace  = residual directions the layer is sensitive to = top-r
    right singular vectors of the stacked gate maps [W_Left; W_Right]
    (9216 x 1152) -- the residual directions that drive the bilinear gate.
  Overlap(r) = ||P_write^T P_read||_F^2 / r  in [0,1]  (mean cos^2 of the
    principal angles; 0 = orthogonal, 1 = identical). Random baseline r/D.
Also per top-component: cosine between the component's write direction
A[:,k] and the nearest read direction (max cos onto the READ subspace).

BIG DATA: A-SVD fit on 64k tokens (user: use lots of tokens, it's cheap).

REGISTERED PREDICTIONS:
  (0) SANITY: random-subspace overlap ~ r/D (=r/1152);
  (a) READ vs WRITE: report overlap per layer at r=32. Register the
      expectation (read!=write) that overlap is LOW (mostly orthogonal,
      < 3x the random baseline) for most layers -- the MLP moves info from
      one residual subspace to another; note any layer that instead writes
      back onto its read subspace (overlap high = self-amplifier);
  (b) report overlap(r) per layer + per-component write->read max cosine;
  NULL: the random-subspace overlap curve (r/D)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'read_write_overlap_results.json'
NFIT = 128   # ~65k tokens
RGRID = [1, 4, 8, 16, 32, 64]


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps); B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


@torch.no_grad()
def capture_all(rows, n):
    caps = {li: [] for li in range(len(m.transformer.h))}
    hooks = [blk.mlp.Down.register_forward_pre_hook(
             (lambda li: lambda mo, inp: caps[li].append(inp[0].detach().float().reshape(-1, HID).cpu()))(li))
             for li, blk in enumerate(m.transformer.h)]
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    for h in hooks: h.remove()
    return {li: torch.cat(caps[li], 0) for li in caps}


def orthonormal(Mat):
    Q, _ = torch.linalg.qr(Mat)
    return Q


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT)
    NL = len(m.transformer.h)
    X = capture_all(rows, NFIT)
    print(f'captured {X[0].shape[0]} tokens', flush=True)

    prof = {}
    for li in range(NL):
        blk = m.transformer.h[li].mlp
        W = blk.Down.weight.data.float().to(DEV)          # (1152, 4608)
        WL = blk.Left.weight.data.float().to(DEV)         # (4608, 1152)
        WR = blk.Right.weight.data.float().to(DEV)
        # WRITE dirs: A-SVD output basis (residual)
        A, _ = asvd_fast(W, X[li].to(DEV))                         # A: (1152, k)
        Aw = A / A.norm(dim=0, keepdim=True).clamp_min(1e-9)
        # READ dirs: right singular vectors of [WL; WR] (residual)
        stacked = torch.cat([WL, WR], 0)                  # (9216, 1152)
        _, _, Vh = torch.linalg.svd(stacked, full_matrices=False)
        Vr = Vh.T                                          # (1152, 1152) residual read dirs
        row = {}
        for r in RGRID:
            Pw = orthonormal(Aw[:, :r])                    # (1152, r)
            Pr = Vr[:, :r]                                  # already orthonormal
            ov = float((Pw.T @ Pr).norm() ** 2 / r)
            row[r] = round(ov, 4)
        # per-component write->read max cosine (top 8 write comps onto full read space? use top-64 read)
        Pr64 = Vr[:, :64]
        comp_maxcos = [round(float((Aw[:, k] @ Pr64).norm()), 3) for k in range(8)]
        prof[li] = {'overlap': row, 'comp_write_read_maxcos_top8': comp_maxcos}
        print(f'mlp{li:2d}: overlap@r ' + ' '.join(f'{r}:{row[r]:.3f}' for r in RGRID)
              + f'  | top-comp write->read cos {comp_maxcos[:4]}', flush=True)

    rand = {r: round(r / D, 4) for r in RGRID}
    ov32 = [prof[li]['overlap'][32] for li in range(NL)]
    mostly_orth = float(np.mean(ov32)) < 3 * rand[32]
    print(f'\nrandom baseline overlap@r: {rand}', flush=True)
    print(f'overlap@32 by layer: {[round(x,3) for x in ov32]}', flush=True)
    print(f'mean overlap@32 {np.mean(ov32):.3f} vs random {rand[32]:.3f} '
          f'-> mostly orthogonal (<3x random): {mostly_orth}', flush=True)

    out = {'profile': prof, 'random_overlap': rand, 'overlap32_by_layer': ov32,
           'mean_overlap32': round(float(np.mean(ov32)), 4), 'n_tokens': NFIT * 256,
           'mostly_orthogonal': bool(mostly_orth), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
