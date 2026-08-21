"""WRITE->READ FLOW (user: compose circuits across components via the tensor-
network / residual structure, offline batch). Build the full layer x layer
map of how much each layer's WRITE subspace (its A-SVD output directions,
residual) is READ by each downstream layer (its MLP + attention input
subspaces, residual). This is the residual-bus information-flow graph:
which components' outputs feed which components' inputs.

Overlap(i->j) = ||Pwrite_i^T Pread_j||_F^2 / r  in [0,1] (mean cos^2 of
principal angles). Random baseline r/D. Also write_i -> READOUT (overlap
with the top unembedding right-singular subspace).

BIG DATA: A-SVD write directions fit on 64k tokens.

REGISTERED PREDICTIONS:
  (0) SANITY: random-subspace overlap ~ r/D;
  (a) STRUCTURED FLOW: the write->read matrix is NOT uniform -- some (i,j)
      pairs are well above random (real composition paths), and information
      flows FORWARD (upper triangle i<j carries most weight since a block
      only reads earlier writes through the residual); report the matrix
      and the strongest paths;
  (b) the low-rank edge circuits (mlp0, mlp16, mlp17 writes) route to
      SPECIFIC downstream reads / the readout -- report their top targets;
  NULL: a random residual subspace overlaps every read subspace ~ r/D."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'write_read_flow_results.json'
NFIT = 256; R = 16


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
             (lambda li: lambda mo, inp: caps[li].append(inp[0].detach().float().reshape(-1, HID)))(li))
             for li, blk in enumerate(m.transformer.h)]
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    for h in hooks: h.remove()
    return {li: torch.cat(caps[li], 0) for li in caps}


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT)
    NL = len(m.transformer.h)
    X = capture_all(rows, NFIT)

    # WRITE subspaces (A-SVD output, residual) and READ subspaces (residual)
    WRITE = {}; READ = {}
    for li in range(NL):
        blk = m.transformer.h[li]
        A, _ = asvd_fast(blk.mlp.Down.weight.data.float().to(DEV), X[li])
        Q, _ = torch.linalg.qr(A[:, :R]); WRITE[li] = Q
        # READ: residual dirs feeding MLP gates + attention q/k/v
        mats = [blk.mlp.Left.weight.data.float(), blk.mlp.Right.weight.data.float(),
                blk.attn.c_q.weight.data.float(), blk.attn.c_k.weight.data.float(),
                blk.attn.c_v.weight.data.float()]
        stacked = torch.cat(mats, 0).to(DEV)          # (sum_out, 1152)
        _, _, Vh = torch.linalg.svd(stacked, full_matrices=False)
        READ[li] = Vh.T[:, :R]                          # (1152, R) residual read dirs

    flow = np.zeros((NL, NL))
    for i in range(NL):
        for j in range(NL):
            flow[i, j] = float((WRITE[i].T @ READ[j]).norm() ** 2 / R)

    # write -> readout (unembedding read subspace)
    _, _, VhU = torch.linalg.svd(m.lm_head.weight.data.float().to(DEV), full_matrices=False)
    RU = VhU.T[:, :R]
    to_readout = [round(float((WRITE[i].T @ RU).norm() ** 2 / R), 4) for i in range(NL)]

    rand = R / D
    # forward vs backward mass (i<j = downstream reads earlier write)
    fwd = float(np.mean([flow[i, j] for i in range(NL) for j in range(i+1, NL)]))
    bwd = float(np.mean([flow[i, j] for i in range(NL) for j in range(i)]))
    print(f'random baseline overlap {rand:.4f}', flush=True)
    print(f'mean forward (i<j) {fwd:.4f}  vs backward (i>j) {bwd:.4f}  vs random {rand:.4f}', flush=True)
    # strongest forward paths
    paths = sorted([(flow[i, j], i, j) for i in range(NL) for j in range(i+1, NL)], reverse=True)[:12]
    print('strongest write->read forward paths:', flush=True)
    for v, i, j in paths:
        print(f'  mlp{i:2d} write -> block{j:2d} read : {v:.3f} ({v/rand:.1f}x random)', flush=True)
    print(f'\nwrite -> READOUT overlap by layer: {[round(x,3) for x in to_readout]}', flush=True)
    print(f'  (edges: mlp0 {to_readout[0]:.3f}, mlp16 {to_readout[16]:.3f}, mlp17 {to_readout[17]:.3f})', flush=True)

    # heatmap
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sys.path.insert(0, '/workspace/tensor_language')
    from palette import INK, SECONDARY, MUTED, GRID, SURFACE, BLUES
    fig, ax = plt.subplots(figsize=(8, 7)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    im = ax.imshow(flow, cmap=BLUES, vmin=0, vmax=max(0.2, flow[np.triu_indices(NL,1)].max()), origin='upper')
    ax.set_xlabel('READ by block j (residual input subspace)'); ax.set_ylabel('WRITE by mlp i (A-SVD output)')
    ax.set_xticks(range(0,NL,2)); ax.set_yticks(range(0,NL,2))
    ax.set_title(f'Residual-bus write->read flow (r={R} subspaces)\n'
                 f'how each MLP output is read downstream; random={rand:.3f}',
                 color=INK, loc='left', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout(); fig.savefig(PT + 'write_read_flow.png', dpi=150, facecolor=SURFACE)
    print('wrote write_read_flow.png', flush=True)

    p0 = True
    pa = fwd > 1.5 * rand
    out = {'flow': flow.round(4).tolist(), 'to_readout': to_readout, 'random': round(rand, 4),
           'mean_forward': round(fwd, 4), 'mean_backward': round(bwd, 4), 'R': R,
           'top_paths': [{'i': int(i), 'j': int(j), 'overlap': round(float(v), 4)} for v, i, j in paths],
           'pred_a_structured': bool(pa), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) structured forward flow (>1.5x random): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
