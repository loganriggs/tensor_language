"""Verify + time a FASTER A-SVD. The bottleneck (timing_rspd) is
torch.linalg.pinv(X.T), which computes a full SVD of the (d_in x N)
activation matrix -- 95% of the A-SVD cost. But pinv(X.T) is only ever
used as Vh @ pinv(X.T). For d_in <= N (the scaling regime) the Moore-
Penrose right inverse is pinv(X.T) = X @ inv(X.T @ X), so
    B_fac = Vh @ X @ inv(X.T @ X + eps I)
needs only a (d_in x d_in) solve -- NO big SVD, and cost is LINEAR in N.
Validate it reconstructs the same rank-r weight as the library, then time."""
import time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from rspd.asvd import generate_lowrank_approximation
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def asvd_fast(W, X, eps=1e-3):
    # tgt.T = (X @ W.T).T = W @ X.T, shape (M, N); thin SVD is cheap
    tgtT = W @ X.T
    U, S, Vh = torch.linalg.svd(tgtT, full_matrices=False)
    A = U * S                                   # (M, k)
    G = X.T @ X                                  # (d_in, d_in)
    G.diagonal().add_(eps)
    VhX = Vh @ X                                 # (k, d_in)
    B = torch.linalg.solve(G, VhX.T).T           # (k, d_in)
    return A, B


@torch.no_grad()
def capture_input(mod, rows, n, in_dim):
    cap = []
    h = mod.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, in_dim)))
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0)


@torch.no_grad()
def main():
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(96)
    for name, mod, in_dim in [('mlp0.Down', m.transformer.h[0].mlp.Down, 4608),
                              ('block1.attn.c_proj', m.transformer.h[1].attn.c_proj, 1152)]:
        W = mod.weight.data.float().to(DEV)
        X = capture_input(mod, rows, 48, in_dim)      # ~12k tok, d_in <= N
        # correctness: compare rank-r recovered weights
        A0, B0 = generate_lowrank_approximation(W, X, target=X @ W.T)
        A1, B1 = asvd_fast(W, X)
        for r in [1, 8, 64]:
            W0 = A0[:, :r] @ B0[:r, :]
            W1 = A1[:, :r] @ B1[:r, :]
            rel = (W0 - W1).norm() / W0.norm().clamp_min(1e-9)
            print(f'{name} r={r}: rel diff vs library {rel.item():.2e}', flush=True)
        # timing
        sync(); t = time.time()
        for _ in range(5):
            generate_lowrank_approximation(W, X, target=X @ W.T)
        sync(); t_lib = (time.time() - t) / 5
        sync(); t = time.time()
        for _ in range(5):
            asvd_fast(W, X)
        sync(); t_fast = (time.time() - t) / 5
        print(f'{name} N={48*256}: library {t_lib*1000:.1f}ms  fast '
              f'{t_fast*1000:.1f}ms  -> {t_lib/t_fast:.1f}x faster\n', flush=True)


if __name__ == '__main__':
    main()
