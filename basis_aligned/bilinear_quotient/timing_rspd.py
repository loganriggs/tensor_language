"""TIMING: GPU vs CPU for the RSPD pipeline. Break the cost into its parts
so we know the bottleneck and the expected throughput.
Parts timed:
  A) A-SVD fit = generate_lowrank_approximation (svd of response + pinv of
     activations) -- on CPU vs GPU, for mlp0.Down (in=4608) and block1.attn
     c_proj (in=1152), at N in {3k, 12k, 24k} tokens.
  B) the internal svd vs pinv separately (which dominates).
  C) one CE forward pass (48 rows = ~12k tokens) on GPU -> tokens/sec.
  D) a full component's CE rank-sweep (10 ranks x 2 methods = 20 forwards).
No predictions -- this is a profiling utility."""
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


@torch.no_grad()
def capture_input(mod, rows, n, in_dim, dev):
    cap = []
    h = mod.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, in_dim).to(dev)))
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0)


@torch.no_grad()
def time_forward(rows, n):
    sync(); t = time.time(); ntok = 0
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        ntok += idx.numel()
    sync()
    return time.time() - t, ntok


@torch.no_grad()
def time_asvd(W, X, dev, reps=3):
    Wd = W.to(dev); Xd = X.to(dev)
    # warm + time full A-SVD
    sync(); t = time.time()
    for _ in range(reps):
        A, B = generate_lowrank_approximation(Wd, Xd, target=Xd @ Wd.T)
    sync(); t_full = (time.time() - t) / reps
    # time svd of response and pinv of activations separately
    tgt = (Xd @ Wd.T).T
    sync(); t = time.time()
    for _ in range(reps):
        U, S, Vh = torch.linalg.svd(tgt, full_matrices=False)
    sync(); t_svd = (time.time() - t) / reps
    sync(); t = time.time()
    for _ in range(reps):
        P = torch.linalg.pinv(Xd.T)
    sync(); t_pinv = (time.time() - t) / reps
    return t_full, t_svd, t_pinv


@torch.no_grad()
def main():
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(96)
    print(f'device {DEV}, cuda {torch.cuda.is_available()}', flush=True)

    # C) forward throughput
    for n in [12, 48]:
        dt, ntok = time_forward(rows, n)
        print(f'\n[C] forward {n} rows: {dt:.3f}s for {ntok} tok '
              f'-> {ntok/dt:,.0f} tok/s', flush=True)

    # A/B) A-SVD GPU vs CPU for two components at several N
    comps = [('mlp0.Down', m.transformer.h[0].mlp.Down, 4608),
             ('block1.attn.c_proj', m.transformer.h[1].attn.c_proj, 1152)]
    for name, mod, in_dim in comps:
        W = mod.weight.data.float().cpu()
        for nr in [12, 48, 96]:
            X = capture_input(mod, rows, nr, in_dim, 'cpu')
            ntok = nr * 256
            gf, gs, gp = time_asvd(W, X, DEV)
            cf, cs, cp = time_asvd(W, X, 'cpu', reps=1)
            print(f'\n[A] {name} N={ntok} (X {tuple(X.shape)}):', flush=True)
            print(f'    GPU  A-SVD {gf*1000:7.1f}ms  (svd {gs*1000:6.1f}  pinv {gp*1000:6.1f})',
                  flush=True)
            print(f'    CPU  A-SVD {cf*1000:7.1f}ms  (svd {cs*1000:6.1f}  pinv {cp*1000:6.1f})',
                  flush=True)
            print(f'    speedup GPU/CPU: {cf/gf:.1f}x  (pinv {cp/gp:.1f}x)', flush=True)
            del X

    # D) full component rank-sweep cost (forwards dominate)
    dt, ntok = time_forward(rows, 48)
    n_forwards = 10 * 2 + 2   # 10 ranks x (asvd+random) + fullrank + ablate
    print(f'\n[D] one component rank-sweep ~= {n_forwards} forwards x {dt:.2f}s '
          f'= {n_forwards*dt:.1f}s (+ 1 A-SVD fit)', flush=True)
    print(f'    6-component front map ~= {6*n_forwards*dt:.0f}s of forwards', flush=True)


if __name__ == '__main__':
    main()
