"""LOWRANK ROUTING ISOLATION -- apply the validated behavior-conditioned
low-rank + removal recipe (650/651) to the newline routing (643/644):
can the "this-period-is-a-real-line-end" routing signal be isolated to a
low-rank direction in the post-front residual, where head-ablation only
localized it to a diffuse band?

Recipe: at end-punct positions, the routing is "raise P(newline) when a
newline truly follows." The behavior-conditioned direction is
w_route = cov(residual-after-block-2, newline-follows indicator) at
end-punct positions -- the post-front-stream direction that tracks
whether this period ends a line. Remove it (rank-1) from the residual
after block 2 and measure whether the routing collapses; controls =
random rank-1 removals.

Routing metric (643): R = P(newline | end-punct & newline-follows)
- P(newline | end-punct & not). Baseline ~ +0.26; the context-blind
part is ~0.

REGISTERED PREDICTIONS:
  (0) SANITY: baseline routing R is clearly positive (reproduces 643);
  (a) RANK-1 CARRIES ROUTING: removing w_route collapses R by >= 50%;
  (b) SPECIFIC: removing a random rank-1 direction changes R by < 25%
      (averaged over 3 draws);
  (c) report R for baseline / remove-w_route / remove-random x3;
  NULL: the 3 random removals are tightly clustered near baseline R."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'lowrank_routing_isolation_results.json'
NFRESH = 48
NL1, NL2 = 198, 628
REMOVE_AFTER = 2                      # remove direction from residual after this block


@torch.no_grad()
def forward(fresh, remove_dir, capture_after=None):
    """Return P(newline) per position; if capture_after set, also return
    the residual after that block (Npos, D)."""
    pnl = torch.zeros(NFRESH, T)
    cap = [] if capture_after is not None else None
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            if capture_after is not None and li == capture_after:
                cap.append(x.detach().float().reshape(-1, D).cpu())
            if remove_dir is not None and li == REMOVE_AFTER:
                x = x - (x @ remove_dir)[..., None] * remove_dir
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        pnl[i:i + B] = (p[..., NL1] + p[..., NL2]).cpu()
    pnl = pnl.reshape(-1).numpy()
    if capture_after is not None:
        return pnl, torch.cat(cap, 0).numpy()
    return pnl


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    cur = fresh[:, :256].reshape(-1).numpy()
    nxt = fresh[:, 1:257].reshape(-1).numpy()

    def end_punct(t):
        s = cl.d1(int(t)).strip()
        return len(s) > 0 and s[-1] in '.!?'
    endp = np.array([end_punct(t) for t in cur])
    follows = np.array([chr(10) in cl.d1(int(t)) for t in nxt])
    A = endp & follows; Bm = endp & ~follows
    print(f'{A.sum()} end-punct->newline, {Bm.sum()} end-punct->not', flush=True)

    # pass 1: capture residual after block 2, build w_route
    base_pnl, X2 = forward(fresh, None, capture_after=REMOVE_AFTER)
    ind = follows[endp].astype(np.float64)
    Xe = X2[endp]; Xec = Xe - Xe.mean(0); indc = ind - ind.mean()
    w = Xec.T @ indc; w = w / (np.linalg.norm(w) + 1e-9)
    w_route = torch.tensor(w, dtype=torch.float32, device=DEV)

    def routing(pnl):
        return float(pnl[A].mean() - pnl[Bm].mean())

    R_base = routing(base_pnl)
    R_rm = routing(forward(fresh, w_route))
    g = np.random.default_rng(0); R_rand = []
    for s in range(3):
        rr = g.standard_normal(D); rr /= np.linalg.norm(rr)
        R_rand.append(routing(forward(fresh, torch.tensor(rr, dtype=torch.float32,
                                                           device=DEV))))
    print(f'baseline routing R      {R_base:+.4f}', flush=True)
    print(f'remove w_route          {R_rm:+.4f}  (lost {100*(1-R_rm/R_base):.0f}%)',
          flush=True)
    for s, r in enumerate(R_rand):
        print(f'remove random-1 #{s}     {r:+.4f}  (lost {100*(1-r/R_base):.0f}%)',
              flush=True)

    p0 = R_base > 0.1
    pa = (1 - R_rm / R_base) >= 0.5
    rand_lost = [1 - r / R_base for r in R_rand]
    pb = np.mean(rand_lost) < 0.25
    null_ok = np.std(rand_lost) < 0.2
    print(f'\n(0) baseline routing positive: {p0}', flush=True)
    print(f'(a) remove w_route collapses routing (>=50%): {pa}', flush=True)
    print(f'(b) random removal specific (<25% lost): {pb} '
          f'(mean {100*np.mean(rand_lost):.0f}%)', flush=True)
    print(f'NULL random tight: {null_ok}', flush=True)

    out = {'R_baseline': round(R_base, 4), 'R_remove_wroute': round(R_rm, 4),
           'R_remove_random': [round(r, 4) for r in R_rand],
           'wroute_lost_frac': round(float(1 - R_rm / R_base), 4),
           'random_lost_frac': [round(float(x), 4) for x in rand_lost],
           'pred_0': bool(p0), 'pred_a_rank1_carries': bool(pa),
           'pred_b_specific': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
