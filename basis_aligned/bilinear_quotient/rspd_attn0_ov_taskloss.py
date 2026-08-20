"""RSPD ATTN0 OV TASKLOSS -- the apples-to-apples validation 588
queued, REDESIGNED after a real conceptual bug was caught before it
ran: RSPD's own rank-r surrogate, priced by real cross-entropy in the
running model, against the program's independently-established
16-direction figure for attn0's write.

First design used W = c_proj.weight @ c_v.weight (a "combined OV
matrix" for all 9 heads at once) with X = token embeddings, meant to
be substituted back into the model by hooking c_v to emit W_r @ x and
making c_proj an identity pass-through. That is mathematically
invalid with multiple heads: c_proj is a DENSE matrix mixing all 9
heads' (128-dim) contributions together, while each head has its own
independent attention pattern. output = Wo @ concat_h(attn_weighted_h
(Wv_h @ x)) does NOT equal attn_weighted(Wo @ Wv @ x) for a single
shared pattern unless all 9 heads share the same pattern, which they
generally don't -- so "combined OV" only factors validly per HEAD
(each head's Wo_h @ Wv_h, both slices of the dense matrices, going
through that head's own single pattern), not across all heads at
once. Caught by a quick sanity run (full-rank cost was +4.9 nats,
nowhere near 0) before being queued to bqrunner or written up.

Correct design, mirroring 580's already-validated mlp0-Down approach
exactly: instead of trying to factor the weight product and
re-inject it through hooks (fragile for a multi-head module), capture
attn0's REAL c_proj INPUT (the true attention-weighted, per-head-
combined value vector, over real FineWeb data -- already correctly
computed by the real model, all 9 heads' true patterns baked in) and
apply RSPD to the single honest linear layer W = c_proj.weight with
X = that real captured input. This sidesteps the multi-head
composition problem entirely: c_proj alone is exactly the kind of
"pure linear layer, real activations" pair RSPD's README requires,
and substituting c_proj with its rank-r surrogate via a hook is
unambiguous (one hook, one linear layer, no double-projection).

REGISTERED PREDICTIONS:
  (0) FULL RANK IS FREE: r = min(1152, N) costs < 1e-3 nats -- sanity,
      VOIDS the run on failure;
  (a) THE BALLPARK CHECK: report the smallest r whose RSPD surrogate
      keeps cost under 0.10 nats (the ledger's own bar, ~14796) --
      report whether it lands within 2x of 16 (range 8-32) as an
      informative check, no hard pass/fail pre-registered since this
      is the open question;
  (b) BEATS A MATCHED RANDOM PROJECTION: at that r, RSPD's surrogate
      costs less than a same-rank random-direction projection of
      c_proj, three draws -- the standing "directions matter, not
      just the count" bar (541);
  (c) COMPARE TO THE LEDGER DIRECTLY: report RSPD's r=16 cost next to
      the ledger's own <0.10-nat bar at r=16 -- a genuine cross-tool
      agreement check;
  NULL: cost is monotonically non-increasing as r grows -- a
      surrogate that gets WORSE with more retained directions means
      the reconstruction machinery is broken."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV

sys.path.insert(0, '/workspace/rspd')
from rspd.asvd import generate_lowrank_approximation

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_attn0_ov_taskloss_results.json'
NFRESH = 48
RANKS = [4, 8, 16, 32, 64, 128, 256, 1152]


@torch.no_grad()
def capture_cproj_input(fresh):
    at = m.transformer.h[0].attn
    cap = []
    hk = at.c_proj.register_forward_pre_hook(
        lambda mo_, a_: cap.append(a_[0].detach().float().reshape(-1, D).cpu()))
    for i in range(0, fresh.shape[0], 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    hk.remove()
    return torch.cat(cap, dim=0)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    at = m.transformer.h[0].attn
    W = at.c_proj.weight.float().cpu()

    fresh = cl.fineweb_rows(NFRESH)
    X = capture_cproj_input(fresh)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(X.shape[0], generator=g)[:4000]
    X = X[perm]
    print(f'captured {tuple(X.shape)} real c_proj inputs', flush=True)

    target = X @ W.T
    A_fac, B_fac = generate_lowrank_approximation(W, X, target=target)
    print(f'A-SVD factors: A {tuple(A_fac.shape)}, B {tuple(B_fac.shape)}',
          flush=True)

    def surrogate_hook(r):
        Wr = (A_fac[:, :r] @ B_fac[:r, :]).to(DEV)
        def fh(mo, args, o_):
            X_in = args[0].float()
            return (X_in @ Wr.T).to(o_.dtype)
        return at.c_proj.register_forward_hook(fh)

    def price(r=None):
        ce = torch.zeros(NFRESH, T)
        for i in range(0, NFRESH, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous()
            tg = bb[:, 1:].reshape(-1)
            B = bb.shape[0]
            hk = surrogate_hook(r) if r is not None else None
            x = F.rms_norm(m.transformer.wte(idx), (D,))
            x0 = x
            v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
            ce[i:i + B] = F.cross_entropy(
                lg.view(-1, lg.size(-1)), tg, reduction='none').view(B, T).cpu()
            if hk is not None:
                hk.remove()
        return float(ce.mean())

    base = price(None)
    full = price(1152) - base
    p0 = abs(full) < 1e-3
    print(f'baseline {base:.4f} | (0) full-rank cost {full:+.5f}: '
          f"{'HELD' if p0 else 'FAILED -- VOID'}", flush=True)
    if not p0:
        json.dump({'void': 'full-rank sanity failed', 'full': full},
                   open(OUT, 'w'), indent=1)
        return

    curve = {}
    for r in RANKS:
        c = price(r) - base
        curve[r] = round(c, 5)
        print(f'r={r:>5}: cost {c:+.5f}', flush=True)
        json.dump({'curve': curve}, open(OUT, 'w'), indent=1)

    monotone = all(curve[RANKS[i]] >= curve[RANKS[i + 1]] - 1e-4
                   for i in range(len(RANKS) - 1))
    print(f'NULL (monotone non-increasing): {"ok" if monotone else "CHECK"}',
          flush=True)

    smallest_r = next((r for r in RANKS if curve[r] < 0.10), None)
    pa_range = smallest_r is not None and 8 <= smallest_r <= 32
    print(f'(a) smallest r under 0.10 nats: {smallest_r} '
          f'(informative range 8-32): {"in range" if pa_range else "outside range"}',
          flush=True)

    r_test = smallest_r if smallest_r else 16
    rand_costs = []
    for seed in range(3):
        g2 = torch.Generator().manual_seed(100 + seed)
        rand_dirs = torch.randn(1152, r_test, generator=g2)
        Q, _ = torch.linalg.qr(rand_dirs)
        Wr = (W @ Q @ Q.T).to(DEV)
        def fh(mo, args, o_, Wr=Wr):
            X_in = args[0].float()
            return (X_in @ Wr.T).to(o_.dtype)
        rc = torch.zeros(NFRESH, T)
        for i in range(0, NFRESH, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous()
            tg = bb[:, 1:].reshape(-1)
            B = bb.shape[0]
            hk = at.c_proj.register_forward_hook(fh)
            x = F.rms_norm(m.transformer.wte(idx), (D,))
            x0 = x
            v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
            rc[i:i + B] = F.cross_entropy(
                lg.view(-1, lg.size(-1)), tg, reduction='none').view(B, T).cpu()
            hk.remove()
        rand_costs.append(float(rc.mean()) - base)
    rand_mean = sum(rand_costs) / len(rand_costs)
    pb = curve.get(r_test, curve[RANKS[-1]]) < rand_mean
    print(f'(b) at r={r_test}: RSPD cost {curve.get(r_test)}, random-projection '
          f'cost {rand_mean:+.5f} (3 draws {rand_costs}): '
          f"{'HELD' if pb else 'FAILED'}", flush=True)

    print(f'(c) RSPD r=16 cost {curve.get(16)} vs ledger bar 0.10 '
          f'(ledger: r=16 keeps cost < 0.10 nats): '
          f"{'consistent' if curve.get(16, 999) < 0.10 else 'inconsistent'}",
          flush=True)

    out = {'baseline': base, 'curve': curve, 'pred_0': bool(p0),
           'smallest_r_under_0.10': smallest_r, 'pred_a_range': bool(pa_range),
           'random_projection_costs': rand_costs, 'random_mean': rand_mean,
           'pred_b': bool(pb), 'r16_cost': curve.get(16),
           'null_ok': bool(monotone), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
