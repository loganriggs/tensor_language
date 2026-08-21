"""MLP17 FUNCTIONAL RANK -- answers the user's Q3: how many quadratic
functions does mlp17 effectively compute? Each output dim of the
bilinear MLP is an exact quadratic form x^T M_k x, so the number of
output directions that MATTER = the number of effective quadratic
functions. 615 found the output is rank-8 by VARIANCE (8 dirs = 95% var).
This measures the FUNCTIONAL / LOSS rank: how many top output directions
must be kept to recover the model's loss, which may be smaller (the user
recalls ~4).

Method: capture mlp17 output O; SVD to its top output directions. Replace
mlp17's output with its rank-r reconstruction (mean + projection onto top
r centered directions), for r = 0(mean-ablate),1,2,3,4,5,6,8, and measure
full-model CE. The functional rank is the smallest r recovering 80% / 95%
of full mlp17's loss benefit over the mean-ablated baseline. Control:
random r-dim subspace.

REGISTERED PREDICTIONS:
  (0) SANITY: rank-8 recovers >= 95% of the loss benefit (matches 615's
      variance rank-8);
  (a) LOW FUNCTIONAL RANK (Q3): the smallest r for 80% of the loss
      benefit is small (<= 5) -- mlp17 computes ~4-5 effective quadratic
      functions, fewer than 8;
  (b) report CE and recovered-fraction at each r, and the min-r for
      80%/95%;
  NULL: a RANDOM r-dim subspace of the output recovers far less loss
      benefit than the top-r SVD directions at the same r."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp17_functional_rank_results.json'
NFRESH = 48

W = {'mu': None, 'Vr': None, 'mode': None}       # mode: None|mean|proj


def hook(mo, i_, o_):
    if W['mode'] is None:
        return o_
    mu = W['mu']
    if W['mode'] == 'mean':
        return mu.expand_as(o_)
    Vr = W['Vr']                                  # (D,r)
    c = (o_ - mu) @ Vr
    return mu + c @ Vr.T


@torch.no_grad()
def ce(fresh):
    V = m.lm_head.weight.shape[0]
    tot = 0.0; n = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        l = F.cross_entropy(lg.view(-1, V), tg, reduction='mean')
        tot += float(l) * tg.numel(); n += tg.numel()
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)

    cap = []
    hk = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    W['mode'] = None
    base_ce = ce(fresh)
    hk.remove()
    O = torch.cat(cap, 0)
    mu = O.mean(0)
    U, S, Vt = torch.linalg.svd(O - mu, full_matrices=False)
    var = (S ** 2); cumvar = torch.cumsum(var, 0) / var.sum()
    mu_d = mu.to(DEV); Vt_d = Vt.to(DEV)

    hk = m.transformer.h[17].mlp.register_forward_hook(hook)
    W['mu'] = mu_d

    W['mode'] = 'mean'; mean_ce = ce(fresh)
    benefit = mean_ce - base_ce                   # loss saved by full mlp17
    print(f'full CE {base_ce:.4f}  mean-ablate CE {mean_ce:.4f}  '
          f'benefit {benefit:.4f} nats', flush=True)

    rows = {}
    for r in [1, 2, 3, 4, 5, 6, 8]:
        W['mode'] = 'proj'; W['Vr'] = Vt_d[:r].T.contiguous()
        c = ce(fresh)
        frac = (mean_ce - c) / (benefit + 1e-9)
        rows[r] = {'CE': round(c, 4), 'recovered': round(float(frac), 4),
                   'cumvar': round(float(cumvar[r - 1]), 4)}
        print(f'rank {r}: CE {c:.4f}  recovered {100*frac:.0f}%  '
              f'(cumvar {100*cumvar[r-1]:.0f}%)', flush=True)

    # NULL: random r=4 subspace
    g = torch.Generator().manual_seed(0)
    Qr, _ = torch.linalg.qr(torch.randn(D, 4, generator=g))
    W['mode'] = 'proj'; W['Vr'] = Qr.to(DEV)
    rand4_ce = ce(fresh); rand4_frac = (mean_ce - rand4_ce) / (benefit + 1e-9)
    hk.remove()

    def min_r(thr):
        for r in [1, 2, 3, 4, 5, 6, 8]:
            if rows[r]['recovered'] >= thr:
                return r
        return None
    r80 = min_r(0.8); r95 = min_r(0.95)
    print(f'\nmin-rank for 80% {r80}, for 95% {r95}', flush=True)
    print(f'NULL random-4 recovered {100*rand4_frac:.0f}% vs top-4 '
          f'{100*rows[4]["recovered"]:.0f}%', flush=True)

    p0 = rows[8]['recovered'] >= 0.95
    pa = r80 is not None and r80 <= 5
    null_ok = rand4_frac < 0.6 * rows[4]['recovered']
    print(f'(0) rank-8 >=95%: {p0}; (a) 80% at r<=5: {pa} (r80={r80}); '
          f'NULL: {null_ok}', flush=True)

    out = {'full_CE': round(base_ce, 4), 'mean_ablate_CE': round(mean_ce, 4),
           'benefit_nats': round(benefit, 4), 'ranks': rows,
           'min_rank_80': r80, 'min_rank_95': r95,
           'random4_recovered': round(float(rand4_frac), 4),
           'pred_0': bool(p0), 'pred_a_low_rank': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
