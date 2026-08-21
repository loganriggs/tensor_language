"""LOWRANK CALIBRATOR CONFIRM -- close the two gaps in 650's rank-1
isolation of the block-17 calibration: (1) the specificity control for
the NECESSITY claim (does removing a RANDOM rank-1 direction leave the
calibration intact, unlike removing w_freq?); (2) the sufficiency curve
(how many kept dims reproduce the calibration benefit?).

650 showed removing the rank-1 w_freq = cov(mlp17 out, target log-freq)
collapses the calibration (rare-benefit 103% lost) while keeping a
random rank-1 gives -89%. The clean necessity control is removing a
random rank-1: it should NOT collapse the calibration.

REGISTERED PREDICTIONS:
  (0) SANITY: reproduce 650 -- remove_wfreq collapses the rare-benefit
      (>=80% lost);
  (a) SPECIFIC NECESSITY: removing a RANDOM rank-1 direction leaves the
      rare-benefit largely intact (<25% lost, averaged over 3 draws) --
      so it is w_freq specifically, not any single direction, that
      carries the calibration;
  (b) SUFFICIENCY CURVE: keeping the top-r frequency-aligned dims (r =
      1,2,4,8) recovers an increasing fraction of the rare-benefit --
      report how many dims reach 80% (the calibration's effective rank);
  (c) report rare-benefit fraction lost/kept for each variant;
  NULL: the 3 random-removal draws are tightly clustered near 0% lost
      (removing a random direction is inert for the calibration)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'lowrank_calibrator_confirm_results.json'
NFRESH = 48
TOPK = 20

W = {'dirs': None, 'mode': None}         # mode: None|mean|keep|remove ; dirs:(D,k)


def hook(mo, i_, o_):
    if W['mode'] is None:
        return o_
    if W['mode'] == 'mean':
        return o_.mean(dim=(0, 1), keepdim=True).expand_as(o_)
    Dk = W['dirs']                                 # (D,k) orthonormal
    coef = o_ @ Dk                                 # (...,k)
    proj = coef @ Dk.T                             # (...,D)
    return proj if W['mode'] == 'keep' else o_ - proj


@torch.no_grad()
def forward_ce(fresh, is_freq):
    V = m.lm_head.weight.shape[0]
    ces = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        ce = F.cross_entropy(lg.view(-1, V), tg, reduction='none').view(B, T)
        ces[i:i + B] = ce.cpu()
    ce = ces.reshape(-1).numpy()
    return float(ce[is_freq].mean()), float(ce[~is_freq].mean())


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V)
    top = set(np.argsort(-freq)[:TOPK].tolist())
    is_freq = np.array([t in top for t in nxt])
    lf = np.log(np.maximum(freq[nxt], 1)).astype(np.float64)

    cap = []
    hk = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    forward_ce(fresh, is_freq)
    hk.remove()
    O = torch.cat(cap, 0).numpy()
    Oc = O - O.mean(0); lfc = lf - lf.mean()

    # rank-r frequency-aligned dims: greedily deflate O's covariance with lf
    dirs = []
    R = Oc.copy()
    for _ in range(8):
        w = R.T @ lfc; w = w / (np.linalg.norm(w) + 1e-9)
        dirs.append(w)
        R = R - (R @ w)[:, None] * w[None, :]      # deflate
    Dfull = np.stack(dirs, 1)                       # (D,8)

    def T_(a):
        return torch.tensor(a, dtype=torch.float32, device=DEV)

    hk = m.transformer.h[17].mlp.register_forward_hook(hook)

    def run(mode, dirs_np):
        W['mode'] = mode
        W['dirs'] = T_(dirs_np) if dirs_np is not None else None
        return forward_ce(fresh, is_freq)

    full = {}; full['freq'], full['rare'] = run(None, None)
    mean = {}; mean['freq'], mean['rare'] = run('mean', None)
    benefit = mean['rare'] - full['rare']
    def lost(rare):
        return (rare - full['rare']) / (benefit + 1e-9)

    rw_f, rw_r = run('remove', Dfull[:, :1])
    print(f'full freq {full["freq"]:.4f} rare {full["rare"]:.4f}; '
          f'mean rare {mean["rare"]:.4f}; benefit {benefit:+.4f}', flush=True)
    print(f'remove_wfreq(r1) rare {rw_r:.4f}  lost {100*lost(rw_r):.0f}%', flush=True)

    g = np.random.default_rng(3); rand_lost = []
    for s in range(3):
        rr = g.standard_normal(D); rr /= np.linalg.norm(rr)
        _, r_ = run('remove', rr[:, None])
        rand_lost.append(lost(r_))
        print(f'remove_random(r1) #{s} rare {r_:.4f}  lost {100*lost(r_):.0f}%',
              flush=True)

    keep_curve = {}
    for r in [1, 2, 4, 8]:
        _, kr = run('keep', Dfull[:, :r])
        keep_curve[r] = round(float(lost(kr)), 3)   # lost => kept = 1-lost
        print(f'keep top-{r} rare {kr:.4f}  kept {100*(1-lost(kr)):.0f}%', flush=True)
    hk.remove()

    p0 = lost(rw_r) >= 0.8
    pa = np.mean(rand_lost) < 0.25
    kept8 = 1 - keep_curve[8]
    pb = kept8 >= 0.5
    null_ok = np.std(rand_lost) < 0.2
    print(f'\n(0) remove_wfreq collapses (>=80% lost): {p0}', flush=True)
    print(f'(a) remove_random specific (<25% lost avg): {pa} '
          f'(mean {100*np.mean(rand_lost):.0f}%)', flush=True)
    print(f'(b) keep top-8 recovers >=50%: {pb} (kept {100*kept8:.0f}%)', flush=True)
    print(f'NULL random-removal tight: {null_ok}', flush=True)

    out = {'full': full, 'mean_ablate': mean, 'benefit': round(benefit, 4),
           'remove_wfreq_lost': round(float(lost(rw_r)), 3),
           'remove_random_lost': [round(float(x), 3) for x in rand_lost],
           'keep_rank_lost': keep_curve,
           'pred_0': bool(p0), 'pred_a_specific': bool(pa),
           'pred_b_sufficiency': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
