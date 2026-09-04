"""LOWRANK CALIBRATOR ISOLATION -- the behavior-conditioned low-rank
method (user's Q5), applied to the cleanest component we have (the
block-17 frequency calibrator, 624-629). Question: can we isolate the
calibration to a low-rank (ideally rank-1) subspace of mlp17's output,
where unit-clustering (578-581) and head-selection (649) could not
isolate anything?

Method (direct form of SVD_r(WX)X+ conditioned on the behavior):
  1. Capture mlp17's output O over real text and the next-token log-
     frequency lf. The behavior (calibration) is "shift logits by token
     frequency", so its direction in output space is
     w_freq = normalized cov(O, lf) -- the rank-1 output direction whose
     projection tracks target frequency.
  2. Replace mlp17's output with variants and measure the calibration
     signature = CE at frequent-target vs rare-target positions:
       full            -- baseline
       mean-ablate     -- calibration removed (626: helps freq, hurts rare)
       keep w_freq     -- rank-1: ONLY the frequency direction kept
       remove w_freq   -- rank D-1: the frequency direction removed
       keep random-1   -- control: a random rank-1 direction kept
If the calibration is ~rank-1: keep-w_freq reproduces full's freq/rare CE
split, remove-w_freq collapses it toward mean-ablate, and random-1 does
neither -- a finer-grained isolation than any prior method reached.

REGISTERED PREDICTIONS:
  (0) SANITY: full vs mean-ablate shows the 626 calibration signature
      (mean-ablate lowers freq CE, raises rare CE);
  (a) RANK-1 KEEPS IT: keep-w_freq reproduces >= 60% of full's
      rare-target calibration benefit (rare CE stays near full, far from
      mean-ablate);
  (b) RANK-1 IS NECESSARY: remove-w_freq loses >= 60% of it (rare CE
      moves toward mean-ablate);
  (c) report freq/rare CE for all five variants + the fraction kept;
  NULL: keep-random-1 reproduces < 20% of the calibration -- the
      isolation is specific to w_freq, not any single direction."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'lowrank_calibrator_isolation_results.json'
NFRESH = 48
TOPK = 20

W = {'dir': None, 'mode': None}          # set per variant for the hook


def hook(mo, i_, o_):
    if W['mode'] is None:
        return o_
    if W['mode'] == 'mean':
        return o_.mean(dim=(0, 1), keepdim=True).expand_as(o_)
    d = W['dir']                                   # (D,) unit
    proj = (o_ @ d)[..., None] * d                 # rank-1 projection
    if W['mode'] == 'keep':
        return proj
    if W['mode'] == 'remove':
        return o_ - proj
    return o_


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

    # pass 1: capture mlp17 output, build w_freq
    cap = []
    hk = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    forward_ce(fresh, is_freq)
    hk.remove()
    O = torch.cat(cap, 0).numpy()                 # (Npos, D)
    Oc = O - O.mean(0); lfc = lf - lf.mean()
    w = Oc.T @ lfc
    w = w / (np.linalg.norm(w) + 1e-9)
    w_freq = torch.tensor(w, dtype=torch.float32, device=DEV)
    g = np.random.default_rng(0); rr = g.standard_normal(D); rr /= np.linalg.norm(rr)
    w_rand = torch.tensor(rr, dtype=torch.float32, device=DEV)

    hk = m.transformer.h[17].mlp.register_forward_hook(hook)
    variants = {}
    for name, mode, d in [('full', None, None), ('mean_ablate', 'mean', None),
                          ('keep_wfreq', 'keep', w_freq), ('remove_wfreq', 'remove', w_freq),
                          ('keep_random1', 'keep', w_rand)]:
        W['mode'] = mode; W['dir'] = d
        f, r = forward_ce(fresh, is_freq)
        variants[name] = {'CE_freq': round(f, 4), 'CE_rare': round(r, 4)}
        print(f'{name:14s} freq {f:.4f}  rare {r:.4f}', flush=True)
    hk.remove()

    full = variants['full']; mean = variants['mean_ablate']
    # calibration benefit on rare targets = how much lower rare CE is with
    # calibration present (full) vs removed (mean_ablate)
    benefit = mean['CE_rare'] - full['CE_rare']       # >0 if calibration helps rare
    def frac_kept(v):
        return (mean['CE_rare'] - v['CE_rare']) / (benefit + 1e-9)
    keep_frac = frac_kept(variants['keep_wfreq'])
    rand_frac = frac_kept(variants['keep_random1'])
    remove_lost = 1 - frac_kept(variants['remove_wfreq'])

    p0 = (mean['CE_freq'] < full['CE_freq']) and (mean['CE_rare'] > full['CE_rare'])
    pa = keep_frac >= 0.60
    pb = remove_lost >= 0.60
    null_ok = rand_frac < 0.20
    print(f'\ncalibration rare-benefit (mean-full) = {benefit:+.4f} nats', flush=True)
    print(f'(0) 626 signature (mean-ablate helps freq, hurts rare): {p0}', flush=True)
    print(f'(a) rank-1 keep_wfreq reproduces {100*keep_frac:.0f}% of benefit: {pa}',
          flush=True)
    print(f'(b) remove_wfreq loses {100*remove_lost:.0f}%: {pb}', flush=True)
    print(f'NULL keep_random1 reproduces {100*rand_frac:.0f}% (<20%): {null_ok}',
          flush=True)

    out = {'variants': variants, 'rare_benefit_nats': round(benefit, 4),
           'rank1_keep_frac': round(float(keep_frac), 4),
           'remove_lost_frac': round(float(remove_lost), 4),
           'random1_keep_frac': round(float(rand_frac), 4),
           'pred_0': bool(p0), 'pred_a_rank1_keeps': bool(pa),
           'pred_b_rank1_necessary': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
