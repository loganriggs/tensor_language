"""W_FREQ GENERALIZATION -- verify the robustness of the one clean result
(the rank-1 frequency-calibration knob, 650-651). Is w_freq a stable
MODEL property, or an artifact of this data slice? Compute w_freq on two
DISJOINT halves of the corpus and test: (1) are the two directions the
same (high cosine)? (2) does w_freq fit on half A still remove the
calibration when applied on half B (cross-removal)?

REGISTERED PREDICTIONS:
  (0) SANITY: on the full data, removing w_freq collapses the calibration
      (reproduces 651: rare-benefit >=80% lost);
  (a) STABLE DIRECTION: cos(w_freq_A, w_freq_B) from disjoint data halves
      is high (>= 0.8) -- the calibration axis is a model property, not
      data noise;
  (b) CROSS-REMOVAL: w_freq fit on half A, applied (removed) on half B,
      collapses B's calibration nearly as much as B's own w_freq does
      (>= 70% of the self-removal effect);
  (c) report cos and cross-removal fractions;
  NULL: a w_freq fit on SHUFFLED frequency labels does not cross-remove
      (random-level)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'w_freq_generalization_results.json'
NFRESH = 96                      # larger, split into two disjoint halves
TOPK = 20

W = {'dir': None}


def hook(mo, i_, o_):
    if W['dir'] is None:
        return o_
    d = W['dir']
    return o_ - (o_ @ d)[..., None] * d


@torch.no_grad()
def capture_and_ce(fresh, rows, is_freq_full, freq_full, remove_dir, want_O=False):
    """Run over the given row indices; return (freq CE, rare CE, [O])."""
    W['dir'] = remove_dir
    hk = m.transformer.h[17].mlp.register_forward_hook(hook)
    V = m.lm_head.weight.shape[0]
    ces = []; labels = []; cap = [] if want_O else None
    capO = []
    hk2 = None
    if want_O:
        hk2 = m.transformer.h[17].mlp.register_forward_hook(
            lambda mo, i_, o_: capO.append(o_.detach().float().reshape(-1, D).cpu()))
    for s in range(0, len(rows), 4):
        ridx = rows[s:s + 4]
        bb = fresh[ridx, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        ce = F.cross_entropy(lg.view(-1, V), tg, reduction='none').cpu().numpy()
        ces.append(ce)
        labels.append(np.array([is_freq_full[int(t)] for t in tg.cpu().numpy()]))
    hk.remove()
    if hk2:
        hk2.remove()
    ce = np.concatenate(ces); lab = np.concatenate(labels)
    fce = float(ce[lab].mean()); rce = float(ce[~lab].mean())
    if want_O:
        return fce, rce, np.concatenate(capO, 0)
    return fce, rce


@torch.no_grad()
def wfreq_from(fresh, rows, freq_full):
    W['dir'] = None
    cap = []
    hk = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    tgts = []
    for s in range(0, len(rows), 4):
        ridx = rows[s:s + 4]
        bb = fresh[ridx, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        tgts.append(bb[:, 1:].reshape(-1).cpu().numpy())
    hk.remove()
    O = np.concatenate(cap, 0); tg = np.concatenate(tgts)
    lf = np.log(freq_full[tg] + 1.0); Oc = O - O.mean(0); tc = lf - lf.mean()
    w = Oc.T @ tc; return w / (np.linalg.norm(w) + 1e-9)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq_full = np.bincount(nxt, minlength=V)
    top = set(np.argsort(-freq_full)[:TOPK].tolist())
    is_freq_full = np.array([1 if t in top else 0 for t in range(V)], dtype=bool)

    rowsA = list(range(0, NFRESH, 2)); rowsB = list(range(1, NFRESH, 2))
    wA = wfreq_from(fresh, rowsA, freq_full)
    wB = wfreq_from(fresh, rowsB, freq_full)
    cos = float(wA @ wB)
    # shuffled-label control on half A
    rng = np.random.default_rng(0)
    wsh = wfreq_from(fresh, rowsA, freq_full[rng.permutation(V)])
    cos_sh = float(wA @ wsh)

    # self-removal on B with wB; cross-removal on B with wA (removing the
    # rank-1 direction raises rare CE if it carried calibration)
    f0, r0 = capture_and_ce(fresh, rowsB, is_freq_full, freq_full, None)
    f_self, r_self = capture_and_ce(fresh, rowsB, is_freq_full, freq_full,
                                    torch.tensor(wB, dtype=torch.float32, device=DEV))
    f_cross, r_cross = capture_and_ce(fresh, rowsB, is_freq_full, freq_full,
                                      torch.tensor(wA, dtype=torch.float32, device=DEV))
    rng2 = np.random.default_rng(1); rr = rng2.standard_normal(D); rr /= np.linalg.norm(rr)
    f_rand, r_rand = capture_and_ce(fresh, rowsB, is_freq_full, freq_full,
                                    torch.tensor(rr, dtype=torch.float32, device=DEV))
    self_effect = r_self - r0
    cross_effect = r_cross - r0
    rand_effect = r_rand - r0
    cross_frac = cross_effect / (self_effect + 1e-9)
    print(f'cos(w_freq_A, w_freq_B) = {cos:.3f}  (shuffled control {cos_sh:.3f})',
          flush=True)
    print(f'on B: self-removal rare-CE +{self_effect:.4f}, cross-removal (wA) '
          f'+{cross_effect:.4f}, random +{rand_effect:.4f}', flush=True)
    print(f'cross/self effect = {cross_frac:.2f}', flush=True)

    pa = cos >= 0.8
    pb = cross_frac >= 0.7
    null_ok = abs(cos_sh) < 0.3 and abs(rand_effect) < 0.3 * abs(self_effect)
    print(f'\n(a) stable direction (cos>=0.8): {pa}', flush=True)
    print(f'(b) cross-removal >=70% of self: {pb}', flush=True)
    print(f'NULL shuffled cos low & random removal small: {null_ok}', flush=True)

    out = {'cos_AB': round(cos, 4), 'cos_shuffled': round(cos_sh, 4),
           'self_removal_rareCE': round(self_effect, 4),
           'cross_removal_rareCE': round(cross_effect, 4),
           'random_removal_rareCE': round(rand_effect, 4),
           'cross_over_self': round(float(cross_frac), 4),
           'pred_a_stable': bool(pa), 'pred_b_cross_removes': bool(pb),
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
