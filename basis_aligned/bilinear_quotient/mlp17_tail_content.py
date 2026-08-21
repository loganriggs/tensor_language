"""MLP17 TAIL CONTENT -- characterize the low-variance tail from 660.
mlp17's top-8 output directions (95% variance) recover only 78% of the
loss; ranks 9+ (last 5% of variance) carry ~22%. What is that tail
doing? Hypothesis: the tail is the distributed CONTENT-writing (subword,
657-658), while the rank-1 frequency calibration w_freq sits in the
high-variance HEAD.

Two probes:
  (1) Where is w_freq (calibration) in the variance spectrum? Report its
      squared overlap with the top-r output SVD subspaces (r=1,2,4,8) --
      expect it to sit in the head (high overlap by rank-8).
  (2) What does the tail write? Split the loss recovery of the tail
      (ranks 9-64 vs top-8) by frequent- vs rare-target and by
      content-class targets: keep ONLY the tail (project onto SVD dirs
      9..64) and measure P(subword)/P(capitalized) and freq/rare CE.
      Expect the tail to carry content-writing (subword/rare), the head
      to carry the calibration.

REGISTERED PREDICTIONS:
  (0) SANITY: w_freq's cumulative overlap with the top-8 SVD subspace is
      high (>= 0.6) -- the calibration lives in the high-variance head;
  (a) TAIL = CONTENT/RARE: keeping only the tail (ranks 9-64) recovers
      more of the loss at RARE-target than FREQUENT-target positions
      (the tail serves rare/content prediction), whereas the head (top-8)
      carries the frequency calibration (freq-target effect);
  (b) report w_freq overlaps and head-vs-tail freq/rare loss recovery;
  NULL: a random 56-dim subspace (matched to the tail's dim) recovers far
      less loss than the real tail."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp17_tail_content_results.json'
NFRESH = 48
TOPK = 20

W = {'mu': None, 'Vr': None, 'mode': None}


def hook(mo, i_, o_):
    if W['mode'] is None:
        return o_
    mu = W['mu']
    if W['mode'] == 'mean':
        return mu.expand_as(o_)
    return mu + ((o_ - mu) @ W['Vr']) @ W['Vr'].T


@torch.no_grad()
def ce_split(fresh, is_freq):
    V = m.lm_head.weight.shape[0]
    ces = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        ces[i:i + B] = F.cross_entropy(lg.view(-1, V), tg, reduction='none').view(B, T).cpu()
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

    cap = []
    hk = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    W['mode'] = None
    base_f, base_r = ce_split(fresh, is_freq)
    hk.remove()
    O = torch.cat(cap, 0); mu = O.mean(0)
    Uo, S, Vt = torch.linalg.svd(O - mu, full_matrices=False)
    mu_d = mu.to(DEV); Vt_d = Vt.to(DEV)

    # (1) w_freq overlap with top-r subspaces
    tgt_lf = np.log(freq[nxt] + 1.0); Oc = (O - mu).numpy()
    w = Oc.T @ (tgt_lf - tgt_lf.mean()); w = w / (np.linalg.norm(w) + 1e-9)
    wv = (Vt.numpy() @ w)                             # coords of w in SVD basis
    overlaps = {r: round(float((wv[:r] ** 2).sum()), 4) for r in [1, 2, 4, 8, 16]}
    print(f'w_freq overlap with top-r SVD subspace: {overlaps}', flush=True)

    hk = m.transformer.h[17].mlp.register_forward_hook(hook)
    W['mu'] = mu_d
    W['mode'] = 'mean'; mean_f, mean_r = ce_split(fresh, is_freq)

    def keep(didx):
        W['mode'] = 'proj'; W['Vr'] = Vt_d[didx].T.contiguous()
        return ce_split(fresh, is_freq)
    head_f, head_r = keep(list(range(8)))            # top-8 head
    tail_f, tail_r = keep(list(range(8, 64)))        # ranks 9..64 tail
    g = torch.Generator().manual_seed(0)
    Qr, _ = torch.linalg.qr(torch.randn(D, 56, generator=g));
    W['mode'] = 'proj'; W['Vr'] = Qr.to(DEV); rand_f, rand_r = ce_split(fresh, is_freq)
    hk.remove()

    def rec(cf, cr):
        # fraction of freq / rare loss-benefit recovered
        return ((mean_f - cf) / (mean_f - base_f + 1e-9),
                (mean_r - cr) / (mean_r - base_r + 1e-9))
    head_rec = rec(head_f, head_r); tail_rec = rec(tail_f, tail_r)
    rand_rec = rec(rand_f, rand_r)
    print(f'head(top-8) recovers freq {100*head_rec[0]:.0f}% rare {100*head_rec[1]:.0f}%',
          flush=True)
    print(f'tail(9-64)  recovers freq {100*tail_rec[0]:.0f}% rare {100*tail_rec[1]:.0f}%',
          flush=True)
    print(f'random-56   recovers freq {100*rand_rec[0]:.0f}% rare {100*rand_rec[1]:.0f}%',
          flush=True)

    p0 = overlaps[8] >= 0.6
    pa = tail_rec[1] > tail_rec[0]                    # tail serves rare more than freq
    null_ok = tail_rec[1] > 2 * rand_rec[1]
    print(f'\n(0) w_freq in high-var head (overlap8>=0.6): {p0} ({overlaps[8]})',
          flush=True)
    print(f'(a) tail serves rare > freq: {pa}', flush=True)
    print(f'NULL tail >> random-56: {null_ok}', flush=True)

    out = {'wfreq_overlap_topr': overlaps,
           'head_recover_freq': round(float(head_rec[0]), 3),
           'head_recover_rare': round(float(head_rec[1]), 3),
           'tail_recover_freq': round(float(tail_rec[0]), 3),
           'tail_recover_rare': round(float(tail_rec[1]), 3),
           'random56_recover_rare': round(float(rand_rec[1]), 3),
           'pred_0': bool(p0), 'pred_a_tail_rare': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
