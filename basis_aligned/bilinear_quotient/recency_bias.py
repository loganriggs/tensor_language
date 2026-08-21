"""RECENCY BIAS -- probe the central "one knob" claim (668): is the
frequency-calibration bias the ONLY additive/removable linear component,
or is there a SECOND -- a rank-1 "recently-seen token" (recency/
repetition) bias? Distinct from induction (which copies a token's
CONTINUATION); here: does the model additively boost the probability of a
token that itself appeared recently, via a removable rank-1 direction?

For each position, the behavior: is P(a recently-seen token) elevated? We
test a specific removable-knob hypothesis: w_rec = cov(mlp17 output,
"target token appeared in the last 32 positions"). Remove it (rank-1) and
measure whether P(correct token) at repeat-targets drops (a recency knob)
vs random removal. Given the 668 law, the prediction is that recency, if
present, is CONDITIONAL (no removable linear carrier), leaving frequency
calibration as the sole additive knob.

REGISTERED PREDICTIONS:
  (0) SANITY: repeat-targets (token seen in last 32) have higher baseline
      P(correct token) than novel targets (a real recency/repetition
      effect exists);
  (a) NO SECOND KNOB (taxonomy prediction): removing the rank-1 w_rec
      does NOT collapse the repeat-target P(correct token) advantage
      (<25% lost) -- recency is conditional/distributed, not an additive
      removable knob; frequency calibration stays the only linear knob;
  (b) report P(correct token) repeat vs novel, and the removal effect;
  NULL: random rank-1 removal barely changes it."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'recency_bias_results.json'
NFRESH = 48
WINDOW = 32

W = {'dir': None, 'mode': None}


def hook(mo, i_, o_):
    if W['mode'] is None:
        return o_
    d = W['dir']
    return o_ - (o_ @ d)[..., None] * d


@torch.no_grad()
def ptok(fresh, tok_of_pos, remove_dir, capture=False):
    W['mode'] = None if remove_dir is None else 'rm'; W['dir'] = remove_dir
    hk = m.transformer.h[17].mlp.register_forward_hook(hook)
    out = np.zeros(NFRESH * T); cap = [] if capture else None
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        if capture:
            cap.append(F.rms_norm(x, (D,)).detach().float().reshape(-1, D).cpu())
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1).reshape(-1, lg.shape[-1])
        npos = p.shape[0]; base = i * T
        tk = tok_of_pos[base:base + npos]
        out[base:base + npos] = p[np.arange(npos), tk].cpu().numpy()
    hk.remove()
    return (out, np.concatenate(cap, 0)) if capture else out


@torch.no_grad()
def capture_mlp17(fresh):
    cap = []
    hk = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    hk.remove()
    return np.concatenate(cap, 0)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    toks = fresh.numpy()
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    tok_of_pos = nxt.astype(np.int64)

    # recent: target token appeared in the previous WINDOW tokens of the row
    recent = np.zeros(NFRESH * T, dtype=bool)
    for r in range(NFRESH):
        for j in range(T):
            tgt = int(toks[r, j + 1])
            lo = max(0, j - WINDOW + 1)
            if tgt in toks[r, lo:j + 1]:
                recent[r * T + j] = True
    print(f'{recent.sum()} recent-target, {(~recent).sum()} novel', flush=True)

    O = capture_mlp17(fresh)
    Oc = O - O.mean(0); ind = recent.astype(np.float64) - recent.mean()
    w = Oc.T @ ind; w = w / (np.linalg.norm(w) + 1e-9)
    w_rec = torch.tensor(w, dtype=torch.float32, device=DEV)
    rng = np.random.default_rng(0); rr = rng.standard_normal(D); rr /= np.linalg.norm(rr)
    w_rand = torch.tensor(rr, dtype=torch.float32, device=DEV)

    base = ptok(fresh, tok_of_pos, None)
    rm = ptok(fresh, tok_of_pos, w_rec)
    rand = ptok(fresh, tok_of_pos, w_rand)

    def gap(p):
        return float(p[recent].mean() - p[~recent].mean())
    g_base, g_rm, g_rand = gap(base), gap(rm), gap(rand)
    print(f'baseline: recent {base[recent].mean():.4f} novel {base[~recent].mean():.4f} '
          f'(gap {g_base:+.4f})', flush=True)
    print(f'remove w_rec gap {g_rm:+.4f} (lost {100*(1-g_rm/g_base):.0f}%)', flush=True)
    print(f'remove random gap {g_rand:+.4f}', flush=True)

    p0 = g_base > 0.005
    lost = 1 - g_rm / g_base if g_base else 0
    pa_no_knob = lost < 0.25
    null_ok = abs(1 - g_rand / g_base) < 0.25 if g_base else True
    print(f'\n(0) recency effect exists: {p0}', flush=True)
    print(f'(a) NO second knob (recency removal <25%): {pa_no_knob} (lost {100*lost:.0f}%)',
          flush=True)
    print(f'NULL random removal barely matters: {null_ok}', flush=True)

    out = {'n_recent': int(recent.sum()), 'recent_base': round(float(base[recent].mean()), 5),
           'novel_base': round(float(base[~recent].mean()), 5),
           'gap_baseline': round(g_base, 5), 'gap_remove_wrec': round(g_rm, 5),
           'gap_remove_random': round(g_rand, 5), 'removal_lost_frac': round(float(lost), 4),
           'pred_0': bool(p0), 'pred_a_no_second_knob': bool(pa_no_knob),
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
