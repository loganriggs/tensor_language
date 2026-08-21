"""QUOTE STATE CAUSAL -- is the mid-network quote-parity register (667)
the CAUSAL driver of quote-tracking behavior, or just a read-correlate
(read!=write, 620)? Remove the quote-parity direction from the residual
after block 6 (its peak) and measure whether the inside-vs-outside
P(closing '"') gap collapses.

w_quote = the linear quote-parity direction fit on the residual after
block 6 (667 gave AUC 0.83 there). Remove it (rank-1) for all positions
after block 6; measure P('"') at inside vs outside positions. Controls:
random rank-1 removal; and (read!=write caution) report whether removal
actually collapses the gap or leaves it (the register may be a correlate
that later layers recompute).

REGISTERED PREDICTIONS:
  (0) SANITY: baseline P('"') inside > outside (reproduces 667: ~3.4x);
  (a) CAUSAL OR CORRELATE (the test): does removing w_quote collapse the
      inside/outside P('"') gap? Report the gap before/after. Registered
      guess: given the read!=write pattern (620, 653), a single linear
      removal may NOT fully collapse it (the parity is recomputed/
      distributed) -- but if it drops the gap >= 50%, the register is a
      causal carrier;
  (b) report gap under baseline / remove-w_quote / remove-random;
  NULL: random rank-1 removal barely changes the gap."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'quote_state_causal_results.json'
NFRESH = 48
QUOTE = 1
REMOVE_AFTER = 6


@torch.no_grad()
def run(fresh, remove_dir, capture=False):
    pq = torch.zeros(NFRESH, T)
    cap = [] if capture else None
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            if li == REMOVE_AFTER:
                if capture:
                    cap.append(x.detach().float().reshape(-1, D).cpu())
                if remove_dir is not None:
                    x = x - (x @ remove_dir)[..., None] * remove_dir
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        pq[i:i + B] = F.softmax(lg, dim=-1)[..., QUOTE].cpu()
    pq = pq.reshape(-1).numpy()
    return (pq, torch.cat(cap, 0).numpy()) if capture else pq


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    toks = fresh[:, :256].numpy()
    inside = np.zeros((NFRESH, T), dtype=np.int64)
    for r in range(NFRESH):
        c = 0
        for j in range(T):
            if int(toks[r, j]) == QUOTE:
                c += 1
            inside[r, j] = c % 2
    inside = inside.reshape(-1)
    A = inside == 1; B_ = inside == 0

    base_pq, X6 = run(fresh, None, capture=True)
    # w_quote from block-6 residual (fit on all, we test causally)
    Xc = X6 - X6.mean(0); yc = inside.astype(np.float64) - inside.mean()
    w = Xc.T @ yc; w = w / (np.linalg.norm(w) + 1e-9)
    w_quote = torch.tensor(w, dtype=torch.float32, device=DEV)

    def gap(pq):
        return float(pq[A].mean() - pq[B_].mean())
    g_base = gap(base_pq)
    g_rm = gap(run(fresh, w_quote))
    rng = np.random.default_rng(0); grand = []
    for s in range(3):
        rr = rng.standard_normal(D); rr /= np.linalg.norm(rr)
        grand.append(gap(run(fresh, torch.tensor(rr, dtype=torch.float32, device=DEV))))
    print(f'baseline gap P(") inside-outside {g_base:+.5f}', flush=True)
    print(f'remove w_quote gap {g_rm:+.5f} (lost {100*(1-g_rm/g_base):.0f}%)', flush=True)
    for s, gr in enumerate(grand):
        print(f'remove random-1 #{s} gap {gr:+.5f} (lost {100*(1-gr/g_base):.0f}%)',
              flush=True)

    p0 = g_base > 0.001
    lost = 1 - g_rm / g_base
    rand_lost = [1 - gr / g_base for gr in grand]
    pa_causal = lost >= 0.5
    null_ok = np.mean(rand_lost) < 0.25
    print(f'\n(0) baseline gap positive: {p0}', flush=True)
    print(f'(a) w_quote removal collapses gap >=50%: {pa_causal} '
          f'(lost {100*lost:.0f}%)', flush=True)
    print(f'NULL random removal barely matters: {null_ok} '
          f'(mean lost {100*np.mean(rand_lost):.0f}%)', flush=True)

    out = {'baseline_gap': round(g_base, 5), 'remove_wquote_gap': round(g_rm, 5),
           'remove_random_gap': [round(x, 5) for x in grand],
           'wquote_lost_frac': round(float(lost), 4),
           'random_lost_frac': [round(float(x), 4) for x in rand_lost],
           'pred_0': bool(p0), 'pred_a_causal': bool(pa_causal), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
