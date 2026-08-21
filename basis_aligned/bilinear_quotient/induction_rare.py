"""INDUCTION RARE -- is 645's induction TRUE in-context copying, or a
memorized skip-bigram? Restrict to RARE current-tokens A, where a
memorized "A tends to be followed by B" cannot help (A is too rare to
have a reliable stored bigram), so any elevation of P(B) must come from
copying A's OWN earlier occurrence in THIS context.

For each repeat position (current token A occurred earlier at j, target
B = token after A at j), split by the corpus frequency of A. If P(B) is
still strongly elevated for rare A, the mechanism is genuine in-context
induction (match-and-copy), not a stored bigram.

REGISTERED PREDICTIONS:
  (0) SANITY: enough rare-A repeat positions (>=100);
  (a) INDUCTION SURVIVES FOR RARE A: P(B) at rare-A repeat positions is
      >= 10x B's base rate -- in-context copying does not need A to be
      frequent;
  (b) DISTANCE ROBUST: P(B) is still elevated when the earlier
      occurrence is far back (>32 tokens), not only for adjacent
      repeats -- a match-and-copy over distance, not a local n-gram;
  (c) report P(B) by A-frequency bucket and by distance-to-antecedent;
  NULL: a control token C is not elevated for rare A either (the
      elevation is specific to the copied continuation)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'induction_rare_results.json'
NFRESH = 48


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    toks = fresh[:, :257].numpy()
    freq = np.bincount(fresh[:, 1:257].reshape(-1).numpy(), minlength=V)

    # induction targets + antecedent distance + A frequency
    TB = np.full((NFRESH, T), -1, np.int64); TC = np.zeros((NFRESH, T), np.int64)
    DIST = np.zeros((NFRESH, T), np.int64); AFREQ = np.zeros((NFRESH, T))
    for r in range(NFRESH):
        last = {}
        for t in range(T):
            a = int(toks[r, t])
            if a in last:
                j = last[a]
                TB[r, t] = int(toks[r, j + 1]); TC[r, t] = (TB[r, t] + 101) % V
                DIST[r, t] = t - j; AFREQ[r, t] = freq[a]
            last[a] = t

    # forward once, gather P(B), P(C)
    PB = np.full(NFRESH * T, np.nan); PC = np.full(NFRESH * T, np.nan)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        for r in range(B):
            gi = (i + r) * T
            for t in range(T):
                if TB[i + r, t] >= 0:
                    PB[gi + t] = float(p[r, t, TB[i + r, t]])
                    PC[gi + t] = float(p[r, t, TC[i + r, t]])

    valid = TB.reshape(-1) >= 0
    afreq = AFREQ.reshape(-1); dist = DIST.reshape(-1)
    base_all = float(freq[TB.reshape(-1)[valid]].mean() / freq.sum())

    # A-frequency buckets (corpus next-token count of A within this sample)
    fb = {'rare A (<=3)': valid & (afreq <= 3),
          'mid A (4-30)': valid & (afreq > 3) & (afreq <= 30),
          'frequent A (>30)': valid & (afreq > 30)}
    out = {'base_rate_B': round(base_all, 6), 'by_A_freq': {}}
    for k, msk in fb.items():
        if msk.sum() < 5:
            continue
        pb = float(np.nanmean(PB[msk])); pc = float(np.nanmean(PC[msk]))
        out['by_A_freq'][k] = {'P_B': round(pb, 5), 'P_control': round(pc, 6),
                               'n': int(msk.sum())}
        print(f'{k:18s} P(B) {pb:.4f}  P(ctrl) {pc:.6f}  (n={int(msk.sum())})',
              flush=True)

    # distance buckets
    db = {'near (<=8)': valid & (dist <= 8), 'mid (9-32)': valid & (dist > 8) & (dist <= 32),
          'far (>32)': valid & (dist > 32)}
    out['by_distance'] = {}
    for k, msk in db.items():
        if msk.sum() < 5:
            continue
        out['by_distance'][k] = {'P_B': round(float(np.nanmean(PB[msk])), 5),
                                 'n': int(msk.sum())}
        print(f'{k:14s} P(B) {out["by_distance"][k]["P_B"]:.4f} '
              f'(n={out["by_distance"][k]["n"]})', flush=True)

    rare = out['by_A_freq'].get('rare A (<=3)')
    p0 = rare is not None and rare['n'] >= 100
    pa = rare is not None and rare['P_B'] >= 10 * base_all
    far = out['by_distance'].get('far (>32)')
    pb = far is not None and far['P_B'] >= 5 * base_all
    null_ok = rare is not None and rare['P_B'] > 10 * rare['P_control']
    print(f'\n(0) enough rare: {p0}', flush=True)
    print(f'(a) induction survives for rare A (>=10x base {base_all:.5f}): {pa}',
          flush=True)
    print(f'(b) distance-robust (far >=5x base): {pb}', flush=True)
    print(f'NULL control not elevated for rare A: {null_ok}', flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_rare_induction': bool(pa),
                'pred_b_distance_robust': bool(pb), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
