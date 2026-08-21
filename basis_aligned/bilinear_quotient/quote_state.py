"""QUOTE STATE -- a fresh, untouched behavior for breadth: does the model
track QUOTATION PARITY (whether the current position is inside or outside
a quote), and is that state linearly present in the residual? A quote-
state feature is a classic "context register" -- distinct from the token-
class/frequency machinery mapped so far.

For each position, compute quote-parity = whether an odd number of "
(double-quote, token id 1) have appeared so far in the row (inside a
quote) vs even (outside). Test:
  (1) Is quote-state linearly decodable from the residual (probe AUC),
      and at what depth does it become available?
  (2) Does it causally matter: at inside-quote vs outside-quote
      positions, is P(closing quote ") elevated inside?

REGISTERED PREDICTIONS:
  (0) SANITY: enough inside-quote and outside-quote positions (>=100
      each) across the corpus;
  (a) STATE IS TRACKED: quote-parity is linearly decodable from the final
      residual with AUC >= 0.8 (a real context register), far above a
      shuffled-label control;
  (b) DEPTH: report the probe AUC at the residual after blocks 2, 6, 12,
      17 -- when does the model know it is inside a quote?
  (c) BEHAVIORAL: P(closing '"') is higher at inside-quote positions than
      outside-quote positions;
  NULL: a shuffled quote-parity label is not decodable (AUC ~ 0.5)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'quote_state_results.json'
NFRESH = 48
QUOTE = 1                       # '"' token id
CAP_LAYERS = [2, 6, 12, 17]


def auc(scores, labels):
    order = np.argsort(scores); ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos = labels == 1; npos = pos.sum(); nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    return float((ranks[pos].sum() - npos * (npos - 1) / 2) / (npos * nneg))


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    toks = fresh[:, :256].numpy()

    # quote parity per position: # of '"' seen up to and including this pos, odd=inside
    inside = np.zeros((NFRESH, T), dtype=np.int64)
    for r in range(NFRESH):
        c = 0
        for j in range(T):
            if int(toks[r, j]) == QUOTE:
                c += 1
            inside[r, j] = c % 2
    inside = inside.reshape(-1)
    n_in = int(inside.sum()); n_out = int((1 - inside).sum())
    print(f'{n_in} inside-quote, {n_out} outside-quote positions', flush=True)

    # capture residuals at chosen depths + final logits
    caps = {li: [] for li in CAP_LAYERS}
    pql = torch.zeros(NFRESH, T)     # P(closing quote)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            if li in CAP_LAYERS:
                caps[li].append(x.detach().float().reshape(-1, D).cpu())
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        pql[i:i + B] = p[..., QUOTE].cpu()
    pql = pql.reshape(-1).numpy()

    # train/test split for the linear probe
    N = NFRESH * T; rng = np.random.default_rng(0); perm = rng.permutation(N)
    tr, te = perm[:N // 2], perm[N // 2:]
    y = inside.astype(np.float64)
    depth_auc = {}
    for li in CAP_LAYERS:
        X = torch.cat(caps[li], 0).numpy()
        Xc = X - X[tr].mean(0)
        w = Xc[tr].T @ (y[tr] - y[tr].mean())
        w = w / (np.linalg.norm(w) + 1e-9)
        s = X[te] @ w
        depth_auc[li] = round(auc(s, inside[te]), 4)
        print(f'depth after block {li:2d}: quote-state probe AUC {depth_auc[li]}',
              flush=True)

    # shuffled-label null at the final depth
    Xf = torch.cat(caps[17], 0).numpy(); Xfc = Xf - Xf[tr].mean(0)
    ysh = y.copy(); rng.shuffle(ysh)
    wsh = Xfc[tr].T @ (ysh[tr] - ysh[tr].mean()); wsh = wsh / (np.linalg.norm(wsh) + 1e-9)
    null_auc = round(auc(Xf[te] @ wsh, ysh[te].astype(int)), 4)

    # behavioral: P(closing quote) inside vs outside
    pq_in = float(pql[inside == 1].mean()); pq_out = float(pql[inside == 0].mean())
    print(f'\nP(") inside {pq_in:.5f}  outside {pq_out:.5f}', flush=True)
    print(f'shuffled-label null AUC {null_auc}', flush=True)

    final_auc = depth_auc[17]
    p0 = n_in >= 100 and n_out >= 100
    pa = final_auc >= 0.8
    pc = pq_in > pq_out
    null_ok = abs(null_auc - 0.5) < 0.1
    print(f'\n(0) enough: {p0}', flush=True)
    print(f'(a) quote-state tracked (final AUC>=0.8): {pa} ({final_auc})', flush=True)
    print(f'(c) P(") higher inside: {pc}', flush=True)
    print(f'NULL shuffled ~0.5: {null_ok}', flush=True)

    out = {'n_inside': n_in, 'n_outside': n_out, 'depth_probe_auc': depth_auc,
           'shuffled_null_auc': null_auc, 'P_quote_inside': round(pq_in, 5),
           'P_quote_outside': round(pq_out, 5),
           'pred_0': bool(p0), 'pred_a_tracked': bool(pa), 'pred_c_behavioral': bool(pc),
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
