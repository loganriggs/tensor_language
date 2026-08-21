"""PAREN DEPTH STATE -- does the quote-register pattern (667-668) hold for
a SECOND stateful register: parenthesis depth? Prediction from the
taxonomy: the state is DECODABLE (linear probe) but its behavioral effect
has NO removable linear carrier (read-correlate, like quote-parity).

paren-open state = whether an unclosed '(' is open at this position
(count of '(' minus ')' > 0). Probe it from residuals at several depths;
test the behavioral effect on P(')') inside-vs-outside; and test whether
removing the rank-1 probe direction collapses the behavioral gap.

REGISTERED PREDICTIONS:
  (0) SANITY: enough inside-paren positions (>=60);
  (a) DECODABLE: paren-open state is linearly decodable (peak probe AUC
      across depths >= 0.75), like quote-parity;
  (b) BEHAVIORAL: P(')') is higher inside an open paren than outside;
  (c) READ-CORRELATE (the taxonomy test): removing the rank-1 paren
      direction (at its peak depth) does NOT collapse the behavioral gap
      (< 25% lost) -- decodable but causally conditional, like quote
      (668);
  NULL: shuffled-label probe AUC ~ 0.5; random rank-1 removal barely
      changes the gap."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'paren_depth_state_results.json'
NFRESH = 48
CAP_LAYERS = [2, 6, 10, 14]


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

    # find '(' and ')' token ids (tokens whose stripped form is exactly that)
    def tid(ch):
        for t in range(m.lm_head.weight.shape[0]):
            if cl.d1(t) == ch:
                return t
        return -1
    OPN, CLS = tid('('), tid(')')
    if OPN < 0 or CLS < 0:
        # fallback: any token containing the char
        OPN = next(t for t in range(1000) if '(' in cl.d1(t))
        CLS = next(t for t in range(1000) if ')' in cl.d1(t))
    print(f"'(' id {OPN}  ')' id {CLS}", flush=True)

    inside = np.zeros((NFRESH, T), dtype=np.int64)
    for r in range(NFRESH):
        d = 0
        for j in range(T):
            s = cl.d1(int(toks[r, j]))
            d += s.count('(') - s.count(')')
            inside[r, j] = 1 if d > 0 else 0
    inside = inside.reshape(-1)
    n_in = int(inside.sum())
    print(f'{n_in} inside-paren, {len(inside)-n_in} outside', flush=True)

    caps = {li: [] for li in CAP_LAYERS}; resid6 = []
    pcl = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            if li in CAP_LAYERS:
                caps[li].append(x.detach().float().reshape(-1, D).cpu())
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        pcl[i:i + B] = F.softmax(lg, dim=-1)[..., CLS].cpu()
    pcl = pcl.reshape(-1).numpy()

    N = NFRESH * T; rng = np.random.default_rng(0); perm = rng.permutation(N)
    tr, te = perm[:N // 2], perm[N // 2:]
    y = inside.astype(np.float64)
    depth_auc = {}; wdirs = {}
    for li in CAP_LAYERS:
        X = torch.cat(caps[li], 0).numpy(); Xc = X - X[tr].mean(0)
        w = Xc[tr].T @ (y[tr] - y[tr].mean()); w = w / (np.linalg.norm(w) + 1e-9)
        wdirs[li] = (X, w)
        depth_auc[li] = round(auc(X[te] @ w, inside[te]), 4)
        print(f'depth block {li:2d}: paren probe AUC {depth_auc[li]}', flush=True)
    peak = max(depth_auc, key=lambda k: depth_auc[k])

    # shuffled null at peak
    Xp, wp = wdirs[peak]; ysh = y.copy(); rng.shuffle(ysh)
    wsh = (Xp[tr] - Xp[tr].mean(0)).T @ (ysh[tr] - ysh[tr].mean())
    wsh = wsh / (np.linalg.norm(wsh) + 1e-9)
    null_auc = round(auc(Xp[te] @ wsh, ysh[te].astype(int)), 4)

    pq_in = float(pcl[inside == 1].mean()); pq_out = float(pcl[inside == 0].mean())

    # causal: remove peak-depth paren direction, measure behavioral gap
    @torch.no_grad()
    def gap_with_removal(remove_dir):
        pc = torch.zeros(NFRESH, T)
        for i in range(0, NFRESH, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous(); B = bb.shape[0]
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for li, blk in enumerate(m.transformer.h):
                x, v1 = blk(x, v1, x0)
                if li == peak and remove_dir is not None:
                    x = x - (x @ remove_dir)[..., None] * remove_dir
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
            pc[i:i + B] = F.softmax(lg, dim=-1)[..., CLS].cpu()
        pc = pc.reshape(-1).numpy()
        return float(pc[inside == 1].mean() - pc[inside == 0].mean())

    g_base = pq_in - pq_out
    wpk = torch.tensor(wdirs[peak][1], dtype=torch.float32, device=DEV)
    g_rm = gap_with_removal(wpk)
    rr = rng.standard_normal(D); rr /= np.linalg.norm(rr)
    g_rand = gap_with_removal(torch.tensor(rr, dtype=torch.float32, device=DEV))
    print(f'\nP(")") inside {pq_in:.5f} outside {pq_out:.5f} (gap {g_base:+.5f})',
          flush=True)
    print(f'remove peak paren dir: gap {g_rm:+.5f} (lost {100*(1-g_rm/g_base):.0f}%)',
          flush=True)
    print(f'remove random: gap {g_rand:+.5f}', flush=True)

    p0 = n_in >= 60
    pa = depth_auc[peak] >= 0.75
    pb = pq_in > pq_out
    lost = 1 - g_rm / g_base if g_base != 0 else 0
    pc_readcorr = lost < 0.25
    null_ok = abs(null_auc - 0.5) < 0.1
    print(f'\n(0) enough: {p0}; (a) decodable (peak {depth_auc[peak]}): {pa}', flush=True)
    print(f'(b) P()) higher inside: {pb}; (c) read-correlate (removal <25%): '
          f'{pc_readcorr} (lost {100*lost:.0f}%)', flush=True)
    print(f'NULL shuffled {null_auc}: {null_ok}', flush=True)

    out = {'n_inside': n_in, 'depth_probe_auc': depth_auc, 'peak_depth': peak,
           'shuffled_null_auc': null_auc, 'P_close_inside': round(pq_in, 5),
           'P_close_outside': round(pq_out, 5), 'gap_baseline': round(g_base, 5),
           'gap_remove_dir': round(g_rm, 5), 'gap_remove_random': round(g_rand, 5),
           'removal_lost_frac': round(float(lost), 4),
           'pred_0': bool(p0), 'pred_a_decodable': bool(pa), 'pred_b_behavioral': bool(pb),
           'pred_c_read_correlate': bool(pc_readcorr), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
