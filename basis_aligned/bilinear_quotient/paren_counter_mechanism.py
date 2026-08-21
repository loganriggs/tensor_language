"""PAREN COUNTER MECHANISM -- HOW is the parenthesis-depth register built?
669 found it is linearly decodable from block 2 (AUC 0.92). Counting
open-minus-close requires integrating over previous tokens, which needs
ATTENTION. Test: does ablating the FRONT attention destroy the paren-
depth decodability (the counter is attention-computed), while ablating
the front MLP does not?

Probe paren-open state from the residual after block 3 under: baseline,
front [0-2] attention mean-ablated, front [0-2] MLP mean-ablated. If the
counter is attention-built, front-attention ablation collapses the probe
AUC toward chance; MLP ablation leaves it.

REGISTERED PREDICTIONS:
  (0) SANITY: baseline paren probe AUC after block 3 is high (>=0.85,
      per 669);
  (a) COUNTER IS ATTENTION-BUILT: front-attention ablation drops the
      probe AUC substantially (toward 0.5 + a bit) -- counting needs
      attention to reach previous parens;
  (b) MLP NOT REQUIRED FOR THE COUNT: front-MLP ablation leaves the probe
      AUC much higher than front-attention ablation does;
  (c) report probe AUC under the three conditions;
  NULL: a shuffled-label probe is ~0.5 in every condition."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'paren_counter_mechanism_results.json'
NFRESH = 48
CAP_AFTER = 3
FRONT = [0, 1, 2]


def auc(scores, labels):
    order = np.argsort(scores); ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos = labels == 1; npos = pos.sum(); nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    return float((ranks[pos].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def meanfill(mo, i_, o_):
    return o_.mean(dim=(0, 1), keepdim=True).expand_as(o_)


@torch.no_grad()
def capture(fresh, blocks, kind):
    handles = []
    if kind is not None:
        for li in blocks:
            sub = (m.transformer.h[li].attn.c_proj if kind == 'attn'
                   else m.transformer.h[li].mlp)
            handles.append(sub.register_forward_hook(meanfill))
    cap = []
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            if li == CAP_AFTER:
                cap.append(x.detach().float().reshape(-1, D).cpu())
                break
    for h in handles:
        h.remove()
    return torch.cat(cap, 0).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    toks = fresh[:, :256].numpy()
    inside = np.zeros((NFRESH, T), dtype=np.int64)
    for r in range(NFRESH):
        d = 0
        for j in range(T):
            s = cl.d1(int(toks[r, j])); d += s.count('(') - s.count(')')
            inside[r, j] = 1 if d > 0 else 0
    inside = inside.reshape(-1)
    N = NFRESH * T; rng = np.random.default_rng(0); perm = rng.permutation(N)
    tr, te = perm[:N // 2], perm[N // 2:]
    y = inside.astype(np.float64)

    def probe_auc(X):
        Xc = X - X[tr].mean(0)
        w = Xc[tr].T @ (y[tr] - y[tr].mean()); w = w / (np.linalg.norm(w) + 1e-9)
        return round(auc(X[te] @ w, inside[te]), 4)

    conds = {'baseline': (None, None), 'front_attn_abl': (FRONT, 'attn'),
             'front_mlp_abl': (FRONT, 'mlp')}
    aucs = {}
    for name, (b, k) in conds.items():
        aucs[name] = probe_auc(capture(fresh, b, k))
        print(f'{name:16s} paren probe AUC {aucs[name]}', flush=True)

    # shuffled null on baseline
    Xb = capture(fresh, None, None); ysh = y.copy(); rng.shuffle(ysh)
    wsh = (Xb[tr] - Xb[tr].mean(0)).T @ (ysh[tr] - ysh[tr].mean())
    wsh = wsh / (np.linalg.norm(wsh) + 1e-9)
    null_auc = round(auc(Xb[te] @ wsh, ysh[te].astype(int)), 4)
    print(f'shuffled null AUC {null_auc}', flush=True)

    drop_attn = aucs['baseline'] - aucs['front_attn_abl']
    drop_mlp = aucs['baseline'] - aucs['front_mlp_abl']
    p0 = aucs['baseline'] >= 0.85
    pa = aucs['front_attn_abl'] < aucs['baseline'] - 0.15
    pb = drop_attn > drop_mlp
    null_ok = abs(null_auc - 0.5) < 0.1
    print(f'\n(0) baseline high: {p0}', flush=True)
    print(f'(a) attention-built (attn ablation drops AUC >0.15): {pa} '
          f'(drop {drop_attn:.3f})', flush=True)
    print(f'(b) attn drop > mlp drop: {pb} (attn {drop_attn:.3f} vs mlp {drop_mlp:.3f})',
          flush=True)
    print(f'NULL shuffled ~0.5: {null_ok}', flush=True)

    out = {'probe_auc': aucs, 'shuffled_null': null_auc,
           'drop_front_attn': round(drop_attn, 4), 'drop_front_mlp': round(drop_mlp, 4),
           'pred_0': bool(p0), 'pred_a_attention_built': bool(pa),
           'pred_b_attn_gt_mlp': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
