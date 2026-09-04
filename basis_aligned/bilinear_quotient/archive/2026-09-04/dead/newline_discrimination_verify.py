"""NEWLINE DISCRIMINATION VERIFY (apply the 726-727 causal lens to a
FLAGSHIP: the newline circuit, FINDINGS item 7 / 635-644). Core claim:
FRONT ATTENTION discriminates real line-ends from mid-paragraph periods
(the `.` bigram fires at every period; the front routes newline vs
continuation). Verify CAUSALLY: at PERIOD positions, does the model's
P(newline) track whether it is actually a line-end, and does ABLATING the
front attention (block0+block1 c_proj) COLLAPSE that discrimination?

Metric: AUC of P(newline | period) vs is-actually-line-end, full model vs
front-attention-ablated vs a random-attention-ablated control.

REGISTERED PREDICTIONS:
  (0) SANITY: full-model AUC > 0.6 (the model does discriminate line-ends);
  (a) FRONT-ATTN DRIVES IT: ablating front attention (blocks 0-1 c_proj)
      DROPS the discrimination AUC substantially (toward 0.5) -- more than a
      matched random-subspace ablation -- confirming front attention
      causally carries the newline discrimination;
  (b) report AUC full / front-ablated / random-ablated;
  NULL: ablating a random same-norm subspace of the same layers barely
      changes the AUC."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'newline_discrimination_verify_results.json'
NEVAL = 128
MODE = {'k': None}   # None | 'front' | 'rand'
RANDDIRS = {}


def hook_factory(name):
    def hook(mo, i_, o_):
        if MODE['k'] is None: return o_
        if MODE['k'] == 'front':
            return torch.zeros_like(o_)       # fully ablate front attn output
        else:  # rand: project out a random 128-dim subspace (matched-ish disruption)
            of = o_.float(); P = RANDDIRS[name]
            return (of - (of @ P) @ P.T).to(o_.dtype)
    return hook


@torch.no_grad()
def collect(rows, n, nl_ids):
    """P(newline) and is_lineend at every period position."""
    pnl = []; lineend = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)
        p = F.softmax(lg.float(), -1)
        pn = p[..., nl_ids].sum(-1).reshape(-1).cpu().numpy()   # P(newline)
        cur = idx.reshape(-1).cpu().numpy(); nxt = tgt.reshape(-1).cpu().numpy()
        for k in range(len(cur)):
            s = cl.d1(int(cur[k])).strip()
            if s.endswith('.') or s.endswith('!') or s.endswith('?'):
                pnl.append(pn[k]); lineend.append(1 if nxt[k] in nl_ids else 0)
    return np.array(pnl), np.array(lineend)


def auc(score, label):
    label = np.asarray(label)
    if label.sum() == 0 or label.sum() == len(label): return float('nan')
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score)+1)
    n1 = label.sum(); n0 = len(label) - n1
    return float((ranks[label == 1].sum() - n1*(n1+1)/2) / (n1*n0))


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    ev = cl.fineweb_rows(NEVAL)
    # newline token ids
    V = m.lm_head.weight.shape[0]
    nl_ids = [t for t in range(V) if '\n' in cl.d1(t)]
    nl_ids = torch.tensor(nl_ids, device=DEV)
    print(f'{len(nl_ids)} newline-containing token ids', flush=True)

    mods = {'b0': m.transformer.h[0].attn.c_proj, 'b1': m.transformer.h[1].attn.c_proj}
    g = torch.Generator().manual_seed(0)
    for name in mods:
        Q, _ = torch.linalg.qr(torch.randn(D, 128, generator=g)); RANDDIRS[name] = Q.to(DEV)
    hooks = {name: mods[name].register_forward_hook(hook_factory(name)) for name in mods}

    MODE['k'] = None
    pnl, le = collect(ev, NEVAL, nl_ids)
    auc_full = auc(pnl, le)
    MODE['k'] = 'front'; pnf, _ = collect(ev, NEVAL, nl_ids); auc_front = auc(pnf, le)
    MODE['k'] = 'rand'; pnr, _ = collect(ev, NEVAL, nl_ids); auc_rand = auc(pnr, le)
    MODE['k'] = None
    for h in hooks.values(): h.remove()

    print(f'period positions: {len(le)}, line-ends: {int(le.sum())}', flush=True)
    print(f'discrimination AUC  full {auc_full:.3f}  front-ablated {auc_front:.3f}  '
          f'rand-ablated {auc_rand:.3f}', flush=True)
    drop_front = auc_full - auc_front; drop_rand = auc_full - auc_rand
    p0 = auc_full > 0.6
    pa = drop_front > 1.5 * max(drop_rand, 1e-6) and drop_front > 0.05
    print(f'\n(0) model discriminates (AUC>0.6): {p0}', flush=True)
    print(f'(a) front attn drives it (drop {drop_front:.3f} > 1.5x rand drop {drop_rand:.3f}): {pa}',
          flush=True)
    out = {'auc_full': round(auc_full,4), 'auc_front_ablated': round(auc_front,4),
           'auc_rand_ablated': round(auc_rand,4), 'drop_front': round(drop_front,4),
           'drop_rand': round(drop_rand,4), 'n_periods': int(len(le)), 'n_lineend': int(le.sum()),
           'pred_0': bool(p0), 'pred_a_front_drives': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
