"""WHY does a random subspace recover 0.84 of a single bilin18 component (§820)? MEAN or
REDUNDANCY? §820 found mean-preserving keep with a RANDOM-orthonormal subspace recovers
0.84-0.95 of a high-benefit single component. Two possible drivers: (M) the component's
constant MEAN μ alone approximates its output; (R) cross-component REDUNDANCY — other intact
components carry the info, so a degraded single component barely hurts. For bilin18 §808 said
μ is negligible (centered≈mean-preserve 0.92), suggesting REDUNDANCY, but §820 hedged
'mean+redundancy'. Settle it: decompose the single-component keep into mean-only vs
centered-random vs mean-preserving-random.

For bilin18 attn0, mlp0, attn1: ablate benefit, then keep-only recovery for
  - mean_only: substitute the constant mean μ (no subspace at all);
  - centered_random: project onto a random-orth subspace, NO mean (isolates redundancy);
  - meanpreserve_random: μ + random projection (reproduces §820's ~0.84);
  - meanpreserve_realcp: the real class+position (reference).

REGISTERED PREDICTIONS:
  (a) REDUNDANCY driver (expected for bilin18, μ negligible): centered_random is already HIGH
      (>=0.7) — a random projection of a single component barely hurts because others
      compensate — and mean_only is modest; so §820's 0.84 is redundancy, not mean;
  (b) MEAN driver: mean_only alone is HIGH (>=0.7) and centered_random is LOW; then μ drives it;
  (c) report all four; the real class+position should still exceed centered_random (its
      specific increment)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cp_null_mechanism_results.json'
NEVAL = 200; MINCOUNT = 5; RTOK = 64; RPOS = 32
COMPS = [('attn', 0), ('mlp', 0), ('attn', 1)]
SUB = {'op': None, 'U': None, 'mean': None, 'name': None}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    name = (w, L)
    def hook(mo, i_, o_):
        if SUB['op'] is None or SUB['name'] != name: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float(); mu = SUB['mean']
        if SUB['op'] == 'ablate': v2 = torch.zeros_like(v)
        elif SUB['op'] == 'mean_only': v2 = mu.expand_as(v)
        elif SUB['op'] == 'centered_random': U = SUB['U']; v2 = (v @ U) @ U.T
        else: U = SUB['U']; v2 = mu + ((v - mu) @ U) @ U.T   # meanpreserve
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def ce_on(rows):
    s = 0.0; nn = 0
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1)))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


@torch.no_grad()
def capture(rows, w, L):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(w, L).register_forward_hook(h)
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r, g):
    rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); out = {}
    for w, L in COMPS:
        name = (w, L); h = comp(w, L).register_forward_hook(mk_hook(w, L)); SUB['name'] = name
        O, toks, pos = capture(rows, w, L)
        gmean = O.mean(0, keepdim=True); SUB['mean'] = gmean
        mean_ratio = float(gmean.norm() / O.norm(dim=1).mean())
        Ut = mean_subspace(O, toks, RTOK, gmean); Up = mean_subspace(O, pos, RPOS, gmean)
        Ucp = torch.linalg.svd(torch.cat([Ut, Up], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
        g = torch.Generator(device=DEV).manual_seed(0); Ur = torch.linalg.qr(torch.randn(D, RTOK+RPOS, generator=g, device=DEV))[0]
        SUB['op'] = None; ce_full = ce_on(rows)
        SUB['op'] = 'ablate'; ce_abl = ce_on(rows); ben = ce_abl - ce_full
        def rec(op, U=None): SUB['op'] = op; SUB['U'] = U; c = ce_on(rows); SUB['op'] = None; return round(float((ce_abl-c)/max(ben, 1e-6)), 4)
        r = {'benefit': round(ben, 3), 'mean_ratio': round(mean_ratio, 3),
             'mean_only': rec('mean_only'), 'centered_random': rec('centered_random', Ur),
             'meanpreserve_random': rec('meanpreserve', Ur), 'meanpreserve_realcp': rec('meanpreserve', Ucp)}
        h.remove(); out[f'{w}{L}'] = r
        print(f'{w}{L}: ben {r["benefit"]} mean/out {r["mean_ratio"]} | mean-only {r["mean_only"]} | centered-rand {r["centered_random"]} | meanpres-rand {r["meanpreserve_random"]} | real-cp {r["meanpreserve_realcp"]}', flush=True)
    big = [k for k in out if out[k]['benefit'] > 0.5]
    cr = np.mean([out[k]['centered_random'] for k in big]); mo = np.mean([out[k]['mean_only'] for k in big])
    out['driver'] = 'redundancy' if cr >= 0.7 and cr > mo else ('mean' if mo >= 0.7 and mo > cr else 'both/mixed')
    out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nDRIVER of the high single-component random-null recovery: {out["driver"]} (mean centered-rand {cr:.2f}, mean-only {mo:.2f})', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
