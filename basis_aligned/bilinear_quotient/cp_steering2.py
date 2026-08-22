"""CAUSAL SUFFICIENCY via steering, DONE RIGHT (cp_steering v1 was confounded: injecting the
full mean μ_B is dominated by the global constant (§821/822 mean_ratio up to 0.97), so the
B-specific class signal was tiny). v2 injects the amplified class-DEVIATION
  Δ_B,c = proj_cp( μ_B,c − μ_global,c )   (the B-specific part only)
ADDED to every FRONT component (layers 0-5, where class+position is computed), scaled by α,
and sweeps α. If class+position is causally sufficient to set the predicted class, a large
enough α should move the next-token distribution toward B's typical continuation p_B (KL
drops), more than a random-direction steer of the same per-α norm (NULL). If even large-α
all-front steering fails, class+position is necessary but not steerable at these sites —
evidence the model redundantly recomputes it from the current-token embedding that never
leaves the stream (FINDINGS 14).

REGISTERED PREDICTIONS:
  (0) SANITY: α=0 reproduces the normal KL exactly; steering hooks fire (α>0 differs);
  (a) SUFFICIENT: at some α, KL(cp-steer||p_B) drops well below KL(normal||p_B) and below the
      matched-norm random-direction steer -> class+position is causally sufficient to steer
      the prediction toward B;
  (b) NOT STEERABLE: if KL never drops toward p_B (even as α grows, or only degrades like
      random), class+position at these write-sites is not sufficient — consistent with
      redundant recomputation from the persistent embedding;
  NULL: random-direction steer of matched per-α norm."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cp_steering2_results.json'
NEVAL = 200; MINCOUNT = 5; RTOK = 64; RPOS = 32
SRC_TOKENS = [262, 257, 290]        # " the", " a", " and"
FRONT = list(range(0, 6))
ALPHAS = [0.0, 2.0, 8.0, 32.0]
ST = {'on': False, 'mode': 'cp', 'alpha': 0.0, 'delta': {}, 'randdir': {}}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    key = (w, L)
    def hook(mo, i_, o_):
        if not ST['on'] or ST['alpha'] == 0.0: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        d = ST['delta'][key] if ST['mode'] == 'cp' else ST['randdir'][key]   # (1,D), matched norm
        v2 = v + ST['alpha'] * d
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


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


@torch.no_grad()
def avg_pred(rows, tok_id=None):
    ps = []
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        lg = forward_logits(idx).float(); p = F.softmax(lg, -1).reshape(-1, lg.shape[-1])
        if tok_id is None: ps.append(p.mean(0).cpu())
        else:
            mk = (idx.reshape(-1) == tok_id)
            if mk.any(): ps.append(p[mk].cpu())
    return (torch.cat(ps, 0) if tok_id is not None else torch.stack(ps, 0)).mean(0)


def kl(p, q):
    p = p + 1e-9; q = q + 1e-9; p = p/p.sum(); q = q/q.sum()
    return float((p * (p/q).log()).sum())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    # per front-component: cp subspace, global mean, and per-source class-deviation delta
    subs = {}; gmeans = {}; Odata = {}
    g = torch.Generator(device=DEV).manual_seed(0)
    for L in FRONT:
        for w in ('attn', 'mlp'):
            O, toks, pos = capture(rows, w, L); gm = O.mean(0, keepdim=True)
            Ut = mean_subspace(O, toks, RTOK, gm); Up = mean_subspace(O, pos, RPOS, gm)
            U = torch.linalg.svd(torch.cat([Ut, Up], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
            subs[(w, L)] = U; gmeans[(w, L)] = gm; Odata[(w, L)] = (O, toks)
    hooks = [comp(w, L).register_forward_hook(mk_hook(w, L)) for L in FRONT for w in ('attn', 'mlp')]
    ST['on'] = False; p_normal = avg_pred(rows)
    out = {'per_source': {}}
    for b in SRC_TOKENS:
        p_B = avg_pred(rows, b)
        # build per-component B-class deviation (projected on cp) + matched-norm random dir
        for (w, L), (O, toks) in Odata.items():
            mk = toks == b
            if mk.sum() < MINCOUNT: ST['delta'][(w, L)] = torch.zeros(1, D, device=DEV); ST['randdir'][(w, L)] = torch.zeros(1, D, device=DEV); continue
            mub = O[mk].mean(0, keepdim=True).to(DEV) - gmeans[(w, L)]     # B-specific deviation
            U = subs[(w, L)]; dcp = (mub @ U) @ U.T                          # projected on cp
            ST['delta'][(w, L)] = dcp
            rd = torch.randn(1, D, generator=g, device=DEV); rd = rd / rd.norm() * dcp.norm()
            ST['randdir'][(w, L)] = rd
        row = {'kl_normal': round(kl(p_normal, p_B), 4), 'cp': {}, 'rand': {}}
        for a in ALPHAS:
            if a == 0.0: continue
            ST['on'] = True; ST['alpha'] = a
            ST['mode'] = 'cp'; row['cp'][str(a)] = round(kl(avg_pred(rows), p_B), 4)
            ST['mode'] = 'rand'; row['rand'][str(a)] = round(kl(avg_pred(rows), p_B), 4)
            ST['on'] = False
        out['per_source'][str(b)] = row
        best_cp = min(row['cp'].values()); print(f"src {b}: normal {row['kl_normal']} | cp-steer {row['cp']} | rand {row['rand']}", flush=True)
    for h in hooks: h.remove()
    # sufficiency: does best cp-steer drop below normal AND below rand at same alpha, averaged?
    drops = []
    for b in SRC_TOKENS:
        r = out['per_source'][str(b)]; best_a = min(r['cp'], key=lambda k: r['cp'][k])
        drops.append((r['kl_normal'] - r['cp'][best_a], r['rand'][best_a] - r['cp'][best_a]))
    md_norm = float(np.mean([d[0] for d in drops])); md_rand = float(np.mean([d[1] for d in drops]))
    out['mean_drop_vs_normal'] = round(md_norm, 4); out['mean_cp_below_rand'] = round(md_rand, 4)
    out['pred_a_sufficient'] = bool(md_norm > 0.2 and md_rand > 0.1); out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nmean KL drop vs normal (best α) {md_norm:+.3f} | cp below rand {md_rand:+.3f}", flush=True)
    print(f"(a) class+position steering causally SUFFICIENT: {out['pred_a_sufficient']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']:.0f}s)")


if __name__ == '__main__':
    main()
