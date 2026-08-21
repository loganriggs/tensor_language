"""FRONT = CLASS+POSITION COMPUTER (end-to-end test of the robust summary). The
scoreboard (790) showed each early component's OUTPUT is ~93% class+position, and the
robust framing (792) is variables, not per-head names. Test the strong claim
SIMULTANEOUSLY: project EVERY early component's output (attn0-5, mlp0-5) onto ONLY its
token-class + position subspace AT ONCE, and measure how much of the full model's CE
is preserved. If keeping only class+position at every early component still runs the
model well, the front genuinely reduces to a class+position computer (accounting for
interactions, not just per-component).

REGISTERED PREDICTIONS:
  (0) SANITY: full model CE recovered when no projection;
  (a) FRONT REDUCES TO CLASS+POSITION: projecting all 12 early components onto
      class+position simultaneously keeps CE-recovery >= 0.7 of the (front-ablated ->
      full) range, and >> projecting onto random same-rank subspaces; so the first 6
      layers' contribution is ~class+position end-to-end;
  (b) report simultaneous keep-(class+position) vs keep-random vs ablate-all-front;
  NULL: random same-rank simultaneous projection recovers far less."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'front_variable_model_results.json'
NEVAL = 64; MINCOUNT = 5; RTOK = 64; RPOS = 32; LAYERS = list(range(6))
SUBS = {}          # (which,L) -> subspace U (D,r) or 'random' handled via RAND
MODE = {'op': None, 'rand': None}


def comp(which, L):
    return m.transformer.h[L].mlp if which == 'mlp' else m.transformer.h[L].attn


def hook_factory(which, L):
    key = (which, L)
    def h(mo, i_, o_):
        if MODE['op'] is None: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else:
            U = MODE['rand'] if MODE['op'] == 'keeprand' else SUBS[key]
            v2 = (v @ U) @ U.T
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def ce_on(rows, n):
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1)))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


@torch.no_grad()
def capture_out(rows, n, which, L):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(which, L).register_forward_hook(h)
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    # first, capture + build class+position subspaces for every early component (no projection active)
    MODE['op'] = None
    for L in LAYERS:
        for w in ('attn', 'mlp'):
            O, toks, pos = capture_out(rows, NEVAL, w, L)
            Utok = mean_subspace(O, toks, RTOK); Upos = mean_subspace(O, pos, RPOS)
            SUBS[(w, L)] = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    hooks = [comp(w, L).register_forward_hook(hook_factory(w, L)) for L in LAYERS for w in ('attn', 'mlp')]

    MODE['op'] = None; ce_full = ce_on(rows, NEVAL)
    MODE['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL); ben = ce_abl - ce_full
    MODE['op'] = 'keep'; ce_keep = ce_on(rows, NEVAL)
    g = torch.Generator(device=DEV).manual_seed(0); MODE['rand'] = torch.linalg.qr(torch.randn(D, RTOK+RPOS, generator=g, device=DEV))[0]
    MODE['op'] = 'keeprand'; ce_keeprand = ce_on(rows, NEVAL)
    MODE['op'] = None
    rec = float((ce_abl - ce_keep)/max(ben, 1e-6)); rec_r = float((ce_abl - ce_keeprand)/max(ben, 1e-6))
    for h in hooks: h.remove()
    print(f'CE_full {ce_full:.3f} | ablate-all-front {ce_abl:.3f} (benefit {ben:.3f})', flush=True)
    print(f'keep-only class+position (all 12 early components at once): CE-recovery {rec:.3f} | random-subspace {rec_r:.3f}', flush=True)

    p0 = ben > 0
    pa = rec >= 0.7 and rec > 2*rec_r
    out = {'ce_full': round(ce_full, 4), 'front_benefit': round(ben, 4), 'keep_classpos_recovery': round(rec, 4),
           'keep_random_recovery': round(rec_r, 4), 'pred_0': bool(p0), 'pred_a_front_is_class_position': bool(pa),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) the front (L0-5) reduces to a class+position computer end-to-end (keep>=0.7 & >>random): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
