"""EARLY-LAYER UNDERSTANDING SCOREBOARD (user: bottom-up, how much of the early
layers do we understand?). For EVERY early component (attn0-5 + mlp0-5), measure
benefit (CE if its output is ablated) and how much of that benefit is captured by
keeping ONLY the token-class + position subspace of its OUTPUT -- the "understood as
class+position" fraction. Nat-weighted total = the single "% of the early-layer
budget we account for as class+position". The remainder per component = what is left
to name (the bottom-up worklist). Emits a stacked-bar figure.

REGISTERED PREDICTIONS:
  (0) SANITY: keep-only class+position >= random same-rank at each component;
  (a) MAJORITY UNDERSTOOD: the nat-weighted keep-only-(class+position) fraction over
      all early components is >= 0.6, and the cleanest components (mlp0, attn1)
      exceed 0.85 -- most of the early-layer budget is class+position;
  (b) report per-component benefit + class / +position / remainder + the nat-weighted
      totals; identify the components with the largest UNNAMED remainder (worklist);
  NULL: random same-rank output subspace recovers far less at each component."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/rspd'); sys.path.insert(0, '/workspace/tensor_language')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'early_scoreboard_results.json'; FIG = PT + 'early_scoreboard.png'
NEVAL = 64; MINCOUNT = 5; RTOK = 64; RPOS = 32; LAYERS = list(range(6))
MODE = {'U': None, 'op': None, 'key': None}      # key = (which, L)
BLUE, GREEN, MUTEDC = '#3987e5', '#2e8b57', '#c9c7bf'


def comp(which, L):
    return m.transformer.h[L].mlp if which == 'mlp' else m.transformer.h[L].attn


def hook_factory(which, L):
    def h(mo, i_, o_):
        if MODE['key'] != (which, L) or MODE['op'] is None: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else:
            U = MODE['U']; v2 = (v @ U) @ U.T if MODE['op'] == 'keep' else v - (v @ U) @ U.T
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
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); forward_logits(idx)
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
    hooks = [comp(w, L).register_forward_hook(hook_factory(w, L)) for L in LAYERS for w in ('mlp', 'attn')]
    MODE['op'] = None; MODE['key'] = None; ce_full = ce_on(rows, NEVAL)
    g = torch.Generator(device=DEV).manual_seed(0); res = {}
    for L in LAYERS:
        for w in ('attn', 'mlp'):
            O, toks, pos = capture_out(rows, NEVAL, w, L)
            Utok = mean_subspace(O, toks, RTOK); Upos = mean_subspace(O, pos, RPOS)
            Uc = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
            MODE['key'] = (w, L); MODE['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL); ben = ce_abl - ce_full
            def keeprec(U):
                MODE['op'] = 'keep'; MODE['U'] = U; c = ce_on(rows, NEVAL); MODE['op'] = None; MODE['U'] = None
                return float((ce_abl - c)/max(ben, 1e-6))
            rt = keeprec(Utok); rc = keeprec(Uc)
            Ur = torch.linalg.qr(torch.randn(D, RTOK+RPOS, generator=g, device=DEV))[0]; rr = keeprec(Ur)
            MODE['op'] = None; MODE['key'] = None
            res[f'{w}{L}'] = {'benefit': round(ben, 4), 'keep_token': round(rt, 4), 'keep_combined': round(rc, 4), 'keep_random': round(rr, 4)}
            print(f'{w}{L}: ben {ben:.2f} | class {rt:.2f} +pos {rc:.2f} rand {rr:.2f}', flush=True)
    for h in hooks: h.remove()

    keys = [f'{w}{L}' for L in LAYERS for w in ('attn', 'mlp')]
    bens = np.array([res[k]['benefit'] for k in keys]); combs = np.clip(np.array([res[k]['keep_combined'] for k in keys]), 0, 1)
    nat_weighted = float((bens*combs).sum()/max(bens.sum(), 1e-9))
    print(f'\nNAT-WEIGHTED understood (class+position): {nat_weighted:.3f} of {bens.sum():.2f} early-layer nats', flush=True)
    worklist = sorted([(k, res[k]['benefit'], round(1-np.clip(res[k]['keep_combined'],0,1),2)) for k in keys if res[k]['benefit']>0.2], key=lambda z:-z[1]*z[2])
    print('WORKLIST (benefit x unnamed-remainder, top):', worklist[:5], flush=True)

    # figure
    xs = np.arange(len(keys))
    tok = np.clip(np.array([res[k]['keep_token'] for k in keys]), 0, 1)
    posadd = np.clip(combs - tok, 0, 1); rem = np.clip(1-combs, 0, 1)
    fig, ax = plt.subplots(figsize=(13, 5.4)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.bar(xs, tok, color=BLUE, label='token-class'); ax.bar(xs, posadd, bottom=tok, color=GREEN, label='+ position')
    ax.bar(xs, rem, bottom=combs, color=MUTEDC, label='unnamed (bottom-up worklist)')
    for i, k in enumerate(keys): ax.text(i, 1.02, f'{res[k]["benefit"]:.1f}', ha='center', fontsize=7, color=MUTED)
    ax.set_ylim(0, 1.13); ax.set_xticks(xs); ax.set_xticklabels(keys, rotation=45, fontsize=8)
    ax.set_ylabel('share of component understood as class+position')
    ax.set_title(f'Early-layer understanding scoreboard (L0-5) -- nat-weighted class+position = {nat_weighted:.0%}\n'
                 'number above bar = benefit (nats); grey = unnamed computation left to name', color=INK, fontsize=12.5, loc='left')
    ax.legend(fontsize=9, loc='lower center'); ax.grid(True, axis='y', color=GRID, lw=0.6)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(SECONDARY)
    fig.tight_layout(); fig.savefig(FIG, dpi=150, facecolor=SURFACE); print('wrote', FIG, flush=True)

    pa = nat_weighted >= 0.6 and res['mlp0']['keep_combined'] >= 0.85 and res['attn1']['keep_combined'] >= 0.85
    out = {'results': res, 'nat_weighted_understood': round(nat_weighted, 4), 'total_early_nats': round(float(bens.sum()), 3),
           'worklist_top': worklist[:6], 'pred_a_majority_understood': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) majority of early layers understood as class+position (nat-weighted>=0.6, mlp0/attn1>=0.85): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
