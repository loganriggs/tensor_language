"""LAYER-UNDERSTANDING DEPTH PROFILE (where does the un-understood ~22% live?).
794/795: the whole model is ~78% class+position, uniform across token-classes. Profile
it across DEPTH: for each layer L (0-17), keep ONLY class+position at BOTH its attn and
mlp (that layer's two components), measure the CE-recovery of THAT layer's benefit. Shows
which layers reduce to class+position and which carry the distributed remainder. Emits a
figure. 128 rows.

REGISTERED PREDICTIONS:
  (0) SANITY: per-layer benefit >= 0;
  (a) DEPTH PROFILE: the big early layers (0,1,5) are high class+position recovery
      (>=0.8); report where recovery is LOWEST (where the distributed computation
      concentrates in depth);
  (b) report per-layer benefit + class+position recovery + random baseline;
  NULL: random same-rank subspace recovers far less at each layer."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/rspd'); sys.path.insert(0, '/workspace/tensor_language')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

D = 1152; NL = 18
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'layer_understanding_profile_results.json'; FIG = PT + 'layer_understanding_profile.png'
NEVAL = 128; MINCOUNT = 5; RTOK = 64; RPOS = 32
SUBS = {}; MODE = {'op': None, 'L': -1}
BLUE, MUTEDC = '#3987e5', '#c9c7bf'


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def hook_factory(w, L):
    key = (w, L)
    def h(mo, i_, o_):
        if MODE['op'] is None or MODE['L'] != L: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else: U = MODE['rand'] if MODE['op'] == 'keeprand' else SUBS[key]; v2 = (v @ U) @ U.T
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
def capture_out(rows, n, w, L):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(w, L).register_forward_hook(h)
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
    MODE['op'] = None
    for L in range(NL):
        for w in ('attn', 'mlp'):
            O, toks, pos = capture_out(rows, NEVAL, w, L)
            Utok = mean_subspace(O, toks, RTOK); Upos = mean_subspace(O, pos, RPOS)
            SUBS[(w, L)] = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    hooks = [comp(w, L).register_forward_hook(hook_factory(w, L)) for L in range(NL) for w in ('attn', 'mlp')]
    g = torch.Generator(device=DEV).manual_seed(0); MODE['rand'] = torch.linalg.qr(torch.randn(D, RTOK+RPOS, generator=g, device=DEV))[0]
    MODE['op'] = None; MODE['L'] = -1; ce_full = ce_on(rows, NEVAL)
    res = {}
    for L in range(NL):
        MODE['L'] = L
        MODE['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL); ben = ce_abl - ce_full
        MODE['op'] = 'keep'; ce_keep = ce_on(rows, NEVAL)
        MODE['op'] = 'keeprand'; ce_keeprand = ce_on(rows, NEVAL); MODE['op'] = None
        rec = float((ce_abl-ce_keep)/max(ben, 1e-6)); recr = float((ce_abl-ce_keeprand)/max(ben, 1e-6))
        res[str(L)] = {'benefit': round(ben, 4), 'keep': round(rec, 4), 'keep_random': round(recr, 4)}
        print(f'L{L}: ben {ben:.2f} keep-class+pos {rec:.2f} (rand {recr:.2f})', flush=True)
    MODE['L'] = -1
    for h in hooks: h.remove()

    xs = np.arange(NL); ben = np.array([res[str(L)]['benefit'] for L in xs]); keep = np.clip(np.array([res[str(L)]['keep'] for L in xs]), 0, 1)
    fig, ax = plt.subplots(figsize=(12, 5)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.bar(xs, keep, color=BLUE, label='class+position'); ax.bar(xs, np.clip(1-keep, 0, 1), bottom=keep, color=MUTEDC, label='distributed remainder')
    for L in xs: ax.text(L, 1.02, f'{ben[L]:.1f}', ha='center', fontsize=7.5, color=MUTED)
    ax.set_ylim(0, 1.13); ax.set_xticks(xs); ax.set_xlabel('layer (attn+mlp together)'); ax.set_ylabel("share of layer's benefit that is class+position")
    ax.set_title('Where the un-understood computation lives across depth\n(number above bar = layer benefit in nats)', color=INK, fontsize=12.5, loc='left')
    ax.legend(fontsize=9, loc='lower right'); ax.grid(True, axis='y', color=GRID, lw=0.6)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(SECONDARY)
    fig.tight_layout(); fig.savefig(FIG, dpi=150, facecolor=SURFACE); print('wrote', FIG, flush=True)

    real = [L for L in range(NL) if res[str(L)]['benefit'] > 0.2]
    lowest = sorted(real, key=lambda L: res[str(L)]['keep'])[:3]
    out = {'results': res, 'lowest_understood_layers': [(L, res[str(L)]['keep'], res[str(L)]['benefit']) for L in lowest], 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nlowest class+position layers (where distributed lives): {out["lowest_understood_layers"]}', flush=True)
    print(f'wrote {OUT} ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
