"""COMBINED INTERPRETABLE fraction ACROSS the early MLPs (generality of 777). For
MLP L0-L5, measure how much of each layer's CE benefit is captured by keeping ONLY
the token-class(64) + position(32) union subspace -- the "interpretable fraction" --
and where the irreducible (distributed) remainder peaks. Also the token-vs-position
split per layer. Emits a stacked-bar figure.

REGISTERED PREDICTIONS:
  (0) SANITY: keep-only-combined >= random same-dim at every layer;
  (a) MOSTLY INTERPRETABLE FRONT: keep-only-(token+position) recovers >= 0.6 of the
      benefit at every early MLP with real benefit (L0,L1,L4) -- the front MLP is
      largely token-class + position throughout;
  (b) PROFILE: L0 is token-class-dominated (position small); position's share rises
      into L1-L3; report the irreducible fraction per layer;
  NULL: random same-dim subspace recovers far less at every layer."""
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
OUT = PT + 'combined_depth_results.json'; FIG = PT + 'combined_depth.png'
NEVAL = 56; MINCOUNT = 5; RTOK = 64; RPOS = 32; LAYERS = [0, 1, 2, 3, 4, 5]
MODE = {'U': None, 'L': -1, 'op': None}
BLUE, GREEN, MUTEDC = '#3987e5', '#2e8b57', '#c9c7bf'


def mlp_hook_factory(L):
    def h(mo, i_, o_):
        if MODE['L'] != L or MODE['op'] is None: return o_
        sh = o_.shape; v = o_.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else:
            U = MODE['U']; v2 = (v @ U) @ U.T if MODE['op'] == 'keep' else v - (v @ U) @ U.T
        return v2.reshape(sh).to(o_.dtype)
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
def capture(rows, n, L):
    cap = []; toks = []; pos = []
    h = m.transformer.h[L].mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    h.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    hooks = [m.transformer.h[L].mlp.register_forward_hook(mlp_hook_factory(L)) for L in LAYERS]
    MODE['op'] = None; ce_full = ce_on(rows, NEVAL)
    g = torch.Generator(device=DEV).manual_seed(0)
    res = {}
    for L in LAYERS:
        O, toks, pos = capture(rows, NEVAL, L)
        Utok = mean_subspace(O, toks, RTOK); Upos = mean_subspace(O, pos, RPOS)
        Uc = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0]
        kdim = min(RTOK+RPOS, Uc.shape[1]); Ucomb = Uc[:, :kdim].contiguous()
        MODE['L'] = L
        MODE['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL); ben = ce_abl - ce_full
        def keeprec(U):
            MODE['op'] = 'keep'; MODE['U'] = U; c = ce_on(rows, NEVAL); MODE['op'] = None; MODE['U'] = None
            return float((ce_abl - c)/max(ben, 1e-6))
        rc = keeprec(Ucomb); rt = keeprec(Utok); rp = keeprec(Upos)
        Ur = torch.linalg.qr(torch.randn(D, kdim, generator=g, device=DEV))[0]; rr = keeprec(Ur)
        MODE['op'] = None
        res[str(L)] = {'benefit': round(ben, 4), 'keep_combined': round(rc, 4), 'keep_token': round(rt, 4),
                       'keep_position': round(rp, 4), 'keep_random': round(rr, 4), 'irreducible': round(max(1-rc, 0), 4)}
        print(f'L{L}: ben {ben:.3f} | combined {rc:.2f} token {rt:.2f} pos {rp:.2f} rand {rr:.2f} | irreducible {max(1-rc,0):.2f}', flush=True)
    for h in hooks: h.remove()

    xs = np.array([int(L) for L in res])
    tok = np.array([res[str(L)]['keep_token'] for L in xs])
    comb = np.array([res[str(L)]['keep_combined'] for L in xs])
    irr = np.clip(1-comb, 0, 1)
    posadd = np.clip(comb - tok, 0, 1)
    fig, ax = plt.subplots(figsize=(10, 5.2)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.bar(xs, np.clip(tok, 0, 1), color=BLUE, label='token-class (lexical)')
    ax.bar(xs, posadd, bottom=np.clip(tok, 0, 1), color=GREEN, label='+ position')
    ax.bar(xs, irr, bottom=np.clip(comb, 0, 1), color=MUTEDC, label='irreducible (distributed)')
    for L in xs: ax.text(L, 1.02, f'{res[str(L)]["benefit"]:.2f}', ha='center', fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.12); ax.set_xlabel('MLP layer'); ax.set_ylabel('share of layer CE-benefit')
    ax.set_title('How much of each early MLP is token-class + position (interpretable) vs distributed\n'
                 '(number above each bar = layer benefit in nats)', color=INK, fontsize=12.5, loc='left')
    ax.legend(fontsize=9, loc='lower right'); ax.grid(True, axis='y', color=GRID, lw=0.6)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(SECONDARY)
    fig.tight_layout(); fig.savefig(FIG, dpi=150, facecolor=SURFACE); print('wrote', FIG, flush=True)

    real = [L for L in LAYERS if res[str(L)]['benefit'] > 0.1]
    pa = all(res[str(L)]['keep_combined'] >= 0.6 for L in real)
    out = {'layers': LAYERS, 'results': res, 'pred_a_mostly_interpretable': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) early MLPs mostly token-class+position (keep-combined>=0.6 where benefit>0.1): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
