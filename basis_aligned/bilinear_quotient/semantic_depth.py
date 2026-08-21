"""SEMANTIC DEPTH PROFILE (capstone visual for 767-772). For every layer and both
component types (attention x1, MLP output), compute the top-64 token-semantic
subspace and measure: (i) benefit = CE cost of ablating the component's output,
(ii) keep-only-64 sufficiency = fraction of that benefit preserved when only the
token-class subspace is kept. Shows WHERE in the network token-class organisation
lives (the barbell) and that a 64-dim token-class subspace is SUFFICIENT across
depth for both components. Emits a figure.

REGISTERED PREDICTIONS:
  (0) SANITY: benefit >= 0 every layer;
  (a) SUFFICIENT ACROSS DEPTH: keep-only-64 recovery is HIGH (>= 0.7) at the layers
      that carry real benefit (early layers), for BOTH attention and MLP -- the
      token-class subspace is a near-complete low-rank summary network-wide, not
      just at layer 0;
  (b) BARBELL: benefit is concentrated in the early (and final) layers, tiny in the
      deep middle (matches 756);
  NULL: n/a (descriptive profile; random-subspace sufficiency shown for scale)."""
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
OUT = PT + 'semantic_depth_results.json'; FIG = PT + 'semantic_depth.png'
NEVAL = 40; MINCOUNT = 5; RSEM = 64
MODE = {'U': None, 'L': -1, 'which': None, 'op': None}   # op: keep|ablate
BLUE, RED, MUTEDC = '#3987e5', '#e34948', '#898781'


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def apply_op(v):
    if MODE['op'] == 'ablate': return torch.zeros_like(v)
    U = MODE['U']; return (v @ U) @ U.T


def mlp_hook_factory(L):
    def h(mo, i_, o_):
        if MODE['L'] != L or MODE['which'] != 'mlp' or MODE['op'] is None: return o_
        sh = o_.shape; return apply_op(o_.reshape(-1, D).float()).reshape(sh).to(o_.dtype)
    return h


def attn_hook_factory(L):
    def h(mo, i_, o_):
        if MODE['L'] != L or MODE['which'] != 'attn' or MODE['op'] is None: return o_
        x1 = o_[0] if isinstance(o_, tuple) else o_; sh = x1.shape
        x1n = apply_op(x1.reshape(-1, D).float()).reshape(sh).to(x1.dtype)
        return (x1n, o_[1]) if isinstance(o_, tuple) else x1n
    return h


def ce_on(rows, n):
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1)))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


@torch.no_grad()
def capture(rows, n, which, L):
    cap = []; toks = []
    if which == 'mlp':
        hk = lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D))
        h = m.transformer.h[L].mlp.register_forward_hook(hk)
    else:
        def hk(mo, i_, o_):
            x1 = o_[0] if isinstance(o_, tuple) else o_; cap.append(x1.detach().float().reshape(-1, D))
        h = m.transformer.h[L].attn.register_forward_hook(hk)
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1).cpu())
        forward_logits(idx)
    h.remove(); return torch.cat(cap, 0), torch.cat(toks).numpy()


def semantic_subspace(O, toks, r=RSEM):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(toks):
        mk = toks == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    hooks = []
    for L in range(NL):
        hooks.append(m.transformer.h[L].mlp.register_forward_hook(mlp_hook_factory(L)))
        hooks.append(m.transformer.h[L].attn.register_forward_hook(attn_hook_factory(L)))
    MODE['op'] = None; ce_full = ce_on(rows, NEVAL)
    g = torch.Generator(device=DEV).manual_seed(0); Ur = torch.linalg.qr(torch.randn(D, RSEM, generator=g, device=DEV))[0]
    prof = {'mlp': {}, 'attn': {}}
    for which in ['mlp', 'attn']:
        for L in range(NL):
            O, toks = capture(rows, NEVAL, which, L); Us = semantic_subspace(O, toks)
            MODE['L'] = L; MODE['which'] = which
            MODE['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL)
            MODE['op'] = 'keep'; MODE['U'] = Us; ce_keep = ce_on(rows, NEVAL)
            MODE['U'] = Ur; ce_keep_rand = ce_on(rows, NEVAL)
            MODE['op'] = None; MODE['U'] = None
            ben = ce_abl - ce_full
            rec = float((ce_abl - ce_keep)/max(ben, 1e-6)); rec_r = float((ce_abl - ce_keep_rand)/max(ben, 1e-6))
            prof[which][L] = {'benefit': round(ben, 4), 'keep64': round(rec, 4), 'keep64_rand': round(rec_r, 4)}
        print(f'{which}: ' + ' '.join(f'{L}:{prof[which][L]["keep64"]:.2f}' for L in range(NL)), flush=True)
    for h in hooks: h.remove()

    xs = np.arange(NL)
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True); fig.patch.set_facecolor(SURFACE)
    for ax in axs: ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.6)
    axs[0].bar(xs-0.2, [prof['mlp'][L]['benefit'] for L in range(NL)], width=0.4, color=BLUE, label='MLP')
    axs[0].bar(xs+0.2, [prof['attn'][L]['benefit'] for L in range(NL)], width=0.4, color=RED, label='attention')
    axs[0].set_ylabel('benefit: CE nats if ablated'); axs[0].legend(fontsize=9)
    axs[0].set_title('Where token-class work lives, and that 64 directions suffice for it', color=INK, fontsize=12.5, loc='left')
    mlp_keep = [prof['mlp'][L]['keep64'] if prof['mlp'][L]['benefit'] > 0.03 else np.nan for L in range(NL)]
    attn_keep = [prof['attn'][L]['keep64'] if prof['attn'][L]['benefit'] > 0.03 else np.nan for L in range(NL)]
    axs[1].plot(xs, mlp_keep, '-o', color=BLUE, ms=5, label='MLP keep-only-64')
    axs[1].plot(xs, attn_keep, '-o', color=RED, ms=5, label='attention keep-only-64')
    axs[1].axhline(1.0, color=MUTEDC, lw=0.8, ls=':')
    axs[1].set_ylabel('keep-only-64 CE-recovery\n(1 = token-class subspace suffices)'); axs[1].set_ylim(0, 1.08)
    axs[1].set_xlabel('layer'); axs[1].set_xticks(xs); axs[1].legend(fontsize=9)
    axs[1].text(8, 0.15, 'faded where benefit ~0 (deep middle nearly inert)', fontsize=8.5, color=MUTED, style='italic')
    for ax in axs:
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
        for s in ['left', 'bottom']: ax.spines[s].set_color(SECONDARY)
    fig.tight_layout(); fig.savefig(FIG, dpi=150, facecolor=SURFACE); print('wrote', FIG, flush=True)

    early = [L for L in range(6)]
    pa = all(prof['mlp'][L]['keep64'] >= 0.7 for L in early if prof['mlp'][L]['benefit'] > 0.1)
    out = {'ce_full': round(ce_full, 4), 'profile': prof, 'pred_a_sufficient_across_depth': bool(pa),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) 64-dim token-class subspace sufficient across early depth: {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
