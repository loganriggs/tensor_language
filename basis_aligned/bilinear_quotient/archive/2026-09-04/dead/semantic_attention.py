"""SEMANTIC SUBSPACE in ATTENTION (does 767-770's canonical token-class structure
generalise to the other component type?). Attention MIXES across positions, so its
output is more CONTEXT-driven than the MLP's -- prediction: attention has a token-
semantic subspace too, but it is LESS causally dominant / lower-share than the MLP's
(more of attention's output is context, not current-token identity). Repeat the
necessary+sufficient+causal-ratio analysis on the attention OUTPUT (x1) of layers
0 and 4, and name a few top directions.

REGISTERED PREDICTIONS:
  (0) SANITY: attention token-means separate (nonzero semantic variance);
  (a) GENERALISES BUT WEAKER: attention has a causal token-semantic subspace
      (remove top-64 -> dCE >= 3x random) at layers 0/4, BUT the keep-only-semantic
      sufficiency at r=64 is LOWER than the MLP's 0.92 (attention output is more
      context-driven -> a smaller share is current-token identity);
  (b) report per-layer causal ratio + keep-only-64 sufficiency + a few named dirs;
  NULL: random same-rank subspace is ~harmless / insufficient."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'semantic_attention_results.json'
NEVAL = 48; MINCOUNT = 5; RSEM = 64; LAYERS = [0, 4]; NDIR = 6
MODE = {'U': None, 'layer': 0, 'op': None}    # op: 'remove' | 'keep' | 'ablate'


def attn_hook_factory(L):
    def hook(mo, i_, o_):
        if MODE['layer'] != L or MODE['op'] is None: return o_
        x1 = o_[0] if isinstance(o_, tuple) else o_
        sh = x1.shape; v = x1.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else:
            U = MODE['U']
            v2 = (v @ U) @ U.T if MODE['op'] == 'keep' else v - (v @ U) @ U.T
        x1n = v2.reshape(sh).to(x1.dtype)
        return (x1n, o_[1]) if isinstance(o_, tuple) else x1n
    return hook


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
def capture_attn(rows, n, L):
    cap = []; toks = []
    def hk(mo, i_, o_):
        x1 = o_[0] if isinstance(o_, tuple) else o_
        cap.append(x1.detach().float().reshape(-1, D))
    h = m.transformer.h[L].attn.register_forward_hook(hk)
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1).cpu())
        forward_logits(idx)
    h.remove(); return torch.cat(cap, 0), torch.cat(toks).numpy()


def semantic_dirs(O, toks):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(toks):
        mk = toks == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    ids = np.array([int(t) for t in np.unique(toks) if (toks == t).sum() >= MINCOUNT])
    Vh = torch.linalg.svd(M, full_matrices=False)[2]
    return Vh, torch.stack(rows, 0), ids


def d1(t):
    try: return repr(cl.d1(int(t)))
    except Exception: return f'<{t}>'


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    hooks = [m.transformer.h[L].attn.register_forward_hook(attn_hook_factory(L)) for L in LAYERS]
    g = torch.Generator(device=DEV).manual_seed(0); layer_res = {}; named = {}
    for L in LAYERS:
        O, toks = capture_attn(rows, NEVAL, L)
        Vh, M, ids = semantic_dirs(O, toks); Us = Vh[:RSEM].T.contiguous()
        MODE['layer'] = L
        MODE['op'] = None; ce_full = ce_on(rows, NEVAL)
        MODE['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL)
        ben = ce_abl - ce_full
        MODE['op'] = 'remove'; MODE['U'] = Us; ce_rem = ce_on(rows, NEVAL)
        Ur = torch.linalg.qr(torch.randn(D, RSEM, generator=g, device=DEV))[0]
        MODE['U'] = Ur; ce_rem_rand = ce_on(rows, NEVAL)
        MODE['op'] = 'keep'; MODE['U'] = Us; ce_keep = ce_on(rows, NEVAL)
        MODE['U'] = Ur; ce_keep_rand = ce_on(rows, NEVAL)
        MODE['op'] = None; MODE['U'] = None
        d_sem = ce_rem - ce_full; d_rand = ce_rem_rand - ce_full
        keep_rec = (ce_abl - ce_keep)/max(ben, 1e-6); keep_rand_rec = (ce_abl - ce_keep_rand)/max(ben, 1e-6)
        layer_res[str(L)] = {'benefit': round(ben, 4), 'remove_ratio': round(float(d_sem/max(d_rand, 1e-6)), 2),
                             'dce_semantic': round(d_sem, 4), 'dce_random': round(d_rand, 4),
                             'keep64_recovery': round(float(keep_rec), 4), 'keep64_random': round(float(keep_rand_rec), 4)}
        proj = (M @ Vh[:NDIR].T).cpu().numpy()
        named[str(L)] = []
        for dctr in range(NDIR):
            col = proj[:, dctr]; hi = ids[np.argsort(-col)[:6]]; lo = ids[np.argsort(col)[:6]]
            named[str(L)].append({'dir': dctr, 'high': [d1(t) for t in hi], 'low': [d1(t) for t in lo]})
        print(f'L{L}: benefit {ben:.3f} | remove-ratio {layer_res[str(L)]["remove_ratio"]} | '
              f'keep64 {keep_rec:.3f} (rand {keep_rand_rec:.3f})', flush=True)
        print(f'   dir0 HIGH {named[str(L)][0]["high"]}', flush=True)
    for h in hooks: h.remove()

    pa = all(layer_res[str(L)]['remove_ratio'] >= 3 for L in LAYERS)
    weaker = layer_res['0']['keep64_recovery'] < 0.92
    null_ok = all(layer_res[str(L)]['keep64_random'] < layer_res[str(L)]['keep64_recovery'] for L in LAYERS)
    out = {'layers': LAYERS, 'layer_results': layer_res, 'named': named,
           'pred_a_generalises': bool(pa), 'attn_weaker_than_mlp': bool(weaker),
           'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) attention has causal token-semantic subspace (>=3x): {pa}; weaker-than-MLP (keep64<0.92): {weaker}; NULL: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
