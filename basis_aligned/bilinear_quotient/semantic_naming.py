"""SEMANTIC NAMING + GENERALITY (interpretability payoff for 767). The token-
semantic subspace of mlp0 is canonical + causally dominant + stable. Now: (1) NAME
the top semantic directions -- which token classes sit at each direction's
extremes? (are they human-readable, not just statistically token-driven?), and
(2) test GENERALITY -- is the token-semantic subspace causally dominant across the
stack (layers 0,4,8,12) or specific to layer 0?

REGISTERED PREDICTIONS:
  (0) SANITY: extreme tokens per direction are decodable;
  (a) NAMEABLE: the top mlp0 semantic directions have coherent token classes at
      their extremes (report them; a direction's top-10 tokens are dominated by a
      recognisable class -- punctuation / digits / function words / whitespace);
  (b) GENERAL: the token-semantic subspace is causally dominant (dCE_semantic >=
      5x dCE_random) at EVERY tested layer -- token-identity structure is a
      stack-wide organising axis, not a layer-0 quirk;
  NULL: random same-rank subspace removal is ~harmless at every layer."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'semantic_naming_results.json'
NEVAL = 48; MINCOUNT = 5; RSEM = 64; LAYERS = [0, 4, 8, 12]; NDIR = 10
PROJ = {'U': None, 'layer': 0}


def proj_hook_factory(L):
    def hook(mo, i_, o_):
        if PROJ['U'] is None or PROJ['layer'] != L: return o_
        U = PROJ['U']; sh = o_.shape; o = o_.reshape(-1, D).float()
        return (o - (o @ U) @ U.T).reshape(sh).to(o_.dtype)
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
def capture_out(rows, n, L):
    cap = []; toks = []
    h = m.transformer.h[L].mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1).cpu())
        forward_logits(idx)
    h.remove(); return torch.cat(cap, 0), torch.cat(toks).numpy()


def token_mean_matrix(O, toks):
    g = O.mean(0, keepdim=True); ids = []; rows = []; wt = []
    for t in np.unique(toks):
        mk = toks == t
        if mk.sum() < MINCOUNT: continue
        ids.append(int(t)); rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0); Mw = M * torch.tensor(wt, device=O.device)[:, None]
    Vh = torch.linalg.svd(Mw, full_matrices=False)[2]
    return Vh, M, np.array(ids)                            # directions, unweighted token-means, token ids


def d1(t):
    try: return repr(cl.d1(int(t)))
    except Exception: return f'<{t}>'


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    hooks = [m.transformer.h[L].mlp.register_forward_hook(proj_hook_factory(L)) for L in LAYERS]

    # (a) NAME mlp0 top directions
    O0, toks0 = capture_out(rows, NEVAL, 0)
    Vh, M, ids = token_mean_matrix(O0, toks0)
    proj = (M @ Vh[:NDIR].T).cpu().numpy()                 # (Ntok, NDIR) token-mean projection per direction
    named = []
    for dctr in range(NDIR):
        col = proj[:, dctr]; hi = ids[np.argsort(-col)[:8]]; lo = ids[np.argsort(col)[:8]]
        named.append({'dir': dctr, 'high': [d1(t) for t in hi], 'low': [d1(t) for t in lo]})
        print(f'dir {dctr}:  HIGH {[d1(t) for t in hi]}', flush=True)
        print(f'         LOW  {[d1(t) for t in lo]}', flush=True)

    # (b) GENERALITY: causal ratio per layer
    g = torch.Generator(device=DEV).manual_seed(0); layer_res = {}
    for L in LAYERS:
        OL, tL = capture_out(rows, NEVAL, L)
        VhL = token_mean_matrix(OL, tL)[0]; Us = VhL[:RSEM].T.contiguous()
        PROJ['layer'] = L
        PROJ['U'] = None; ce_full = ce_on(rows, NEVAL)
        PROJ['U'] = Us; ce_sem = ce_on(rows, NEVAL)
        Ur = torch.linalg.qr(torch.randn(D, RSEM, generator=g, device=DEV))[0]
        PROJ['U'] = Ur; ce_rand = ce_on(rows, NEVAL); PROJ['U'] = None
        d_sem = ce_sem - ce_full; d_rand = ce_rand - ce_full
        layer_res[str(L)] = {'dce_semantic': round(d_sem, 4), 'dce_random': round(d_rand, 4),
                             'ratio': round(d_sem/max(d_rand, 1e-6), 2)}
        print(f'L{L}: dCE_semantic {d_sem:.3f}  dCE_random {d_rand:.3f}  ratio {d_sem/max(d_rand,1e-6):.1f}', flush=True)
    for h in hooks: h.remove()

    pb = all(layer_res[str(L)]['dce_semantic'] >= 5*max(layer_res[str(L)]['dce_random'], 1e-6) for L in LAYERS)
    null_ok = all(layer_res[str(L)]['dce_random'] < 0.05 for L in LAYERS)
    out = {'named_directions': named, 'layer_causal': layer_res, 'layers': LAYERS,
           'pred_b_general': bool(pb), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(b) semantic subspace causally dominant at EVERY layer (>=5x): {pb}; NULL random harmless: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
