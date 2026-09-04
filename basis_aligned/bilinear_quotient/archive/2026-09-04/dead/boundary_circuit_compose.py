"""BOUNDARY CIRCUIT COMPOSE (find + verify a cross-layer circuit; user:
compose circuits across components, verify the decomposition). The boundary
-> continuation motif appears SEPARATELY at three components: block0.attn
dir1 (703), block1.attn rank-1 (701), mlp16 rank-1 (715). Are they ONE
composed cross-layer circuit? Test SELECTIVITY: ablating each should hurt
next-token prediction AT sentence-boundary positions (current token . ! ? \n)
MORE than at non-boundary positions. And composition: the three together
should be boundary-selective, above random directions.

For each component, ablate = project its A-SVD write direction OUT of that
module's output during the forward. Metric: selectivity = CE_increase(at
boundary positions) - CE_increase(at non-boundary positions).

REGISTERED PREDICTIONS:
  (0) SANITY: baseline CE reproduces; boundary/non-boundary buckets non-empty;
  (a) BOUNDARY-SELECTIVE: each of the 3 named components, ablated, raises
      boundary-position CE MORE than non-boundary CE (selectivity > 0), and
      the THREE TOGETHER are strongly boundary-selective -- they compose
      into a boundary->continuation circuit;
  (b) report selectivity per component + combined;
  NULL: random write directions (same rank, same layers) have selectivity
      ~0 (no boundary preference)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'boundary_circuit_compose_results.json'
NFIT = 64; NEVAL = 96
BOUND = {'.', '!', '?', '\n', '."', '.)', '?"', '!"', '."', ';'}

ABL = {}   # module_name -> unit direction (D,) to project out, or None


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps); B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


@torch.no_grad()
def capture_in(mod, rows, n, in_dim):
    cap = []
    h = mod.register_forward_pre_hook(lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, in_dim).cpu()))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0)


def mk_hook(name):
    def hook(mo, i_, o_):
        d = ABL.get(name)
        if d is None: return o_
        of = o_.float(); proj = (of @ d)[..., None] * d
        return (of - proj).to(o_.dtype)
    return hook


@torch.no_grad()
def per_tok_ce(rows, n):
    ce = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        ce.append(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='none').cpu())
    return torch.cat(ce).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT); ev = cl.fineweb_rows(NEVAL)
    # boundary mask: CURRENT token (input) is a sentence-ender
    bmask = []
    for r in range(NEVAL):
        for t in ev[r, :256].tolist():
            s = cl.d1(int(t)).strip()
            bmask.append(s in BOUND or s.endswith('.') or s.endswith('?') or s.endswith('!'))
    bmask = np.array(bmask[:NEVAL*256])
    print(f'boundary positions: {bmask.sum()} / {len(bmask)}', flush=True)

    # extract the 3 named write directions
    comps = {}
    b0 = m.transformer.h[0].attn.c_proj
    X = capture_in(b0, rows, NFIT, D).to(DEV)
    A, _ = asvd_fast(b0.weight.data.float().to(DEV), X); comps['block0.attn'] = (A[:, 1] / A[:, 1].norm())  # dir1
    b1 = m.transformer.h[1].attn.c_proj
    X = capture_in(b1, rows, NFIT, D).to(DEV)
    A, _ = asvd_fast(b1.weight.data.float().to(DEV), X); comps['block1.attn'] = (A[:, 0] / A[:, 0].norm())
    d16 = m.transformer.h[16].mlp.Down
    X = capture_in(d16, rows, NFIT, HID).to(DEV)
    A, _ = asvd_fast(d16.weight.data.float().to(DEV), X); comps['mlp16'] = (A[:, 0] / A[:, 0].norm())

    mods = {'block0.attn': b0, 'block1.attn': b1, 'mlp16': m.transformer.h[16].mlp}
    hooks = {name: mods[name].register_forward_hook(mk_hook(name)) for name in mods}

    for n in ABL: ABL[n] = None
    base = per_tok_ce(ev, NEVAL)

    def selectivity(active):
        for n in mods: ABL[n] = None
        for n in active: ABL[n] = comps[n] if n != 'mlp16' else comps[n]
        ce = per_tok_ce(ev, NEVAL)
        for n in mods: ABL[n] = None
        dc = ce - base
        return float(dc[bmask].mean()), float(dc[~bmask].mean())

    res = {}
    for name in comps:
        bnd, non = selectivity([name])
        res[name] = {'boundary_dCE': round(bnd, 4), 'nonboundary_dCE': round(non, 4),
                     'selectivity': round(bnd - non, 4)}
        print(f'{name:12s}: boundary dCE {bnd:+.3f}  non {non:+.3f}  selectivity {bnd-non:+.3f}', flush=True)
    bnd_all, non_all = selectivity(list(comps))
    res['ALL'] = {'boundary_dCE': round(bnd_all, 4), 'nonboundary_dCE': round(non_all, 4),
                  'selectivity': round(bnd_all - non_all, 4)}
    print(f'{"ALL 3":12s}: boundary dCE {bnd_all:+.3f}  non {non_all:+.3f}  selectivity {bnd_all-non_all:+.3f}', flush=True)

    # null: random directions same layers
    g = torch.Generator().manual_seed(0)
    rnd = {}
    rnd['block0.attn'] = torch.randn(D, generator=g); rnd['block1.attn'] = torch.randn(D, generator=g)
    rnd['mlp16'] = torch.randn(D, generator=g)
    for n in rnd: rnd[n] = (rnd[n]/rnd[n].norm()).to(DEV)
    for n in mods: ABL[n] = None
    for n in mods: ABL[n] = rnd[n]
    ce = per_tok_ce(ev, NEVAL); dc = ce - base
    for n in mods: ABL[n] = None
    null_sel = float(dc[bmask].mean() - dc[~bmask].mean())
    for name in hooks: hooks[name].remove()
    print(f'\nNULL random dirs (all 3): selectivity {null_sel:+.3f}', flush=True)

    pa = all(res[n]['selectivity'] > 0 for n in comps) and res['ALL']['selectivity'] > 0
    null_ok = abs(null_sel) < 0.5 * res['ALL']['selectivity']
    print(f'(a) each + combined boundary-selective: {pa}; NULL ~0: {null_ok}', flush=True)

    out = {'components': res, 'null_selectivity': round(null_sel, 4),
           'n_boundary': int(bmask.sum()), 'pred_a': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
