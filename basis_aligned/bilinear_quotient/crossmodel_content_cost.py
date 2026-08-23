"""§1057 showed the middle content SUBSPACE is universal across the bilinear family; this checks it is LOAD-BEARING
family-wide (closing §1057's caveat). Replicate bilin18's front-linear / middle-nonlinear pattern (§1000/§1042) on
bilin12 and swiglu18: replace a single layer's MLP output with a held-out LINEAR map of its input, and measure
loss-recovery vs mean-ablate. A FRONT layer should be well-recovered by a linear map (grammar is ~linear/token); a
MIDDLE layer should be POORLY recovered (its content is bilinear/high-rank, a linear map cannot capture it). Per-layer
(single replacement) to avoid the compounding confound (§1050).

REGISTERED PREDICTIONS:
  (0) SANITY: mean-ablate = recovery 0; shuffled-input linear ~0.
  (a) FAMILY FRONT IS ~LINEAR: a front layer's MLP (bilin12 L1 / swiglu18 L1) has HIGH linear-map loss-recovery (>0.7);
  (b) FAMILY MIDDLE IS NONLINEAR/HIGH-RANK: a middle layer's MLP (bilin12 L6 / swiglu18 L10) has markedly LOWER linear
      recovery than the front -> the load-bearing high-rank content is a family property, not just a bilin18 subspace
      artifact. Report per-model per-layer linear recovery + shuffled null."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/rspd')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'crossmodel_content_cost_results.json'
DEV = 'cuda'; NEVAL = 200; SEQ = 256; RIDGE = 1e2
# per model: (front_layer, middle_layer)
MODELS = {'bilin12': (1, 6), 'swiglu18': (1, 10)}
SUB = {'L': None, 'mode': None, 'M': None, 'gmean': None, 'D': None}


def hookify(mdl, L):
    def h(mo, i_, o_):
        if SUB['L'] != L or SUB['mode'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_).float(); o = o_[0] if isinstance(o_, tuple) else o_
        B, T, D = o.shape
        if SUB['mode'] == 'mean':
            ny = SUB['gmean'].view(1, 1, D).expand(B, T, D)
        else:
            xf = x.reshape(-1, D)
            if SUB['mode'] == 'shuf': xf = xf[torch.randperm(xf.shape[0], device=DEV)]
            x1 = torch.cat([xf, torch.ones(xf.shape[0], 1, device=DEV)], 1)
            ny = (x1 @ SUB['M']).reshape(B, T, D)
        return (ny.to(o.dtype),) + tuple(o_[1:]) if isinstance(o_, tuple) else ny.to(o.dtype)
    return h


@torch.no_grad()
def capture(mdl, L, blocks, D):
    xs = []; ys = []; hs = []
    mlp = mdl.transformer.h[L].mlp
    def h(mo, i_, o_):
        xs.append((i_[0] if isinstance(i_, tuple) else i_).float().reshape(-1, D))
        ys.append((o_[0] if isinstance(o_, tuple) else o_).float().reshape(-1, D))
    hh = mlp.register_forward_hook(h)
    for i in range(0, blocks.shape[0], 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous(); mdl(idx, idx)
    hh.remove()
    return torch.cat(xs, 0), torch.cat(ys, 0)


@torch.no_grad()
def ce(mdl, blocks):
    tot = 0.0; nb = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        out = mdl(idx, tgt); loss = out[1] if isinstance(out, (tuple, list)) else out
        tot += float(loss) * idx.shape[0]; nb += idx.shape[0]
    return tot / nb


@torch.no_grad()
def run(name, front, mid, blocks):
    mdl, cfg = load_elriggs(name, device=DEV, dtype=torch.float32); mdl.eval()
    D = mdl.transformer.wte.weight.shape[1]; SUB['D'] = D
    nb = blocks.shape[0]; ntr = int(0.6*nb); tr = blocks[:ntr].contiguous(); te = blocks[ntr:].contiguous()
    res = {'D': D, 'layers': len(mdl.transformer.h), 'per_layer': {}}
    hooks = {L: mdl.transformer.h[L].mlp.register_forward_hook(hookify(mdl, L)) for L in (front, mid)}
    SUB['L'] = None; SUB['mode'] = None; ce_full = ce(mdl, te); res['ce_full'] = round(ce_full, 4)
    for L, tag in ((front, 'front'), (mid, 'middle')):
        X, Y = capture(mdl, L, tr, D)
        X1 = torch.cat([X, torch.ones(X.shape[0], 1, device=DEV)], 1)
        SUB['M'] = torch.linalg.solve(X1.T @ X1 + RIDGE*torch.eye(D+1, device=DEV), X1.T @ Y)
        SUB['gmean'] = Y.mean(0); del X, Y, X1
        SUB['L'] = L
        SUB['mode'] = 'mean'; ce_ma = ce(mdl, te); denom = max(ce_ma - ce_full, 1e-6)
        SUB['mode'] = 'lin'; ce_lin = ce(mdl, te)
        SUB['mode'] = 'shuf'; ce_sh = ce(mdl, te)
        SUB['mode'] = None; SUB['L'] = None
        res['per_layer'][f'{tag}_L{L}'] = {
            'meanabl_cost': round(ce_ma-ce_full, 4),
            'linear_recovery': round(float((ce_ma-ce_lin)/denom), 3),
            'shuffled_null': round(float((ce_ma-ce_sh)/denom), 3)}
        print(f"{name} {tag} L{L}: cost {ce_ma-ce_full:.3f} | linear-recovery {res['per_layer'][f'{tag}_L{L}']['linear_recovery']} | null {res['per_layer'][f'{tag}_L{L}']['shuffled_null']}", flush=True)
    for h in hooks.values(): h.remove()
    del mdl; torch.cuda.empty_cache()
    return res


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    out = {'bilin18_ref': 'front mlp1 ~0.9 linear (table); deep-middle full-rank (~0.4-0.6)', 'models': {}}
    for name, (fr, mi) in MODELS.items():
        print(f"=== {name} ===", flush=True); out['models'][name] = run(name, fr, mi, blocks)
    ok = True
    for name in MODELS:
        pl = out['models'][name]['per_layer']
        fr = [v for k, v in pl.items() if k.startswith('front')][0]['linear_recovery']
        mi = [v for k, v in pl.items() if k.startswith('middle')][0]['linear_recovery']
        ok = ok and (fr > mi)
    out['pred_front_linear_gt_middle'] = bool(ok)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred front-linear > middle (family-wide) {out['pred_front_linear_gt_middle']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
