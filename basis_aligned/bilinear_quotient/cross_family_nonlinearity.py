"""FINAL family-generalization axis: does the linear->nonlinear->linear DEPTH ARC (§941/§942) hold in the sibling
family models? §942 found the middle-nonlinearity dip universal across bilin18 + GPT-2/large; complete the family
picture with swiglu18 (SwiGLU MLP) and bilin12. Per-layer held-out R^2 of a linear map from each MLP's input to
its output (residual-only, transform-invariant). Low R^2 = nonlinearity heavily used.

REGISTERED PREDICTIONS:
  (0) SANITY: bilin18 reproduces its front>middle<back arc.
  (a) FAMILY-WIDE ARC: swiglu18 and bilin12 also show a MIDDLE DIP in linear-R^2 (middle < front and < back) ->
      the linear->nonlinear->linear depth arc is a family property (with GPT-2, general); report band means."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
import census_lib as cl
from bilin18_joint_removal import m as BILIN, DEV
from tier2_model import load_elriggs
import torch.nn.functional as F

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_family_nonlinearity_results.json'
NEVAL = 100; SEQ = 256; RIDGE_MAP = 1e3


@torch.no_grad()
def r2_by_layer(mdl, blocks, Dm, nlayer):
    nb = blocks.shape[0]; ntr = int(0.7*nb); TRAIN = np.zeros(nb, bool); TRAIN[:ntr] = True
    trm = torch.tensor(np.repeat(TRAIN, SEQ-1), device=DEV)
    capin = {L: [] for L in range(nlayer)}; capout = {L: [] for L in range(nlayer)}; hs = []
    for L in range(nlayer):
        def mk(L):
            def h(mo, i_, o_):
                capin[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, Dm))
                capout[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, Dm))
            return h
        hs.append(mdl.transformer.h[L].mlp.register_forward_hook(mk(L)))
    for i in range(0, nb, 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous()
        x = F.rms_norm(mdl.transformer.wte(idx), (Dm,)); x0 = x; v1 = None
        for blk in mdl.transformer.h: x, v1 = blk(x, v1, x0)
    for h in hs: h.remove()
    r2s = []
    for L in range(nlayer):
        Xin = torch.cat(capin[L], 0); O = torch.cat(capout[L], 0); capin[L] = None; capout[L] = None
        A = Xin[trm].T @ Xin[trm] + RIDGE_MAP*torch.eye(Dm, device=DEV); M = torch.linalg.solve(A, Xin[trm].T @ O[trm])
        pred = Xin[~trm] @ M; ss_res = (O[~trm]-pred).pow(2).sum(); ss_tot = (O[~trm]-O[~trm].mean(0)).pow(2).sum()
        r2s.append(round(float(1 - ss_res/ss_tot), 3)); del Xin, O
    return r2s


def bands(r2):
    n = len(r2); a = n//3
    return round(float(np.mean(r2[:a])), 3), round(float(np.mean(r2[a:2*a])), 3), round(float(np.mean(r2[2*a:])), 3)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    out = {'models': {}}
    r2 = r2_by_layer(BILIN, blocks, 1152, 18); f, mi, b = bands(r2)
    out['models']['bilin18'] = {'r2': r2, 'front': f, 'middle': mi, 'back': b}; print(f"bilin18: front {f} middle {mi} back {b}", flush=True)
    for short in ['swiglu18', 'bilin12']:
        try:
            mdl, cfg = load_elriggs(short); Dm = cfg.get('n_embd'); nl = cfg.get('n_layer')
            r2 = r2_by_layer(mdl, blocks, Dm, nl); f, mi, b = bands(r2)
            out['models'][short] = {'r2': r2, 'front': f, 'middle': mi, 'back': b}; print(f"{short}: front {f} middle {mi} back {b}", flush=True)
            del mdl; torch.cuda.empty_cache()
        except Exception as e:
            out['models'][short] = {'error': f'{type(e).__name__}: {e}'}; print(f"{short} ERROR {e}", flush=True)
    ok = all('middle' in out['models'][k] and out['models'][k]['middle'] < out['models'][k]['front'] and out['models'][k]['middle'] < out['models'][k]['back'] for k in out['models'])
    out['pred_a_family_arc'] = bool(ok)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) linear->nonlinear->linear arc family-wide (middle dip all): {ok}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
