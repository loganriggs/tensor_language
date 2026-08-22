"""IS THE LINEAR->NONLINEAR->LINEAR DEPTH ARC (§941) UNIVERSAL? In bilin18 the MLPs are ~linear at the front and
readout but ~60% multiplicative in the middle. Is a middle-nonlinearity dip a general LM property (like §937 bag,
§925 separability)? For bilin18 and GPT-2 / GPT-2-large (GELU MLPs), fit a ridge LINEAR map from each MLP's input
to its output (train) and measure the held-out R^2 (variance of the output explained by a linear read of the
input). Low R^2 = the layer's nonlinearity is heavily used. Plot R^2 by RELATIVE depth.

REGISTERED PREDICTIONS:
  (0) SANITY: front-layer R^2 is high in every model; a shuffled-input linear map has R^2 ~0.
  (a) UNIVERSAL MIDDLE DIP: every model shows a MIDDLE DIP in linear-R^2 (middle MLPs least linearly
      reconstructable) with more-linear front and back -> linear->nonlinear->linear is a general depth pattern,
      not a bilin18 quirk;
  (b) report per-layer R^2 for each model + the front/middle/back means."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m as BILIN, DEV
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_nonlinearity_crossmodel_results.json'
NEVAL = 120; SEQ = 256; RIDGE_MAP = 1e3


def bilin_forward(idx):
    x = F.rms_norm(BILIN.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in BILIN.transformer.h: x, v1 = blk(x, v1, x0)
    return x


def r2_by_layer(mlps, run_forward, blocks, Dm):
    nb = blocks.shape[0]; ntr = int(0.7*nb); TRAIN = np.zeros(nb, bool); TRAIN[:ntr] = True
    trm = torch.tensor(np.repeat(TRAIN, SEQ-1), device=DEV)
    capin = {L: [] for L in range(len(mlps))}; capout = {L: [] for L in range(len(mlps))}; hs = []
    for L, mlp in enumerate(mlps):
        def mk(L):
            def h(mo, i_, o_):
                xin = i_[0] if isinstance(i_, tuple) else i_
                capin[L].append(xin.detach().float().reshape(-1, Dm))
                capout[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, Dm))
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    for i in range(0, nb, 4): run_forward(blocks[i:i+4].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    r2s = []; r2sh = []
    for L in range(len(mlps)):
        Xin = torch.cat(capin[L], 0); O = torch.cat(capout[L], 0); capin[L] = None; capout[L] = None
        Xtr = Xin[trm]; Otr = O[trm]; Xte = Xin[~trm]; Ote = O[~trm]
        A = Xtr.T @ Xtr + RIDGE_MAP*torch.eye(Dm, device=DEV); M = torch.linalg.solve(A, Xtr.T @ Otr)
        pred = Xte @ M; ss_res = (Ote - pred).pow(2).sum(); ss_tot = (Ote - Ote.mean(0)).pow(2).sum()
        r2s.append(round(float(1 - ss_res/ss_tot), 3))
        pr = torch.randperm(Xtr.shape[0], generator=torch.Generator(device=DEV).manual_seed(0), device=DEV)
        Msh = torch.linalg.solve(A, Xtr.T @ Otr[pr]); psh = Xte @ Msh
        r2sh.append(round(float(1 - (Ote-psh).pow(2).sum()/ss_tot), 3))
        del Xin, O
    return r2s, r2sh


def bands(r2):
    n = len(r2); a = n//3
    return round(float(np.mean(r2[:a])), 3), round(float(np.mean(r2[a:2*a])), 3), round(float(np.mean(r2[2*a:])), 3)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    out = {'models': {}}
    # bilin18
    r2, r2sh = r2_by_layer([blk.mlp for blk in BILIN.transformer.h], bilin_forward, blocks, D)
    f, mi, b = bands(r2); out['models']['bilin18'] = {'r2_by_layer': r2, 'shuffled_r2_by_layer': r2sh, 'front': f, 'middle': mi, 'back': b}
    print(f"bilin18: front {f} middle {mi} back {b} | per-layer {r2}", flush=True)
    # GPT-2 family
    for mid in ['gpt2', 'gpt2-large']:
        mdl = AutoModelForCausalLM.from_pretrained(mid).to(DEV).eval(); Dm = mdl.config.n_embd
        r2, r2sh = r2_by_layer([blk.mlp for blk in mdl.transformer.h], lambda idx: mdl(idx), blocks, Dm)
        f, mi, b = bands(r2); out['models'][mid] = {'r2_by_layer': r2, 'shuffled_r2_by_layer': r2sh, 'front': f, 'middle': mi, 'back': b}
        print(f"{mid}: front {f} middle {mi} back {b} | per-layer {r2}", flush=True)
        del mdl; torch.cuda.empty_cache()
    out['pred_a_universal_middle_dip'] = bool(all(out['models'][k]['middle'] < out['models'][k]['front'] and out['models'][k]['middle'] < out['models'][k]['back'] for k in out['models']))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) universal middle-nonlinearity dip (middle R2 < front and back in all): {out['pred_a_universal_middle_dip']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
