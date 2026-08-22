"""IS bilin18's DISTRIBUTED-COOPERATIVE structure (§956, super-additive attention) due to the BILINEAR MLP
specifically, or the training FAMILY? §956 found bilin18 super-additive (3.52) vs GPT-2 sub-additive (0.71). Test
the sibling family models trained the same way with DIFFERENT MLPs: swiglu18 (SwiGLU MLP, same 18L/1152/9h as
bilin18) and bilin12 (bilinear, 12L). If swiglu18 is ALSO super-additive, the distributed structure is
family/training-driven; if swiglu18 is sub-additive (like GPT-2), it is specific to the bilinear MLP.

Transform-invariant metric (avoids per-model output-clamp differences): super-additivity ratio on the FINAL-LAYER
RESIDUAL change. For the front third of attention layers, mean-ablate each singly and sum ||ΔR_final||²; mean-ablate
the whole front band together and take ||ΔR_final||²; ratio = band / sum-of-singles. bilin18 measured in-experiment
too (residual metric) for a fair comparison.

REGISTERED PREDICTIONS:
  (0) SANITY: bilin18 ratio > 1 (super-additive) on the residual metric, reproducing §956's sign.
  (a) report swiglu18 and bilin12 ratios and state plainly: if swiglu18 >> 1 (like bilin18), distributed-
      cooperative is FAMILY-driven; if swiglu18 ~<1, it is bilinear-MLP-specific. No forced direction (calibration)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
import census_lib as cl
from bilin18_joint_removal import m as BILIN, DEV
from tier2_model import load_elriggs
import torch.nn.functional as F

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_family_superadditivity_results.json'
NEVAL = 96; SEQ = 256
ABL = {'layers': set(), 'means': None}


def attn_hook(L, Dm):
    def h(mo, i_, o_):
        if L not in ABL['layers']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        yn = ABL['means'][L].view(1, 1, Dm).expand(B, T, Dm).clone()
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


@torch.no_grad()
def final_resid(mdl, idx, Dm):
    x = F.rms_norm(mdl.transformer.wte(idx), (Dm,)); x0 = x; v1 = None
    for blk in mdl.transformer.h: x, v1 = blk(x, v1, x0)
    return x  # (b,T,Dm) final residual, before readout


@torch.no_grad()
def measure(mdl, blocks, Dm, nlayer):
    front = list(range(0, nlayer//3))
    # attn output means over data
    sums = {L: torch.zeros(Dm, device=DEV) for L in front}; cnt = 0; hs = []
    for L in front:
        def mk(L):
            def h(mo, i_, o_): sums[L] += (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, Dm).sum(0)
            return h
        hs.append(mdl.transformer.h[L].attn.register_forward_hook(mk(L)))
    for i in range(0, blocks.shape[0], 4): final_resid(mdl, blocks[i:i+4].to(DEV)[:, :-1].contiguous(), Dm); cnt += min(4, blocks.shape[0]-i)*(SEQ-1)
    for h in hs: h.remove()
    ABL['means'] = {L: sums[L]/cnt for L in front}
    hooks = [mdl.transformer.h[L].attn.register_forward_hook(attn_hook(L, Dm)) for L in front]
    # clean final residual
    ABL['layers'] = set()
    Rc = []
    for i in range(0, blocks.shape[0], 4): Rc.append(final_resid(mdl, blocks[i:i+4].to(DEV)[:, :-1].contiguous(), Dm).reshape(-1, Dm))
    Rc = torch.cat(Rc, 0)
    def deltasq(active):
        ABL['layers'] = active; R = []
        for i in range(0, blocks.shape[0], 4): R.append(final_resid(mdl, blocks[i:i+4].to(DEV)[:, :-1].contiguous(), Dm).reshape(-1, Dm))
        ABL['layers'] = set(); R = torch.cat(R, 0)
        return float((R - Rc).pow(2).sum(1).mean())
    singles = [deltasq({L}) for L in front]
    band = deltasq(set(front))
    for h in hooks: h.remove()
    s = float(sum(singles))
    return {'n_front': len(front), 'sum_singles_dR2': round(s, 2), 'band_dR2': round(band, 2), 'ratio': round(band/max(s, 1e-9), 2)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    out = {'metric': 'final-residual dR2 super-additivity (band / sum-of-singles)', 'models': {}}
    r = measure(BILIN, blocks, 1152, 18); out['models']['bilin18'] = r; print(f"bilin18: {r}", flush=True)
    for short, nl, dm in [('swiglu18', 18, 1152), ('bilin12', 12, 768)]:
        try:
            mdl, cfg = load_elriggs(short); dm = cfg.get('n_embd', dm); nl = cfg.get('n_layer', nl)
            r = measure(mdl, blocks, dm, nl); out['models'][short] = r; print(f"{short}: {r}", flush=True)
            del mdl; torch.cuda.empty_cache()
        except Exception as e:
            out['models'][short] = {'error': f'{type(e).__name__}: {e}'}; print(f"{short} ERROR {e}", flush=True)
    ratios = {k: v.get('ratio') for k, v in out['models'].items() if 'ratio' in v}
    out['ratios'] = ratios
    sw = out['models'].get('swiglu18', {}).get('ratio')
    out['swiglu18_also_superadditive'] = bool(sw is not None and sw > 1.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"ratios {ratios}", flush=True)
    print(f"(interp) swiglu18 also super-additive (distributed = family-driven): {out['swiglu18_also_superadditive']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
