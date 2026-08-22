"""CAUSAL ISOLATION of the distributed-cooperative structure (§965): is it the softmax-FREE (squared) ATTENTION
specifically, or the bilinear MLP / family broadly? Compare a MATCHED PAIR that differs (per repo name) only in
attention type: bilin12 = 'gpt2-bilinear-sqrd-attn-12l-6h-768embd' (squared attention) vs bilinsm12 =
'gpt2-bilinear-12l-6h-768embd' (standard/softmax attention) — both bilinear MLP, both 12L/6h/768. Measure the
attention super-additivity ratio (final-residual dR2, band / sum-of-singles, §965) for each, and REPORT each
model's config attention flags so the interpretation is grounded in the actual architecture.

REGISTERED PREDICTIONS:
  (0) SANITY: bilin12 reproduces its super-additive ratio (~1.6, §965); report cfg flags for both.
  (a) SQRD-ATTENTION IS THE CAUSE: if bilinsm12 (softmax attention) is SUB-additive (ratio < ~1) while bilin12
      (squared attention) is SUPER-additive (> 1.5), the distributed-cooperative structure is caused by the
      softmax-FREE attention specifically (matched bilinear MLP + size) -> sharpens §965; if both super-additive,
      it is broader than attention type. No forced direction (calibration); the cfg flags anchor the reading."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
import census_lib as cl
from bilin18_joint_removal import DEV
from tier2_model import load_elriggs
import torch.nn.functional as F

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_variant_superadditivity_results.json'
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
    return x


@torch.no_grad()
def measure(mdl, blocks, Dm, nlayer):
    front = list(range(0, nlayer//3))
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
    ABL['layers'] = set(); Rc = []
    for i in range(0, blocks.shape[0], 4): Rc.append(final_resid(mdl, blocks[i:i+4].to(DEV)[:, :-1].contiguous(), Dm).reshape(-1, Dm))
    Rc = torch.cat(Rc, 0)
    def deltasq(active):
        ABL['layers'] = active; R = []
        for i in range(0, blocks.shape[0], 4): R.append(final_resid(mdl, blocks[i:i+4].to(DEV)[:, :-1].contiguous(), Dm).reshape(-1, Dm))
        ABL['layers'] = set(); return float((torch.cat(R, 0) - Rc).pow(2).sum(1).mean())
    singles = [deltasq({L}) for L in front]; band = deltasq(set(front))
    for h in hooks: h.remove()
    s = float(sum(singles))
    return {'n_front': len(front), 'sum_singles': round(s, 2), 'band': round(band, 2), 'ratio': round(band/max(s, 1e-9), 2)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    out = {'models': {}}
    for short in ['bilin12', 'bilinsm12']:
        try:
            mdl, cfg = load_elriggs(short); Dm = cfg.get('n_embd'); nl = cfg.get('n_layer')
            flags = {k: cfg.get(k) for k in cfg if 'attn' in k.lower() or 'soft' in k.lower() or 'sqr' in k.lower() or 'bilinear' in k.lower()}
            r = measure(mdl, blocks, Dm, nl); r['cfg_flags'] = flags
            out['models'][short] = r; print(f"{short}: ratio {r['ratio']} | flags {flags}", flush=True)
            del mdl; torch.cuda.empty_cache()
        except Exception as e:
            out['models'][short] = {'error': f'{type(e).__name__}: {e}'}; print(f"{short} ERROR {e}", flush=True)
    b = out['models'].get('bilin12', {}).get('ratio'); bsm = out['models'].get('bilinsm12', {}).get('ratio')
    out['sqrd_attn_causes_distributed'] = bool(b is not None and bsm is not None and b > 1.5 and bsm < 1.2)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"bilin12(sqrd) ratio {b} vs bilinsm12(softmax?) ratio {bsm}", flush=True)
    print(f"(a) squared attention causes distributed structure: {out['sqrd_attn_causes_distributed']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
