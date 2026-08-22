"""COMPLETE the 2x2 causal isolation of the distributed-cooperative structure: {squared / softmax attention} x
{bilinear / standard MLP}. §969 gave two cells (bilin12 = squared+bilinear 1.61; bilinsm12 = softmax+bilinear
1.13). Add sqrd12 = 'gpt2-sqrd-attn-12l-6h-768embd' (squared attention + STANDARD MLP, per repo name) to test
whether SQUARED ATTENTION ALONE (without the bilinear MLP) produces the super-additive distributed structure.
Measure its super-additivity ratio (final-residual dR2 method) and report cfg flags.

  cell                         attn      MLP        ratio (§969 / this)
  bilin12    squared+bilinear  sqrd      bilinear   1.61
  bilinsm12  softmax+bilinear  softmax   bilinear   1.13
  sqrd12     squared+standard  sqrd      standard   ??? (this run)

REGISTERED PREDICTIONS:
  (0) SANITY: report sqrd12 cfg flags (expect squared_attn=True, bilinear=False).
  (a) SQUARED ATTENTION ALONE SUFFICES: sqrd12 (squared attn, standard MLP) is SUPER-additive (> ~1.5, like
      bilin12), confirming squared attention causes the distributed structure independent of the bilinear MLP; if
      sqrd12 is near-additive (~1.1, like the softmax cell), the bilinear MLP is also required. No forced
      direction — the 2x2 reads the interaction; cfg flags anchor it."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
import census_lib as cl
from bilin18_joint_removal import DEV
from tier2_model import load_elriggs
import torch.nn.functional as F

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_mlp_2x2_superadditivity_results.json'
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
    return {'ratio': round(band/max(s, 1e-9), 2), 'sum_singles': round(s, 2), 'band': round(band, 2)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    out = {'ref_969': {'bilin12(sqrd+bilinear)': 1.61, 'bilinsm12(softmax+bilinear)': 1.13}, 'models': {}}
    for short in ['sqrd12']:
        try:
            mdl, cfg = load_elriggs(short); Dm = cfg.get('n_embd'); nl = cfg.get('n_layer')
            flags = {k: cfg.get(k) for k in cfg if 'attn' in k.lower() or 'soft' in k.lower() or 'sqr' in k.lower() or 'bilinear' in k.lower()}
            r = measure(mdl, blocks, Dm, nl); r['cfg_flags'] = flags
            out['models'][short] = r; print(f"{short}: ratio {r['ratio']} | flags {flags}", flush=True)
            del mdl; torch.cuda.empty_cache()
        except Exception as e:
            out['models'][short] = {'error': f'{type(e).__name__}: {e}'}; print(f"{short} ERROR {e}", flush=True)
    sq = out['models'].get('sqrd12', {}).get('ratio')
    out['squared_attn_alone_suffices'] = bool(sq is not None and sq > 1.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"2x2: bilin12(sqrd+bilin) 1.61 | bilinsm12(soft+bilin) 1.13 | sqrd12(sqrd+std) {sq}", flush=True)
    print(f"(a) squared attention alone suffices (sqrd12 super-additive): {out['squared_attn_alone_suffices']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
