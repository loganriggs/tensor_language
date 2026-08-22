"""IS bilin18's "distributed super-additive circuit" theme SPECIAL or just generic residual-network compounding?
§948/§953/§954 found single components near-free but ensembles load-bearing (super-additive). But some
super-additivity is generic (ablating more layers compounds). Test honestly: for bilin18, GPT-2, GPT-2-large,
measure the ATTENTION super-additivity RATIO = (mean-ablate a whole third of the attention layers together) /
(sum of the individual single-layer mean-ablate costs), for the FRONT third. If bilin18's ratio is much HIGHER
than the GPT-2 models', its circuits are unusually distributed/redundant (special); if comparable, the
super-additivity is a generic property (honest deflation of the "distributed-cooperative" framing).

REGISTERED PREDICTIONS:
  (0) SANITY: every model's band-together cost > its sum-of-singles (ratio > 1) — some compounding is generic.
  (a) report the ratio per model and state plainly whether bilin18 is UNUSUAL (ratio >> GPT-2's) or COMPARABLE
      (theme is generic). No directional prediction forced — this is a calibration/deflation check;
  (b) report per-model: front sum-of-singles, front-band-together, ratio."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m as BILIN, DEV
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_model_superadditivity_results.json'
NEVAL = 120; SEQ = 256
ABL = {'layers': set(), 'means': None, 'mod': None}


def bilin_forward(idx):
    x = F.rms_norm(BILIN.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in BILIN.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(BILIN.lm_head(F.rms_norm(x, (D,)))/30.0)


def make_hook(L, Dm):
    def h(mo, i_, o_):
        if L not in ABL['layers']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        yn = ABL['means'][L].view(1, 1, Dm).expand(B, T, Dm).clone()
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


@torch.no_grad()
def ce_pass(run_fwd, blocks, tgtfn):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = tgtfn(run_fwd(idx)).float()
        tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel()
    return tot/n


@torch.no_grad()
def run_model(attn_mods, run_fwd, tgtfn, blocks, Dm, nlayer):
    front = list(range(0, nlayer//3))
    sums = {L: torch.zeros(Dm, device=DEV) for L in front}; cnt = 0; hs = []
    for L in front:
        def mk(L):
            def h(mo, i_, o_): sums[L] += (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, Dm).sum(0)
            return h
        hs.append(attn_mods[L].register_forward_hook(mk(L)))
    for i in range(0, blocks.shape[0], 8): run_fwd(blocks[i:i+8].to(DEV)[:, :-1].contiguous()); cnt += min(8, blocks.shape[0]-i)*(SEQ-1)
    for h in hs: h.remove()
    ABL['means'] = {L: sums[L]/cnt for L in front}
    hooks = [attn_mods[L].register_forward_hook(make_hook(L, Dm)) for L in front]
    ABL['layers'] = set(); ce_full = ce_pass(run_fwd, blocks, tgtfn)
    singles = []
    for L in front:
        ABL['layers'] = {L}; singles.append(ce_pass(run_fwd, blocks, tgtfn) - ce_full)
    ABL['layers'] = set(front); band = ce_pass(run_fwd, blocks, tgtfn) - ce_full
    ABL['layers'] = set()
    for h in hooks: h.remove()
    s = float(sum(max(x, 0) for x in singles))
    return {'n_front': len(front), 'sum_singles': round(s, 4), 'band_together': round(band, 4),
            'ratio': round(band/max(s, 1e-6), 2)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    out = {'models': {}}
    r = run_model([blk.attn for blk in BILIN.transformer.h], bilin_forward, lambda x: x, blocks, D, 18)
    out['models']['bilin18'] = r; print(f"bilin18: {r}", flush=True)
    for mid in ['gpt2', 'gpt2-large']:
        mdl = AutoModelForCausalLM.from_pretrained(mid).to(DEV).eval(); Dm = mdl.config.n_embd; nl = mdl.config.n_layer
        r = run_model([blk.attn for blk in mdl.transformer.h], lambda idx: mdl(idx).logits if hasattr(mdl(idx), 'logits') else mdl(idx), lambda x: x, blocks, Dm, nl)
        out['models'][mid] = r; print(f"{mid}: {r}", flush=True)
        del mdl; torch.cuda.empty_cache()
    ratios = {k: out['models'][k]['ratio'] for k in out['models']}
    out['ratios'] = ratios
    out['bilin18_unusual'] = bool(ratios['bilin18'] > 1.5*np.mean([ratios['gpt2'], ratios['gpt2-large']]))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"super-additivity ratios: {ratios}", flush=True)
    print(f"bilin18 unusually distributed (ratio >> GPT-2): {out['bilin18_unusual']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
