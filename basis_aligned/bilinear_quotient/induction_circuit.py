"""CONFIRM the induction CIRCUIT is the two-step attn0(prev-token) -> attn5(copy) composition, with a control
(sharpens §877, which only scanned single-layer ablations). The standard induction circuit is a prev-token
head writing the match key, feeding a copy head. §877 found single-ablation drops at L0 and L5 (both front);
this tests the composition directly: ablate attn0 alone, attn5 alone, BOTH, and a late CONTROL attention
layer, on the synthetic induction task ([P][P] random tokens).

REGISTERED PREDICTIONS:
  (0) SANITY: no-ablation induction score large (~§877's 11.8); random-token second copy near-perfectly copied;
  (a) TWO-STEP FRONT CIRCUIT: ablating attn0 OR attn5 each removes most induction; ablating BOTH removes ~all
      of it (>= the max of the singles, ~ total), while ablating a LATE control layer (attn13) barely dents it
      -> induction is the attn0->attn5 front composition, not spread over the whole stack;
  (b) if the late control layer also collapses induction, induction is not front-specific (report plainly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'induction_circuit_results.json'
NSYN = 48; L = 64; CONTROL_L = 13
ABL = {'layers': set()}


def ablate_hook_factory(Li):
    def hook(mo, i_, o_):
        if Li not in ABL['layers']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; z = torch.zeros_like(y)
        return (z,) + tuple(o_[1:]) if isinstance(o_, tuple) else z
    return hook


def bilin_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def induction_score(seqs):
    idx = seqs[:, :-1].contiguous(); tgt = seqs[:, 1:].contiguous()
    lp = F.log_softmax(bilin_logits(idx).float(), -1)
    l = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float(l[:, :L-1].mean()), float(l[:, L:2*L-1].mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    g = torch.Generator(device=DEV).manual_seed(0)
    base = torch.randint(0, 50000, (NSYN, L), generator=g, device=DEV)
    seqs = torch.cat([base, base], 1)
    hs = [m.transformer.h[Li].attn.register_forward_hook(ablate_hook_factory(Li)) for Li in [0, 5, CONTROL_L]]
    def score(layers):
        ABL['layers'] = set(layers); f, s = induction_score(seqs); ABL['layers'] = set()
        return round(f - s, 3), round(s, 3)
    conds = {'none': [], 'attn0': [0], 'attn5': [5], 'attn0+attn5': [0, 5], f'control_attn{CONTROL_L}': [CONTROL_L]}
    res = {}
    for name, layers in conds.items():
        ind, second = score(layers); res[name] = {'induction_score': ind, 'second_copy_loss': second}
        print(f"{name:>16}: induction score {ind} | second-copy loss {second}", flush=True)
    for h in hs: h.remove()
    base_ind = res['none']['induction_score']
    out = {'conditions': res, 'base_induction': base_ind,
           'drop_attn0': round(base_ind - res['attn0']['induction_score'], 3),
           'drop_attn5': round(base_ind - res['attn5']['induction_score'], 3),
           'drop_both': round(base_ind - res['attn0+attn5']['induction_score'], 3),
           'drop_control': round(base_ind - res[f'control_attn{CONTROL_L}']['induction_score'], 3),
           'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_two_step_front_circuit'] = bool(
        out['drop_attn0'] > 2 and out['drop_attn5'] > 2 and
        out['drop_both'] >= max(out['drop_attn0'], out['drop_attn5']) - 0.5 and
        out['drop_control'] < 0.5 * min(out['drop_attn0'], out['drop_attn5']))
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\ndrops: attn0 {out['drop_attn0']} | attn5 {out['drop_attn5']} | both {out['drop_both']} | control(attn{CONTROL_L}) {out['drop_control']}", flush=True)
    print(f"(a) two-step front circuit (attn0->attn5), control spared: {out['pred_a_two_step_front_circuit']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
