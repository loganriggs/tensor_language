"""DOES the L5 induction head cause the seen-token discount on NATURAL text? (causal tie between §877
induction and §876 seen/first-mention split). §877: L5 attention gates induction (synthetic). §876:
already-seen tokens are cheap, first-mentions dear. If L5's induction IS the seen-token discount, then
ablating L5 attention on FineWeb should raise the loss on INDUCTABLE / already-seen positions far more than
on FIRST-MENTION positions (which have nothing to copy). Compare to ablating a CONTROL attention layer
(L11, a middle topic-aggregator, not the induction gate).

Method: on FineWeb, split next-token positions into INDUCTABLE (its (current,next) bigram occurred earlier)
vs FIRST-MENTION (next-token type unseen in context) vs OTHER. Measure per-bucket CE for: full model, L5-attn
ablated, L11-attn ablated (control). Report the CE increase per bucket.

REGISTERED PREDICTIONS:
  (0) SANITY: full-model inductable CE << first-mention CE (§876/§877 reproduce);
  (a) L5 = the copy discount: ablating L5 attention raises INDUCTABLE CE far more than FIRST-MENTION CE
      (inductable increase / first-mention increase > 2) -> induction serves copying seen tokens, not novel
      prediction; and L5's inductable-increase >> the control layer L11's inductable-increase;
  (b) if L5 ablation hurts first-mentions as much as inductable, L5 is not specifically the copy mechanism
      (report honestly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'induction_ablate_natural_results.json'
NEVAL = 200; SEQ = 256; ABLATE_L = 5; CONTROL_L = 11
ABL = {'layer': -1}


def ablate_hook_factory(Lget):
    def hook(mo, i_, o_):
        if ABL['layer'] != Lget: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; z = torch.zeros_like(y)
        return (z,) + tuple(o_[1:]) if isinstance(o_, tuple) else z
    return hook


def bilin_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def bucket_losses(blocks, hooks):
    """returns per-position loss (flat) + bucket masks, for whatever ABL['layer'] is set."""
    Ls = []
    for i in range(0, blocks.shape[0], 4):
        bb = blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(bilin_logits(idx).float(), -1)
        Ls.append((-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).cpu())
    return torch.cat(Ls, 0).numpy().reshape(-1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    # bucket masks (flat over (nb, SEQ-1))
    inductable = np.zeros((nb, SEQ-1), dtype=bool); firstment = np.zeros((nb, SEQ-1), dtype=bool)
    for r in range(nb):
        seen_tok = set(); seen_big = {}
        for p in range(SEQ-1):
            cur = int(S[r, p]); nxt = int(S[r, p+1])
            firstment[r, p] = nxt not in seen_tok
            if cur in seen_big and seen_big[cur] == nxt: inductable[r, p] = True
            seen_big[cur] = nxt; seen_tok.add(cur)
    inductable = inductable.reshape(-1); firstment = firstment.reshape(-1)
    other = ~inductable & ~firstment
    # register ablation hooks on both layers (gated by ABL['layer'])
    hs = [m.transformer.h[ABLATE_L].attn.register_forward_hook(ablate_hook_factory(ABLATE_L)),
          m.transformer.h[CONTROL_L].attn.register_forward_hook(ablate_hook_factory(CONTROL_L))]
    def stats(L):
        ABL['layer'] = L; loss = bucket_losses(blocks, hs)
        return {'inductable': round(float(loss[inductable].mean()), 3),
                'first_mention': round(float(loss[firstment].mean()), 3),
                'other': round(float(loss[other].mean()), 3), 'overall': round(float(loss.mean()), 3)}
    full = stats(-1); abl5 = stats(ABLATE_L); abl11 = stats(CONTROL_L)
    for h in hs: h.remove()
    def inc(a, b, k): return round(a[k]-b[k], 3)
    out = {'ablate_layer': ABLATE_L, 'control_layer': CONTROL_L,
           'n_inductable': int(inductable.sum()), 'n_first_mention': int(firstment.sum()), 'n_other': int(other.sum()),
           'full': full, 'ablate_L5': abl5, 'ablate_L11_control': abl11,
           'L5_increase': {'inductable': inc(abl5, full, 'inductable'), 'first_mention': inc(abl5, full, 'first_mention'), 'other': inc(abl5, full, 'other')},
           'L11_increase': {'inductable': inc(abl11, full, 'inductable'), 'first_mention': inc(abl11, full, 'first_mention'), 'other': inc(abl11, full, 'other')},
           'runtime_s': round(time.time()-t0, 1)}
    r5 = out['L5_increase']['inductable']/max(out['L5_increase']['first_mention'], 1e-6)
    out['L5_inductable_over_firstmention'] = round(r5, 2)
    out['pred_a_L5_is_copy_discount'] = bool(r5 > 2 and out['L5_increase']['inductable'] > out['L11_increase']['inductable'])
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"full: inductable {full['inductable']} | first-mention {full['first_mention']} | other {full['other']}", flush=True)
    print(f"ablate L5: inductable {abl5['inductable']} | first-mention {abl5['first_mention']} | other {abl5['other']}", flush=True)
    print(f"ablate L11 (control): inductable {abl11['inductable']} | first-mention {abl11['first_mention']}", flush=True)
    print(f"L5 CE increase: inductable +{out['L5_increase']['inductable']} vs first-mention +{out['L5_increase']['first_mention']} (ratio {r5:.2f})", flush=True)
    print(f"L11 CE increase: inductable +{out['L11_increase']['inductable']} vs first-mention +{out['L11_increase']['first_mention']}", flush=True)
    print(f"(a) L5 is the copy discount: {out['pred_a_L5_is_copy_discount']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
