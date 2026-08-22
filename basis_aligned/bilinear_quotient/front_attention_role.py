"""WHY is front attention so critical (§951: front-attn mean-ablate 3.66, the largest band cost)? Name its role by
splitting the CE cost of mean-ablating the FRONT attention (L0-5) by POSITION TYPE:
  - inductable: current token seen earlier with the SAME next token (bigram-inductable -> induction/copy),
  - first_mention: the next token is NEW (not seen -> genuine content prediction),
  - seen_other: everything else.
If front-attn ablation hurts INDUCTABLE positions most, front attention is doing induction/copying; if
FIRST-MENTION most, it is content-gathering; if roughly UNIFORM across types, it is general positional/sequence
mixing. Compare to mean-ablating ALL attention (whole context engine) and a random 6-layer attention band.

REGISTERED PREDICTIONS:
  (0) SANITY: baseline within-type CE ordering first_mention > seen_other > inductable (§879); random band lands
      below the front band.
  (a) FRONT ATTN IS BROAD CONTEXT-MIXING with an INDUCTION component: mean-ablating front attention hurts ALL
      position types substantially (it is the general early context substrate), AND hurts inductable positions
      MORE than a random attention band does (early induction/copy lives in front attention);
  (b) report front-attn-ablate CE cost per position type vs all-attn and random-band."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'front_attention_role_results.json'
NEVAL = 200; SEQ = 256
ABL = {'layers': set(), 'means': None}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def attn_hook(L):
    def h(mo, i_, o_):
        if L not in ABL['layers']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        yn = ABL['means'][L].view(1, 1, D).expand(B, T, D).clone()
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def per_pos_ce(blocks):
    outs = []
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1)
        lpf = lp.reshape(-1, lp.shape[-1])
        outs.append((-lpf[torch.arange(tf.shape[0], device=DEV), tf]).cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    # position-type masks (per next-token target)
    inductable = np.zeros((nb, SEQ-1), bool); firstment = np.zeros((nb, SEQ-1), bool)
    for r in range(nb):
        seen = set(); big = {}
        for p in range(SEQ-1):
            cur = int(S[r, p]); nx = int(S[r, p+1]); firstment[r, p] = nx not in seen
            if cur in big and big[cur] == nx: inductable[r, p] = True
            big[cur] = nx; seen.add(cur)
    inductable = inductable.reshape(-1); firstment = firstment.reshape(-1) & ~inductable; other = ~inductable & ~firstment
    masks = {'inductable': inductable, 'first_mention': firstment, 'seen_other': other}
    # attn means
    sums = {L: torch.zeros(D, device=DEV) for L in range(18)}; cnt = 0; hs = []
    for L in range(18):
        def mk(L):
            def h(mo, i_, o_): sums[L] += (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D).sum(0)
            return h
        hs.append(m.transformer.h[L].attn.register_forward_hook(mk(L)))
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous()); cnt += min(8, nb-i)*(SEQ-1)
    for h in hs: h.remove()
    ABL['means'] = {L: sums[L]/cnt for L in range(18)}
    hooks = [m.transformer.h[L].attn.register_forward_hook(attn_hook(L)) for L in range(18)]
    ABL['layers'] = set(); base = per_pos_ce(blocks)
    def by_type(w): return {k: round(float(w[msk].mean()), 4) for k, msk in masks.items()}
    out = {'baseline_by_type': by_type(base), 'conditions': {}}
    conds = {'front_attn_L0_5': list(range(0, 6)), 'all_attn': list(range(18)),
             'random_attn_6': [1, 4, 7, 10, 13, 16]}
    for name, layers in conds.items():
        ABL['layers'] = set(layers); w = per_pos_ce(blocks); ABL['layers'] = set()
        delta = w - base
        out['conditions'][name] = {k: round(float(delta[msk].mean()), 4) for k, msk in masks.items()}
        print(f"{name:>16} Δce by type: {out['conditions'][name]}", flush=True)
    for h in hooks: h.remove()
    fa = out['conditions']['front_attn_L0_5']; rb = out['conditions']['random_attn_6']
    out['front_induction_excess_vs_random'] = round(fa['inductable'] - rb['inductable'], 4)
    out['pred_a_front_broad_plus_induction'] = bool(min(fa.values()) > 0.3 and fa['inductable'] > rb['inductable'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"baseline by type: {out['baseline_by_type']}", flush=True)
    print(f"front-attn induction excess vs random-band: {out['front_induction_excess_vs_random']:+.4f}", flush=True)
    print(f"(a) front attn = broad context-mixing + induction: {out['pred_a_front_broad_plus_induction']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
