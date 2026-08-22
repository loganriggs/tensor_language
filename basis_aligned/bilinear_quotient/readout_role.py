"""WHAT DO THE FINAL (near-LINEAR, §941/§942 R^2 0.89/0.93) READOUT LAYERS L16-17 DO? Complete the bottom-up
stack (front grammar / middle content / readout). Two probes:
 (1) LOGIT LENS: apply the model's readout head to the residual at each late layer L (13..17) and measure the
     next-token CE. Shows how much of the final prediction is already formed by each layer.
 (2) SKIP vs MEAN-ABLATE L16 and L17: replace the block output with its INPUT (identity skip = the block adds
     nothing) vs the mean (mean-ablate), and measure the CE cost. If skipping costs far less than mean-ablating,
     the block is a small refinement of an already-formed prediction, not a load-bearing transform.

REGISTERED PREDICTIONS:
  (0) SANITY: logit-lens CE decreases with depth toward the final CE; final-layer lens == full CE.
  (a) READOUT = LATE LINEAR REFINEMENT: logit-lens CE at L15 is already close to the final CE (a modest gap), and
      L16-17 each provide a SMALL refinement; SKIPPING L16-17 (identity) costs LESS than mean-ablating them ->
      the last blocks are near-linear refinements of an already-formed prediction, consistent with their high
      linear-R^2 (§941);
  (b) report per-late-layer logit-lens CE + skip-cost and mean-ablate-cost for L16 and L17."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'readout_role_results.json'
NEVAL = 200; SEQ = 256; NLAYER = 18
LENS_LAYERS = [12, 13, 14, 15, 16, 17]
MOD = {'L': -1, 'mode': 'off', 'mean': None}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def block_hook(L):
    def h(mo, i_, o_):
        if MOD['L'] != L or MOD['mode'] == 'off': return o_
        x = i_[0] if isinstance(i_, tuple) else i_  # block input
        y = o_[0] if isinstance(o_, tuple) else o_
        if MOD['mode'] == 'skip':
            yn = x  # identity: block adds nothing
        else:
            yn = MOD['mean'].expand_as(y).clone()
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


@torch.no_grad()
def forward_all_resids(idx):
    """return list of residual after each block (b,T,D) and final logits."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None; res = []
    for blk in m.transformer.h:
        x, v1 = blk(x, v1, x0); res.append(x)
    return res, readout(x)


@torch.no_grad()
def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    # logit lens per late layer + block-output means for mean-ablate
    lens_tot = {L: 0.0 for L in LENS_LAYERS}; n = 0
    blkout_sum = {L: torch.zeros(D, device=DEV) for L in [16, 17]}; cnt = 0
    ce_full = 0.0
    for i in range(0, nb, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        res, lg = forward_all_resids(idx)
        ce_full += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum'))
        for L in LENS_LAYERS:
            ll = readout(res[L]).float()
            lens_tot[L] += float(F.cross_entropy(ll.reshape(-1, ll.shape[-1]), tgt.reshape(-1), reduction='sum'))
        for L in [16, 17]: blkout_sum[L] += res[L].reshape(-1, D).sum(0)
        n += tgt.numel(); cnt += idx.shape[0]*(SEQ-1)
    ce_full /= n
    out = {'ce_full': round(ce_full, 4), 'logit_lens_ce': {str(L): round(lens_tot[L]/n, 4) for L in LENS_LAYERS}}
    print(f"ce_full {ce_full:.4f}", flush=True)
    for L in LENS_LAYERS: print(f"  logit-lens L{L}: CE {out['logit_lens_ce'][str(L)]:.4f}", flush=True)
    means = {L: (blkout_sum[L]/cnt).view(1, 1, D) for L in [16, 17]}
    # skip vs mean-ablate for L16, L17
    hooks = [m.transformer.h[L].register_forward_hook(block_hook(L)) for L in [16, 17]]
    def ce_pass():
        tot = 0.0; nn = 0
        for i in range(0, nb, 8):
            bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
            lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); nn += tgt.numel()
        return tot/nn
    out['skip_cost'] = {}; out['meanablate_cost'] = {}
    for L in [16, 17]:
        MOD['mean'] = means[L]
        MOD['L'] = L; MOD['mode'] = 'skip'; ce_sk = ce_pass()
        MOD['mode'] = 'mean'; ce_mn = ce_pass(); MOD['L'] = -1; MOD['mode'] = 'off'
        out['skip_cost'][str(L)] = round(ce_sk - ce_full, 4); out['meanablate_cost'][str(L)] = round(ce_mn - ce_full, 4)
        print(f"L{L}: skip Δce {out['skip_cost'][str(L)]:+.4f} | mean-ablate Δce {out['meanablate_cost'][str(L)]:+.4f}", flush=True)
    for h in hooks: h.remove()
    gap15 = out['logit_lens_ce']['15'] - ce_full
    out['lens_L15_gap'] = round(gap15, 4)
    out['pred_a_late_refinement'] = bool(gap15 < 1.0 and all(out['skip_cost'][str(L)] < out['meanablate_cost'][str(L)] for L in [16, 17]))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"L15 logit-lens gap to final {gap15:+.4f}", flush=True)
    print(f"(a) readout = late linear refinement (L15 close, skip<meanablate): {out['pred_a_late_refinement']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
