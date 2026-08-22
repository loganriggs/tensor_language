"""HOW REDUNDANT is the middle content computation? §940/§941/§946 found each middle MLP/component has a TINY
individual ablation cost (~0.01-0.05 nats) yet the middle band as a whole is the biggest contributor (§940) —
i.e. the content computation is distributed/redundant across many layers. Quantify it: SKIP (replace whole block
output with its input = identity, so the block adds nothing) an increasing number of MIDDLE blocks (from L6..L15)
and measure the CE cost. Skip from the LATEST middle layer backward (L15, then L15+L14, ...). This says how many
middle layers are dispensable before content collapses.

REGISTERED PREDICTIONS:
  (0) SANITY: skipping 0 blocks == full CE; skipping ALL middle blocks (L6-15) costs a large amount (content
      needs the middle); a random-position skip-set of the same size is a control.
  (a) REDUNDANT MIDDLE: skipping the first FEW middle blocks costs little (sublinear) -> the content computation
      is redundant/distributed; the cost accelerates as more are removed. Report the CE-vs-#skipped curve;
  (b) compare skipping k middle blocks vs k FRONT blocks (L0-5) — front should be far less dispensable (grammar
      write is localized, §915)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_redundancy_results.json'
NEVAL = 200; SEQ = 256
SKIP = {'layers': set()}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def block_skip_hook(L):
    def h(mo, i_, o_):
        if L not in SKIP['layers']: return o_
        x = i_[0] if isinstance(i_, tuple) else i_  # block input (identity: block adds nothing)
        return (x,) + tuple(o_[1:]) if isinstance(o_, tuple) else x
    return h


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def ce_pass(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel()
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    hooks = [m.transformer.h[L].register_forward_hook(block_skip_hook(L)) for L in range(18)]
    SKIP['layers'] = set(); ce_full = ce_pass(blocks)
    out = {'ce_full': round(ce_full, 4), 'middle_skip': {}, 'front_skip': {}, 'random_skip': {}}
    MIDDLE = list(range(15, 5, -1))   # L15 down to L6 (skip latest-first)
    FRONT = list(range(5, -1, -1))    # L5 down to L0
    for k in [0, 1, 2, 3, 4, 6, 8, 10]:
        SKIP['layers'] = set(MIDDLE[:k]); ce = ce_pass(blocks)
        out['middle_skip'][str(k)] = round(ce - ce_full, 4)
        print(f"skip {k} middle blocks {sorted(SKIP['layers'])}: Δce {ce-ce_full:+.4f}", flush=True)
    for k in [1, 2, 3, 4, 6]:
        SKIP['layers'] = set(FRONT[:k]); ce = ce_pass(blocks)
        out['front_skip'][str(k)] = round(ce - ce_full, 4)
        print(f"skip {k} front blocks {sorted(SKIP['layers'])}: Δce {ce-ce_full:+.4f}", flush=True)
    # random-position control (size 6), a few seeds
    rng = np.random.RandomState(0); rand_costs = []
    for s in range(3):
        sel = set(rng.choice(18, 6, replace=False).tolist()); SKIP['layers'] = sel; ce = ce_pass(blocks)
        rand_costs.append(round(ce - ce_full, 4))
    out['random_skip']['size6_seeds'] = rand_costs
    for h in hooks: h.remove()
    m6 = out['middle_skip'].get('6'); f6 = out['front_skip'].get('6')
    out['skip6_middle_vs_front'] = {'middle': m6, 'front': f6}
    out['pred_a_redundant_middle'] = bool(out['middle_skip']['4'] < 0.5 * out['front_skip'].get('4', 1e9) if out['front_skip'].get('4') else False)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"skip-6 middle Δce {m6} vs front Δce {f6} | random-6 {rand_costs}", flush=True)
    print(f"(a) middle more redundant than front (skip-4 middle << skip-4 front): {out['pred_a_redundant_middle']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
