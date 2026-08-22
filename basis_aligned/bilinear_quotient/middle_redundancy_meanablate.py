"""REDUNDANCY of the middle content computation, done CLEANLY (fixing §947's skip-confound). MEAN-ABLATE the
OUTPUT of an increasing number of middle MLPs simultaneously (replace each with its global mean — this keeps the
block's residual-rescale x=λ0·x+λ1·x0 intact, unlike block-skip §947) and measure Δce. Compare to mean-ablating
front MLPs. If mean-ablating a few middle MLPs costs little and grows sub-linearly, the content computation is
redundant/distributed (as §940/§941/§946 suggested per-layer).

REGISTERED PREDICTIONS:
  (0) SANITY: mean-ablating 0 MLPs == full CE; a single middle MLP costs ~0.01-0.05 (matches §941/§946).
  (a) REDUNDANT MIDDLE: mean-ablating k middle MLPs (L6-15) grows SUB-LINEARLY / stays modest for small k
      (redundant), and the FRONT MLPs (esp mlp0/mlp1) are far LESS dispensable per layer (grammar/token write is
      localized, §915/§933) -> middle content computation is distributed; front is localized;
  (b) report Δce vs #MLPs-mean-ablated for middle (L6-15, latest-first) and front (L0-5), + the per-layer-sum
      comparison (super-additive = distributed-but-not-fully-redundant)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_redundancy_meanablate_results.json'
NEVAL = 200; SEQ = 256
ABL = {'layers': set(), 'means': None}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def mlp_meanabl_hook(L):
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
    # global mean of each MLP output
    sums = {L: torch.zeros(D, device=DEV) for L in range(18)}; cnt = 0; hs = []
    for L in range(18):
        def mk(L):
            def h(mo, i_, o_): sums[L] += (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D).sum(0)
            return h
        hs.append(m.transformer.h[L].mlp.register_forward_hook(mk(L)))
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); forward_logits(idx); cnt += idx.shape[0]*(SEQ-1)
    for h in hs: h.remove()
    ABL['means'] = {L: sums[L]/cnt for L in range(18)}
    hooks = [m.transformer.h[L].mlp.register_forward_hook(mlp_meanabl_hook(L)) for L in range(18)]
    ABL['layers'] = set(); ce_full = ce_pass(blocks)
    out = {'ce_full': round(ce_full, 4), 'middle_meanablate': {}, 'front_meanablate': {}, 'per_layer_middle': {}}
    MIDDLE = list(range(15, 5, -1)); FRONT = list(range(5, -1, -1))
    for k in [0, 1, 2, 3, 4, 6, 8, 10]:
        ABL['layers'] = set(MIDDLE[:k]); ce = ce_pass(blocks)
        out['middle_meanablate'][str(k)] = round(ce - ce_full, 4)
        print(f"mean-ablate {k} middle MLPs: Δce {ce-ce_full:+.4f}", flush=True)
    for k in [1, 2, 3, 4, 6]:
        ABL['layers'] = set(FRONT[:k]); ce = ce_pass(blocks)
        out['front_meanablate'][str(k)] = round(ce - ce_full, 4)
        print(f"mean-ablate {k} front MLPs: Δce {ce-ce_full:+.4f}", flush=True)
    # per-layer middle (for super-additivity check)
    persum = 0.0
    for L in range(6, 16):
        ABL['layers'] = {L}; ce = ce_pass(blocks); c = round(ce - ce_full, 4)
        out['per_layer_middle'][str(L)] = c; persum += c
    for h in hooks: h.remove()
    out['per_layer_middle_sum'] = round(persum, 4); out['all10_middle'] = out['middle_meanablate']['10']
    out['superadditive_ratio'] = round(out['middle_meanablate']['10']/max(persum, 1e-6), 2)
    m4 = out['middle_meanablate']['4']; f4 = out['front_meanablate']['4']
    out['pred_a_redundant_middle'] = bool(m4 < f4)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"middle all-10 {out['all10_middle']} vs per-layer-sum {persum:.3f} (super-additive {out['superadditive_ratio']}x)", flush=True)
    print(f"skip-4: middle {m4} vs front {f4}", flush=True)
    print(f"(a) middle more redundant than front (meanablate-4 middle < front): {out['pred_a_redundant_middle']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
