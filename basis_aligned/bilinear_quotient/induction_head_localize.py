"""Is induction localized to a few HEADS (classic induction heads), even though it is layer-DISTRIBUTED (§953)?
Resolve §953's caveat (it ablated whole layers = 9 heads). Ablate each individual HEAD by mean-ablating its
128-dim slice of the attention c_proj INPUT (the concatenated per-head value-aggregates, D=9*128), and measure the
CE cost on INDUCTABLE positions. Sweep heads in the front-to-mid layers (L0-8, where §952/§953 put induction).

REGISTERED PREDICTIONS:
  (0) SANITY: most heads cost ~0 on inductable; the summed head costs are in the ballpark of the layer costs (§953).
  (a) A FEW INDUCTION HEADS: a small number of heads carry most of the inductable-ablation cost (top-5 heads >
      50% of the total positive cost) -> classic induction heads exist at HEAD granularity even though induction
      is layer-distributed (the localization is at the head, not the layer, level);
  (b) if head costs are also flat/distributed (top-5 < 50%), induction is distributed even at head level (report)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; HEAD = 128; NH = 9; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'induction_head_localize_results.json'
NEVAL = 140; SEQ = 256; LAYERS = list(range(0, 9))
ABL = {'L': -1, 'h': -1, 'means': None}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def cproj_prehook(L):
    def h(mo, args):
        if ABL['L'] != L: return args
        x = args[0]; x = x.clone(); s = ABL['h']*HEAD
        x[:, :, s:s+HEAD] = ABL['means'][(L, ABL['h'])].view(1, 1, HEAD)
        return (x,) + tuple(args[1:])
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
        outs.append((-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf]).cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    inductable = np.zeros((nb, SEQ-1), bool)
    for r in range(nb):
        big = {}
        for p in range(SEQ-1):
            cur = int(S[r, p]); nx = int(S[r, p+1])
            if cur in big and big[cur] == nx: inductable[r, p] = True
            big[cur] = nx
    inductable = inductable.reshape(-1)
    # per-head means of c_proj input
    sums = {(L, h): torch.zeros(HEAD, device=DEV) for L in LAYERS for h in range(NH)}; cnt = 0; hs = []
    for L in LAYERS:
        def mk(L):
            def hook(mo, args):
                x = args[0].detach().float().reshape(-1, D)
                for h in range(NH): sums[(L, h)] += x[:, h*HEAD:(h+1)*HEAD].sum(0)
                return None
            return hook
        hs.append(m.transformer.h[L].attn.c_proj.register_forward_pre_hook(mk(L)))
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous()); cnt += min(8, nb-i)*(SEQ-1)
    for h in hs: h.remove()
    ABL['means'] = {k: v/cnt for k, v in sums.items()}
    hooks = [m.transformer.h[L].attn.c_proj.register_forward_pre_hook(cproj_prehook(L)) for L in LAYERS]
    ABL['L'] = -1; base = per_pos_ce(blocks); base_ind = float(base[inductable].mean())
    out = {'baseline_inductable': round(base_ind, 4), 'head_cost': {}}
    for L in LAYERS:
        for h in range(NH):
            ABL['L'] = L; ABL['h'] = h; w = per_pos_ce(blocks); ABL['L'] = -1
            out['head_cost'][f"L{L}h{h}"] = round(float((w[inductable]-base[inductable]).mean()), 4)
    for hh in hooks: hh.remove()
    costs = out['head_cost']; ranked = sorted(costs, key=costs.get, reverse=True)
    postot = sum(max(v, 0) for v in costs.values())
    top5 = ranked[:5]; out['top5_heads'] = {k: costs[k] for k in top5}
    out['top5_share'] = round(sum(max(costs[k], 0) for k in top5)/max(postot, 1e-6), 3)
    out['sum_positive_head_cost'] = round(postot, 3)
    out['pred_a_few_induction_heads'] = bool(out['top5_share'] > 0.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top-5 induction heads: {out['top5_heads']}", flush=True)
    print(f"top-5 share {out['top5_share']} of total positive {postot:.2f} (baseline inductable {base_ind:.3f})", flush=True)
    print(f"(a) a few induction heads carry it: {out['pred_a_few_induction_heads']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
