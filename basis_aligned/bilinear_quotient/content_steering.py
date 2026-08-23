"""CAPSTONE (understanding -> control): the content machine's log-probs shift under topical injection (§1016-1024), but
does that translate into BEHAVIORAL control -- can we change what the model actually PREDICTS (its top-k), not just the
log-prob of neighbors? Inject K copies of a topic word early and measure the fraction of eval positions where a
topic-neighbor of the injected word enters the model's TOP-5 predictions at a distant query, vs baseline. If
understanding the content machine gives control, injection should substantially raise that fraction.

REGISTERED PREDICTIONS:
  (0) SANITY/NULL: with no injection, a topic-neighbor is in the top-5 at baseline rate; a FUNCTION-word injection
      barely changes it (no topic to steer).
  (a) BEHAVIORAL STEERING: injecting K=8 copies of a topic word raises the fraction of positions where one of the
      injected topic's neighbors is in the TOP-5 prediction, substantially above baseline (delta > ~0.1) and rising
      with K -> understanding the content machine lets us steer the model's actual output, not just its log-probs;
  (b) report top-5 topic-hit fraction vs K for content words and the function-word control."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_steering_results.json'
NEVAL = 200; SEQ = 256; QUERY = 150; NNEIGH = 20; TOPK = 5; KS = [0, 1, 4, 8]
POSITIONS = [3, 12, 24, 36, 48, 60, 72, 84]
CONTENT_WORDS = [' football', ' hospital', ' ocean', ' music', ' science', ' army', ' church', ' garden']
FUNCTION_WORDS = [' the', ' of', ' and', ' to']


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def neighbors(wid, k):
    Wm = m.lm_head.weight.float(); wv = Wm[wid] / (Wm[wid].norm() + 1e-9)
    sims = (Wm / (Wm.norm(dim=1, keepdim=True) + 1e-9)) @ wv; sims[wid] = -1e9
    return set(torch.topk(sims, k).indices.tolist())


@torch.no_grad()
def top5_hit_fraction(blocks, wid, neigh_set, K):
    hits = 0; n = 0
    neigh_t = torch.tensor(sorted(neigh_set), device=DEV)
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); idx = bb[:, :QUERY].contiguous().clone()
        for p in POSITIONS[:K]: idx[:, p] = wid
        top5 = forward_logits(idx).float()[:, -1].topk(TOPK, -1).indices  # (B, 5)
        has = (bb[:, :QUERY] == wid).any(1)
        # is any neighbor in the top-5?
        hit = (top5.unsqueeze(-1) == neigh_t.view(1, 1, -1)).any(-1).any(-1)  # (B,)
        hits += int(hit[~has].sum()); n += int((~has).sum())
    return hits / max(n, 1)


@torch.no_grad()
def sweep(blocks, words):
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    byK = {str(K): [] for K in KS}
    for w in words:
        wid = tid(w)
        if wid is None: continue
        ns = neighbors(wid, NNEIGH)
        for K in KS: byK[str(K)].append(top5_hit_fraction(blocks, wid, ns, K))
    return {K: round(float(np.mean(byK[K])), 4) for K in byK}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    out = {'content_top5_hit_by_K': sweep(rows, CONTENT_WORDS), 'function_top5_hit_by_K': sweep(rows, FUNCTION_WORDS)}
    c = out['content_top5_hit_by_K']; f = out['function_top5_hit_by_K']
    out['content_delta_K8'] = round(c['8'] - c['0'], 4); out['function_delta_K8'] = round(f['8'] - f['0'], 4)
    out['pred_0_function_control'] = bool(out['function_delta_K8'] < 0.5 * out['content_delta_K8'])
    out['pred_a_behavioral_steering'] = bool(out['content_delta_K8'] > 0.1 and c['8'] > c['4'] > c['0'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"content top-5 topic-hit fraction by K: {c}", flush=True)
    print(f"function top-5 topic-hit fraction by K: {f}", flush=True)
    print(f"content ΔK8 {out['content_delta_K8']} | function ΔK8 {out['function_delta_K8']}", flush=True)
    print(f"pred_0 function-control {out['pred_0_function_control']} | pred_a behavioral-steering {out['pred_a_behavioral_steering']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
