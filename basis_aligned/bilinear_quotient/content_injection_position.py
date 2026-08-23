"""GENERATIVE receptive-field profile of the content pool (completes the injection modality §1016-1018; cross-validates
§995 banding + §981-983 recency from the INPUT side). Inject a single topical content word W at varying DISTANCES
before the query (pos 150) and measure the topic-neighbor log-prob boost. This maps how the content-priming strength
depends on where the word sits -- the input-side analog of the attention receptive field.

REGISTERED PREDICTIONS:
  (0) SANITY: boost > 0 at all distances (a topical word anywhere in the pooled context primes its topic).
  (a) RECENCY-WEIGHTED BUT BROAD: the content boost RISES as the injection approaches the query (recency weighting,
      consistent with §981-983 recency routing) yet remains SUBSTANTIAL even at the largest distance (~147 tokens
      back) -> broad, long-range pooling with recency emphasis (matches §995: content unsaturated/broad). Report the
      boost at each distance + the near/far ratio;
  (b) two independent instruments (attention banding §995 and input injection here) should agree that content is
      long-range: the far-distance boost is a large fraction of the near boost, not ~0."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_injection_position_results.json'
NEVAL = 200; SEQ = 256; QUERY = 150; NNEIGH = 20
POSITIONS = [3, 50, 90, 120, 140, 148]   # injection positions (distance from query = QUERY - pos)
CONTENT_WORDS = [' football', ' hospital', ' ocean', ' music', ' science', ' army', ' church', ' garden']


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def neighbors(wid, k):
    W = m.lm_head.weight.float(); wv = W[wid] / (W[wid].norm() + 1e-9)
    sims = (W / (W.norm(dim=1, keepdim=True) + 1e-9)) @ wv; sims[wid] = -1e9
    return torch.topk(sims, k).indices


@torch.no_grad()
def boost_at(blocks, wid, neigh, pos):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); base_idx = bb[:, :QUERY].contiguous()
        inj_idx = base_idx.clone(); inj_idx[:, pos] = wid
        lb = F.log_softmax(forward_logits(base_idx).float()[:, -1], -1)
        li = F.log_softmax(forward_logits(inj_idx).float()[:, -1], -1)
        has = (base_idx == wid).any(1)
        dlp = (li[:, neigh] - lb[:, neigh]).mean(1)[~has]
        tot += float(dlp.sum()); n += int((~has).sum())
    return tot / max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    perpos = {str(p): [] for p in POSITIONS}
    for w in CONTENT_WORDS:
        wid = tid(w)
        if wid is None: continue
        neigh = neighbors(wid, NNEIGH)
        for p in POSITIONS: perpos[str(p)].append(boost_at(rows, wid, neigh, p))
    out = {'by_position': {str(p): round(float(np.mean(perpos[str(p)])), 4) for p in POSITIONS},
           'distance_from_query': {str(p): QUERY - p for p in POSITIONS}}
    far = out['by_position'][str(POSITIONS[0])]; near = out['by_position'][str(POSITIONS[-1])]
    out['far_boost'] = far; out['near_boost'] = near; out['far_over_near'] = round(far/max(near, 1e-6), 3)
    out['pred_0_all_positive'] = bool(all(v > 0.05 for v in out['by_position'].values()))
    out['pred_a_recency_but_broad'] = bool(near > far and far > 0.15)  # rises toward query, yet far still substantial
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"boost by injection position: {out['by_position']}", flush=True)
    print(f"far (dist {QUERY-POSITIONS[0]}) {far} | near (dist {QUERY-POSITIONS[-1]}) {near} | far/near {out['far_over_near']}", flush=True)
    print(f"pred_0 all-positive {out['pred_0_all_positive']} | pred_a recency-but-broad {out['pred_a_recency_but_broad']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
