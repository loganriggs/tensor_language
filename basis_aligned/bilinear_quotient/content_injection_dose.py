"""DOSE-RESPONSE test of the bag-of-words POOLING/accumulation (extends §1016's single-word priming). A "bag of words"
ACCUMULATES: more topical evidence -> stronger topic. Inject K copies of a topical content word W at early positions
and measure the topic-neighbor log-prob boost (W's unembedding-neighbors, EXCLUDING W and context) at a distant query,
for K = 0,1,2,4,8. If content is a broad bag-of-words pool, the boost should GROW with K (accumulation), not saturate
at K=1. Function-word dose is the control (should stay ~flat).

REGISTERED PREDICTIONS:
  (0) NULL: K=0 boost ~0; a FUNCTION-word dose stays ~flat across K (no topic to accumulate).
  (a) POOLING/ACCUMULATION: the topic-neighbor boost INCREASES monotonically with K for content words (K=8 > K=4 >
      K=2 > K=1), confirming the bag ACCUMULATES topical evidence (broad pooling, §995), rather than a single word
      saturating the effect;
  (b) report the boost vs K for content words and function words + whether it is monotonic and its K=8/K=1 ratio."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_injection_dose_results.json'
NEVAL = 200; SEQ = 256; QUERY = 150; NNEIGH = 20; KS = [0, 1, 2, 4, 8]
INJ_POSITIONS = [3, 12, 24, 36, 48, 60, 72, 84]   # up to 8 early positions (all < QUERY)
CONTENT_WORDS = [' football', ' hospital', ' ocean', ' music', ' science', ' army', ' church', ' garden']
FUNCTION_WORDS = [' the', ' of', ' and', ' to']


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
def dose_boost(blocks, wid, neigh, K):
    # inject K copies of wid at the first K INJ_POSITIONS; boost = mean Δlp(neigh) at query vs no injection
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); base_idx = bb[:, :QUERY].contiguous()
        inj_idx = base_idx.clone()
        for p in INJ_POSITIONS[:K]: inj_idx[:, p] = wid
        lp_base = F.log_softmax(forward_logits(base_idx).float()[:, -1], -1)
        lp_inj = F.log_softmax(forward_logits(inj_idx).float()[:, -1], -1)
        has = (base_idx == wid).any(1)
        dlp = (lp_inj[:, neigh] - lp_base[:, neigh]).mean(1)[~has]
        tot += float(dlp.sum()); n += int((~has).sum())
    return tot / max(n, 1)


@torch.no_grad()
def sweep(blocks, words, tid):
    perK = {str(K): [] for K in KS}
    for w in words:
        wid = tid(w)
        if wid is None: continue
        neigh = neighbors(wid, NNEIGH)
        for K in KS: perK[str(K)].append(dose_boost(blocks, wid, neigh, K))
    return {K: round(float(np.mean(perK[K])), 4) for K in perK}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    out = {'content_by_K': sweep(rows, CONTENT_WORDS, tid), 'function_by_K': sweep(rows, FUNCTION_WORDS, tid)}
    print(f"content by K: {out['content_by_K']}", flush=True)
    print(f"function by K: {out['function_by_K']}", flush=True)
    c = out['content_by_K']
    cvals = [c[str(K)] for K in KS]
    out['content_monotonic'] = bool(all(cvals[i+1] >= cvals[i] - 0.01 for i in range(len(cvals)-1)))
    out['k8_over_k1_ratio'] = round(c['8']/max(c['1'], 1e-6), 2)
    fvals = [out['function_by_K'][str(K)] for K in KS]
    out['function_flat'] = bool(max(fvals) - min(fvals) < 0.1)
    out['pred_0_null_ok'] = bool(abs(c['0']) < 0.05 and out['function_flat'])
    out['pred_a_pooling_accumulates'] = bool(out['content_monotonic'] and c['8'] > c['1'] + 0.1)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"content K8/K1 ratio {out['k8_over_k1_ratio']} monotonic {out['content_monotonic']} | function flat {out['function_flat']}", flush=True)
    print(f"pred_0 null {out['pred_0_null_ok']} | pred_a pooling-accumulates {out['pred_a_pooling_accumulates']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
