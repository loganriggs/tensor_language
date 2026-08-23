"""GENERATIVE input-side validation of the bag-of-words content mechanism (§932/§967/§995). All prior content
causality was ACTIVATION/weight-level (topic interchange §894). Here test it from the INPUT: inject a strongly
topical CONTENT word W at an EARLY position of a context and measure whether the model's prediction at a DISTANT query
shifts toward W's TOPIC -- specifically the mean log-prob change of W's TOPIC-NEIGHBORS (tokens whose unembedding rows
are cosine-near W's), EXCLUDING W itself and any context tokens (so a pure copy/induction of W does NOT count). If the
content machine is a broad bag-of-words, injecting a topical word raises its topic-neighbors downstream; injecting a
FUNCTION word (no topic) should not.

REGISTERED PREDICTIONS:
  (0) NULL/specificity: injecting W raises W's OWN topic-neighbors MORE than the neighbors of a RANDOM unrelated word
      (the shift is topic-specific, not a generic confidence bump).
  (a) TOPICAL PRIMING (bag-of-words, input side): injecting a topical CONTENT word W at an early position raises the
      mean log-prob of W's topic-neighbors (excl W, excl context) at a distant query, and MUCH more than injecting a
      FUNCTION word -> generative confirmation of bag-of-words content priming;
  (b) LONG-RANGE: the effect is present with injection near the start (pos ~3) and query far away (~150).
  Report: mean neighbor-Δlogprob for content-word injection, function-word injection, and the random-word null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_injection_priming_results.json'
NEVAL = 200; SEQ = 256; INJ_POS = 3; QUERY = 150; NNEIGH = 20
CONTENT_WORDS = [' football', ' hospital', ' ocean', ' music', ' science', ' army', ' church', ' garden', ' computer', ' medicine']
FUNCTION_WORDS = [' the', ' of', ' and', ' to', ' in', ' for']


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def neighbors(wid, k, exclude):
    W = m.lm_head.weight.float()  # (V, D)
    wv = W[wid] / (W[wid].norm() + 1e-9)
    sims = (W / (W.norm(dim=1, keepdim=True) + 1e-9)) @ wv
    sims[wid] = -1e9
    for e in exclude: sims[e] = -1e9
    return torch.topk(sims, k).indices


@torch.no_grad()
def mean_neighbor_dlp(blocks, wid, neigh):
    # inject wid at INJ_POS; measure mean Δlog-prob of `neigh` at the query (predicting token QUERY), vs no injection
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV)
        base_idx = bb[:, :QUERY].contiguous()
        inj_idx = base_idx.clone(); inj_idx[:, INJ_POS] = wid
        lp_base = F.log_softmax(forward_logits(base_idx).float()[:, -1], -1)
        lp_inj = F.log_softmax(forward_logits(inj_idx).float()[:, -1], -1)
        # exclude contexts where wid already appears (avoid copy/existing-topic confound)
        has = (base_idx == wid).any(1)
        dlp = (lp_inj[:, neigh] - lp_base[:, neigh]).mean(1)  # per-row mean over neighbors
        dlp = dlp[~has]
        tot += float(dlp.sum()); n += int((~has).sum())
    return tot / max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    out = {'content': {}, 'function': {}, 'null_random': {}}
    rng = np.random.RandomState(0)
    # content words
    cvals = []
    for w in CONTENT_WORDS:
        wid = tid(w)
        if wid is None: continue
        neigh = neighbors(wid, NNEIGH, exclude=[])
        dlp = mean_neighbor_dlp(rows, wid, neigh)
        out['content'][w] = round(dlp, 4); cvals.append(dlp)
        # null: same injection, but measure the neighbors of a RANDOM unrelated word
        rwid = int(rng.randint(0, m.lm_head.weight.shape[0]))
        rneigh = neighbors(rwid, NNEIGH, exclude=[])
        out['null_random'][w] = round(mean_neighbor_dlp(rows, wid, rneigh), 4)
        print(f"content {w!r}: own-neighbor Δlp {out['content'][w]} | random-neighbor Δlp {out['null_random'][w]}", flush=True)
    fvals = []
    for w in FUNCTION_WORDS:
        wid = tid(w)
        if wid is None: continue
        neigh = neighbors(wid, NNEIGH, exclude=[])
        dlp = mean_neighbor_dlp(rows, wid, neigh)
        out['function'][w] = round(dlp, 4); fvals.append(dlp)
        print(f"function {w!r}: own-neighbor Δlp {out['function'][w]}", flush=True)
    out['content_mean'] = round(float(np.mean(cvals)), 4)
    out['function_mean'] = round(float(np.mean(fvals)), 4)
    out['null_random_mean'] = round(float(np.mean(list(out['null_random'].values()))), 4)
    out['pred_0_specific'] = bool(out['content_mean'] > out['null_random_mean'] + 0.05)
    out['pred_a_topical_priming'] = bool(out['content_mean'] > 0.05 and out['content_mean'] > 2*max(out['function_mean'], 1e-6))
    out['inj_pos'] = INJ_POS; out['query'] = QUERY; out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"content-inject own-neighbor Δlp {out['content_mean']} | function {out['function_mean']} | random-null {out['null_random_mean']}", flush=True)
    print(f"pred_0 topic-specific {out['pred_0_specific']} | pred_a topical-priming (content>>function) {out['pred_a_topical_priming']}", flush=True)
    print(f"(inject pos {INJ_POS} -> query {QUERY}, long-range) wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
