"""CAPACITY of the content bag: §1022 showed TWO topics superpose additively with no dilution. How many topics can the
bag hold before they compete? Inject N different-topic words (N=1,2,3,4,6) at early positions and measure each
injected topic's neighbor-boost, as a fraction of its SOLO boost (retention). A finite-capacity pool should stay
additive (retention ~1) for small N and then DILUTE (retention < 1) as N grows and topics compete for the
representation. This maps the bag's capacity -- connects to §930 (content is a high-rank continuum).

REGISTERED PREDICTIONS:
  (0) SANITY: at N=1 retention = 1 by definition; each solo boost > 0.
  (a) FINITE CAPACITY: mean per-topic retention stays near 1 for small N (additive, §1022) and DECLINES as N grows
      (topics dilute each other as the bag fills) -> the content pool has finite capacity; report retention vs N and
      whether retention(6) < retention(2);
  (b) if retention stays ~1 even at N=6, the bag has high capacity within this range (report that honestly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_injection_capacity_results.json'
NEVAL = 200; SEQ = 256; QUERY = 150; NNEIGH = 20
POSITIONS = [3, 12, 24, 36, 48, 60]  # up to 6 injection slots (all < QUERY)
WORDS = [' football', ' ocean', ' hospital', ' music', ' science', ' church']  # 6 distinct topics
NS = [1, 2, 3, 4, 6]


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
def boost_multi(blocks, inj_wids, inj_pos, target_neigh, target_wids):
    # inject inj_wids at inj_pos; measure mean Δlp of target_neigh at query, excluding rows where any target appears
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); base_idx = bb[:, :QUERY].contiguous(); inj_idx = base_idx.clone()
        for w, p in zip(inj_wids, inj_pos): inj_idx[:, p] = w
        lb = F.log_softmax(forward_logits(base_idx).float()[:, -1], -1)
        li = F.log_softmax(forward_logits(inj_idx).float()[:, -1], -1)
        has = torch.zeros(base_idx.shape[0], dtype=torch.bool, device=DEV)
        for w in target_wids: has |= (base_idx == w).any(1)
        dlp = (li[:, target_neigh] - lb[:, target_neigh]).mean(1)[~has]
        tot += float(dlp.sum()); n += int((~has).sum())
    return tot / max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    wids = [tid(w) for w in WORDS]; neigh = [neighbors(wid, NNEIGH) for wid in wids]
    # solo boost per word (inject only that word at POSITIONS[0])
    solo = [boost_multi(rows, [wids[j]], [POSITIONS[0]], neigh[j], [wids[j]]) for j in range(len(WORDS))]
    print(f"solo boosts: {[round(s,3) for s in solo]}", flush=True)
    out = {'solo': [round(s, 4) for s in solo], 'by_N': {}}
    for N in NS:
        inj_wids = wids[:N]; inj_pos = POSITIONS[:N]
        rets = []
        for j in range(N):  # measure each injected topic's retention
            b = boost_multi(rows, inj_wids, inj_pos, neigh[j], [wids[j]])
            rets.append(b / max(solo[j], 1e-6))
        out['by_N'][str(N)] = {'mean_retention': round(float(np.mean(rets)), 3), 'mean_boost': round(float(np.mean([solo[j]*rets[j] for j in range(N)])), 3)}
        print(f"N={N}: mean per-topic retention {out['by_N'][str(N)]['mean_retention']} (mean boost {out['by_N'][str(N)]['mean_boost']})", flush=True)
    r2 = out['by_N']['2']['mean_retention']; r6 = out['by_N']['6']['mean_retention']
    out['retention_2'] = r2; out['retention_6'] = r6
    out['pred_a_finite_capacity'] = bool(r6 < r2 - 0.15)
    out['pred_b_high_capacity'] = bool(r6 > 0.8)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"retention: N=2 {r2} vs N=6 {r6}", flush=True)
    print(f"pred_a finite-capacity(dilutes) {out['pred_a_finite_capacity']} | pred_b high-capacity(stays additive) {out['pred_b_high_capacity']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
