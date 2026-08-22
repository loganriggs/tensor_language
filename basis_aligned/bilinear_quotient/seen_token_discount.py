"""WHAT makes SEEN tokens cheaper than first-mentions? (characterizes the seen-other bucket — the last
under-explained 20% of the loss, §879/§880). Exact-bigram repeats are handled by induction (§877), but the
seen-other bucket (token type seen before, bigram NOT repeated) is still far cheaper than first-mention (1.94
vs 4.24). Is that a SOFT-COPY / salience mechanism — the model raises a word's probability because it already
appeared, more so when the prior occurrence is RECENT and FREQUENT — and is it attention-carried (L5)?

Method: for target tokens whose type appeared earlier in the context but NOT as a repeated (current,next)
bigram, bucket by RECENCY (distance to last prior occurrence) and COUNT (number of prior occurrences), and
measure mean loss in each cell. Baselines: first-mention (no prior) and inductable (exact bigram). Causal:
re-measure with L5 attention ablated -> does the recency discount shrink (soft copy is attention/L5-carried)?

REGISTERED PREDICTIONS:
  (0) SANITY: seen-token loss < first-mention loss; inductable cheapest;
  (a) SOFT-COPY SALIENCE: seen-token loss DECREASES with more prior occurrences AND with closer recency
      (recent/frequent priors -> cheaper) -> the seen-token discount is a soft-copy/salience effect, not just
      flat topic membership;
  (b) ATTENTION-CARRIED: ablating L5 attention raises seen-token loss MORE for recent/frequent priors than
      for far/rare ones (flattens the recency-count gradient) -> the soft copy runs through the same
      content/induction head;
  (c) if loss is flat across recency/count, the seen discount is topic/frequency, not copy (report plainly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'seen_token_discount_results.json'
NEVAL = 220; SEQ = 256; ABLATE_L = 5
REC_BINS = [(1, 8), (9, 32), (33, 96), (97, 255)]
CNT_BINS = [(1, 1), (2, 2), (3, 5), (6, 9999)]
ABL = {'on': False}


def ablate_hook(mo, i_, o_):
    if not ABL['on']: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; z = torch.zeros_like(y)
    return (z,) + tuple(o_[1:]) if isinstance(o_, tuple) else z


def bilin_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def per_pos_loss(blocks):
    out = []
    for i in range(0, blocks.shape[0], 4):
        bb = blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(bilin_logits(idx).float(), -1)
        out.append((-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).cpu().numpy())
    return np.concatenate(out, 0)   # (nb, SEQ-1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    # per-position: recency of last prior occurrence of TARGET, count of prior occurrences, inductable flag, first-mention
    recency = np.full((nb, SEQ-1), -1, dtype=np.int64); count = np.zeros((nb, SEQ-1), dtype=np.int64)
    inductable = np.zeros((nb, SEQ-1), dtype=bool)
    for r in range(nb):
        last = {}; cnt = {}; seen_big = {}
        for pp in range(SEQ-1):
            cur = int(S[r, pp]); nxt = int(S[r, pp+1])
            if nxt in last:
                recency[r, pp] = pp+1 - last[nxt]   # distance from target position to last occurrence of that token
                count[r, pp] = cnt.get(nxt, 0)
            if cur in seen_big and seen_big[cur] == nxt: inductable[r, pp] = True
            seen_big[cur] = nxt
            last[nxt] = pp+1; cnt[nxt] = cnt.get(nxt, 0) + 1  # register the target token occurrence at pp+1
    recency = recency.reshape(-1); count = count.reshape(-1); inductable = inductable.reshape(-1)
    first = (recency < 0)
    seen_other = (recency > 0) & ~inductable

    def grid(loss):
        g = {}
        for (rlo, rhi) in REC_BINS:
            for (clo, chi) in CNT_BINS:
                mk = seen_other & (recency >= rlo) & (recency <= rhi) & (count >= clo) & (count <= chi)
                if mk.sum() >= 20: g[f"rec{rlo}-{rhi}_cnt{clo}-{chi}"] = {'loss': round(float(loss[mk].mean()), 3), 'n': int(mk.sum())}
        return g

    L0 = per_pos_loss(blocks)
    hh = m.transformer.h[ABLATE_L].attn.register_forward_hook(ablate_hook)
    ABL['on'] = True; L1 = per_pos_loss(blocks); ABL['on'] = False; hh.remove()
    out = {'baseline': {'first_mention': round(float(L0[first].mean()), 3),
                        'inductable': round(float(L0[inductable].mean()), 3),
                        'seen_other': round(float(L0[seen_other].mean()), 3)},
           'seen_grid_full': grid(L0), 'seen_grid_L5ablated': grid(L1),
           # marginal effects
           'by_recency': {f"{lo}-{hi}": round(float(L0[seen_other & (recency >= lo) & (recency <= hi)].mean()), 3)
                          for (lo, hi) in REC_BINS if (seen_other & (recency >= lo) & (recency <= hi)).sum() >= 20},
           'by_count': {f"{lo}-{hi}": round(float(L0[seen_other & (count >= lo) & (count <= hi)].mean()), 3)
                        for (lo, hi) in CNT_BINS if (seen_other & (count >= lo) & (count <= hi)).sum() >= 20},
           'L5_ablation_increase_seen': round(float(L1[seen_other].mean() - L0[seen_other].mean()), 3),
           'L5_ablation_increase_recent': round(float(L1[seen_other & (recency <= 32)].mean() - L0[seen_other & (recency <= 32)].mean()), 3),
           'L5_ablation_increase_far': round(float(L1[seen_other & (recency >= 97)].mean() - L0[seen_other & (recency >= 97)].mean()), 3),
           'runtime_s': round(time.time()-t0, 1)}
    rec_vals = list(out['by_recency'].values()); cnt_vals = list(out['by_count'].values())
    out['pred_a_softcopy'] = bool(len(rec_vals) >= 2 and rec_vals[0] < rec_vals[-1] - 0.2 and len(cnt_vals) >= 2 and cnt_vals[-1] < cnt_vals[0] - 0.2)
    out['pred_b_attention_carried'] = bool(out['L5_ablation_increase_recent'] > out['L5_ablation_increase_far'])
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"baseline: first-mention {out['baseline']['first_mention']} | seen-other {out['baseline']['seen_other']} | inductable {out['baseline']['inductable']}", flush=True)
    print(f"seen loss by RECENCY (near->far): {out['by_recency']}", flush=True)
    print(f"seen loss by COUNT (rare->frequent): {out['by_count']}", flush=True)
    print(f"L5 ablation raises seen loss: recent +{out['L5_ablation_increase_recent']} vs far +{out['L5_ablation_increase_far']} (all seen +{out['L5_ablation_increase_seen']})", flush=True)
    print(f"(a) soft-copy salience: {out['pred_a_softcopy']} | (b) attention-carried: {out['pred_b_attention_carried']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
