"""WHERE DOES EVERY NAT OF THE LOSS GO? (capstone synthesis of the whole program). Decompose bilin18's mean
cross-entropy by POSITION TYPE crossed with the GRAMMAR/CONTENT chain-rule split, so the total loss is
attributed to named mechanisms. Each next-token position is exactly one of:
  - INDUCTABLE: its (current,next) bigram already occurred in the context -> served by induction/copy (attn5, §877).
  - FIRST-MENTION: next-token type unseen in the context -> the hard, largely irreducible floor (§876).
  - SEEN-OTHER: next-token type seen but bigram not repeated -> topic/frequency territory.
Within each bucket, split CE into GRAMMAR (predict the next class) and CONTENT (within-class word choice) by
the chain rule (§829). Report each bucket's fraction of positions, mean CE, grammar/content split, and its
CONTRIBUTION to the total loss (fraction x CE). This is the quantitative budget the mechanism story predicts.
Controls: unigram-frequency baseline CE (the starting entropy); buckets partition all positions (fractions
sum to 1, contributions sum to the total).

REGISTERED PREDICTIONS:
  (0) SANITY: fraction-weighted bucket CEs sum to the overall CE; unigram baseline >> model CE;
  (a) THE LOSS LIVES IN FIRST-MENTION CONTENT: the first-mention bucket dominates the total-loss contribution,
      and within every bucket CONTENT (within-class) >> GRAMMAR (class); induction (inductable bucket) is a
      large fraction of positions but a SMALL loss contribution (cheap) -> the budget is: grammar cheap
      everywhere, induction cheap, and the bulk is first-mention content (topic-constrained word choice);
  (b) report the full table regardless."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'loss_budget_results.json'
NEVAL = 240; SEQ = 256
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}


def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])


def classify(s):
    t = s.strip()
    if t == '' or not t[0].isalnum(): return 'punct'
    if t[0].isdigit(): return 'number'
    low = t.lower()
    if low in DET: return 'det'
    if low in PREP: return 'prep'
    if low in CONJ: return 'conj'
    if low in PRON: return 'pron'
    if t[0].isupper(): return 'cap'
    return 'word'


def bilin_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    V = int(m.lm_head.weight.shape[0])
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(S.reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, len(CLASSES)).float()
    # unigram baseline entropy on the eval tokens
    cnt = np.bincount(S.reshape(-1), minlength=V).astype(np.float64); p = cnt/cnt.sum()
    tgt_all = S[:, 1:].reshape(-1); base_ce = float(-np.log(p[tgt_all] + 1e-12).mean())
    # per-position class-CE / within-CE
    cl_ce = []; wi_ce = []
    for i in range(0, nb, 4):
        bb = blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(bilin_logits(idx).float(), -1); pcl = lp.exp() @ Cmat
        tf = tgt.reshape(-1); lpf = lp.reshape(-1, V); tcl = cidx[tf]
        lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
        lp_cls = (pcl.reshape(-1, len(CLASSES))[torch.arange(tf.shape[0], device=DEV), tcl] + 1e-12).log()
        cl_ce.append((-lp_cls).cpu().numpy()); wi_ce.append((-(lp_tok - lp_cls)).cpu().numpy())
    cl_ce = np.concatenate(cl_ce).reshape(nb, SEQ-1); wi_ce = np.concatenate(wi_ce).reshape(nb, SEQ-1)
    # bucket masks
    inductable = np.zeros((nb, SEQ-1), dtype=bool); firstment = np.zeros((nb, SEQ-1), dtype=bool)
    for r in range(nb):
        seen_tok = set(); seen_big = {}
        for pp in range(SEQ-1):
            cur = int(S[r, pp]); nxt = int(S[r, pp+1])
            firstment[r, pp] = nxt not in seen_tok
            if cur in seen_big and seen_big[cur] == nxt: inductable[r, pp] = True
            seen_big[cur] = nxt; seen_tok.add(cur)
    inductable = inductable.reshape(-1); firstment = firstment.reshape(-1) & ~inductable
    other = ~inductable & ~firstment
    cl_f = cl_ce.reshape(-1); wi_f = wi_ce.reshape(-1); tot_f = cl_f + wi_f; N = len(tot_f)
    out = {'overall_ce': round(float(tot_f.mean()), 3), 'unigram_baseline_ce': round(base_ce, 3),
           'overall_class_ce': round(float(cl_f.mean()), 3), 'overall_within_ce': round(float(wi_f.mean()), 3),
           'buckets': {}}
    for name, mk in [('inductable', inductable), ('first_mention', firstment), ('seen_other', other)]:
        frac = float(mk.mean()); ce = float(tot_f[mk].mean()); contrib = frac*ce
        out['buckets'][name] = {'fraction': round(frac, 3), 'mean_ce': round(ce, 3),
                                'class_ce': round(float(cl_f[mk].mean()), 3), 'within_ce': round(float(wi_f[mk].mean()), 3),
                                'loss_contribution': round(contrib, 3), 'pct_of_total_loss': round(100*contrib/tot_f.mean(), 1)}
    fm = out['buckets']['first_mention']
    out['pred_a_loss_is_firstmention_content'] = bool(
        fm['pct_of_total_loss'] > 50 and fm['within_ce'] > fm['class_ce'] and
        out['buckets']['inductable']['pct_of_total_loss'] < out['buckets']['first_mention']['pct_of_total_loss'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"overall CE {out['overall_ce']} (unigram baseline {out['unigram_baseline_ce']}) = class {out['overall_class_ce']} + within {out['overall_within_ce']}", flush=True)
    for name in ['inductable', 'first_mention', 'seen_other']:
        b = out['buckets'][name]
        print(f"  {name:>13}: {b['fraction']*100:4.1f}% of pos | CE {b['mean_ce']:.2f} (class {b['class_ce']:.2f} + within {b['within_ce']:.2f}) | {b['pct_of_total_loss']:4.1f}% of total loss", flush=True)
    print(f"(a) loss lives in first-mention content: {out['pred_a_loss_is_firstmention_content']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
