"""TEST the §972 hedging interpretation directly (not assert it): are GRAMMAR-ERROR positions (top-1 class != true
class) cases where the model HEDGES to a frequent FUNCTION word because the true CONTENT word is unpredictable —
rather than genuine grammar failures? For each error type (HIT / CONTENT-ERROR / GRAMMAR-ERROR) measure:
  (i)   log-frequency of the TRUE next token (rarer = harder content),
  (ii)  fraction of the PREDICTED top-1 tokens that are FUNCTION-class (det/prep/conj/pron/punct),
  (iii) the model's probability RANK of the true token (does it still rank the true token reasonably = hedged, or
        catastrophically = failed).

REGISTERED PREDICTIONS (hedging account):
  (a) on GRAMMAR-ERROR positions the TRUE token is RARER than on hit/content-error positions, the PREDICTED top-1
      is FUNCTION-class far more often than the corpus base rate, AND the true token still gets a modest median
      rank (not catastrophic) -> the model hedges to a common function word on hard content positions rather than
      failing at grammar;
  (b) report per-error-type: mean log-freq of true token, function-class frac of top-1, median true-token rank."""
import json, time, sys, torch
import numpy as np
from collections import Counter
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'error_hedging_test_results.json'
NEVAL = 200; SEQ = 256
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
FUNCTION = {'det', 'prep', 'conj', 'pron', 'punct'}
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}


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


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    V = int(m.lm_head.weight.shape[0])
    tok2cls = np.full(V, 7, np.int64)
    uniq = np.unique(blocks.cpu().numpy().reshape(-1))
    for tid in uniq: tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    is_func = np.array([CLASSES[c] in FUNCTION for c in tok2cls], bool)
    cidx = torch.tensor(tok2cls, device=DEV)
    # empirical corpus frequency (log) of each token from the targets
    tgt_all = blocks.cpu().numpy()[:, 1:].reshape(-1); ct = Counter(tgt_all); Ntot = len(tgt_all)
    logfreq = np.log(np.array([ct.get(int(t), 1) for t in range(V)]) / Ntot + 1e-12)
    agg = {k: {'logfreq_true': [], 'top1_is_func': [], 'true_rank': []} for k in ['hit', 'content_error', 'grammar_error']}
    for i in range(0, nb, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); tf = tgt.reshape(-1); lgf = lg.reshape(-1, lg.shape[-1])
        pred = lgf.argmax(1)
        # rank of true token = number of tokens with higher logit
        true_logit = lgf[torch.arange(tf.shape[0], device=DEV), tf]
        rank = (lgf > true_logit.unsqueeze(1)).sum(1)  # 0 = top-1
        true_cls = cidx[tf]; pred_cls = cidx[pred]
        is_hit = pred == tf; is_cont = (~is_hit) & (pred_cls == true_cls); is_gram = (~is_hit) & (pred_cls != true_cls)
        tf_np = tf.cpu().numpy(); pred_np = pred.cpu().numpy(); rank_np = rank.cpu().numpy()
        for name, msk in [('hit', is_hit.cpu().numpy()), ('content_error', is_cont.cpu().numpy()), ('grammar_error', is_gram.cpu().numpy())]:
            agg[name]['logfreq_true'].append(logfreq[tf_np[msk]])
            agg[name]['top1_is_func'].append(is_func[pred_np[msk]])
            agg[name]['true_rank'].append(rank_np[msk])
    base_func_rate = float(is_func[tgt_all].mean())  # corpus base rate of function-class tokens
    out = {'corpus_base_func_rate': round(base_func_rate, 4), 'by_error_type': {}}
    for k in agg:
        lf = np.concatenate(agg[k]['logfreq_true']); tf_ = np.concatenate(agg[k]['top1_is_func']); rk = np.concatenate(agg[k]['true_rank'])
        out['by_error_type'][k] = {'mean_logfreq_true': round(float(lf.mean()), 3),
                                   'top1_func_frac': round(float(tf_.mean()), 3),
                                   'median_true_rank': int(np.median(rk)), 'n': int(len(lf))}
        print(f"{k:>14}: true logfreq {out['by_error_type'][k]['mean_logfreq_true']} | top1 func-frac {out['by_error_type'][k]['top1_func_frac']} | median true-rank {out['by_error_type'][k]['median_true_rank']}", flush=True)
    g = out['by_error_type']['grammar_error']; c = out['by_error_type']['content_error']; h = out['by_error_type']['hit']
    out['pred_a_hedging'] = bool(g['mean_logfreq_true'] < h['mean_logfreq_true'] and g['top1_func_frac'] > base_func_rate + 0.1 and g['median_true_rank'] < 50)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"corpus base func-rate {base_func_rate:.3f}", flush=True)
    print(f"(a) grammar-errors are content-hedging (rare true token, function top-1, modest true-rank): {out['pred_a_hedging']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
