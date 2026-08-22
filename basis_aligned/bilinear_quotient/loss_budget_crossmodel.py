"""IS THE LOSS BUDGET UNIVERSAL? (generality check for the §879 capstone). §879: in bilin18, 78% of the loss
is first-mention content, induction is nearly free, grammar is a cheap uniform tax. Is that budget SHAPE a
property of language models in general, or specific to bilin18? Run the exact same position-type x
grammar/content decomposition on GPT-2 and GPT-2-large (HF, same GPT-2 BPE vocab, same FineWeb tokens).

CAVEAT: GPT-2 is WebText-trained, slightly out-of-distribution on FineWeb, so absolute CE is inflated a
little; the BUDGET SHAPE (which bucket dominates, grammar cheap, induction cheap) is what we test and is
robust to a level shift.

REGISTERED PREDICTIONS:
  (0) SANITY: each model's fraction-weighted bucket CEs sum to its overall CE;
  (a) UNIVERSAL SHAPE: in BOTH GPT-2 models the FIRST-MENTION bucket dominates the loss (>60% of total),
      induction is cheap (inductable bucket < 5% of loss), and grammar (class CE) is small and roughly
      uniform across buckets -> the budget shape from §879 is not bilin18-specific;
  (b) if the shape differs markedly (e.g. induction carries a large share, or grammar dominates), the
      §879 budget is model-specific (report plainly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import DEV
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'loss_budget_crossmodel_results.json'
NEVAL = 200; SEQ = 256; MODELS = ['gpt2', 'gpt2-large']
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


@torch.no_grad()
def budget_for(mdl, blocks, cidx, Cmat, V, inductable, firstment, other):
    cl_ce = []; wi_ce = []
    for i in range(0, blocks.shape[0], 4):
        bb = blocks[i:i+4].to(DEV); lg = mdl(bb[:, :-1]).logits.float(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(lg, -1); pcl = lp.exp() @ Cmat
        tf = tgt.reshape(-1); lpf = lp.reshape(-1, V); tcl = cidx[tf]
        lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
        lp_cls = (pcl.reshape(-1, len(CLASSES))[torch.arange(tf.shape[0], device=DEV), tcl] + 1e-12).log()
        cl_ce.append((-lp_cls).cpu().numpy()); wi_ce.append((-(lp_tok - lp_cls)).cpu().numpy())
    cl_f = np.concatenate(cl_ce); wi_f = np.concatenate(wi_ce); tot_f = cl_f + wi_f
    res = {'overall_ce': round(float(tot_f.mean()), 3), 'overall_class_ce': round(float(cl_f.mean()), 3),
           'overall_within_ce': round(float(wi_f.mean()), 3), 'buckets': {}}
    for name, mk in [('inductable', inductable), ('first_mention', firstment), ('seen_other', other)]:
        frac = float(mk.mean()); ce = float(tot_f[mk].mean()); contrib = frac*ce
        res['buckets'][name] = {'fraction': round(frac, 3), 'mean_ce': round(ce, 3),
                                'class_ce': round(float(cl_f[mk].mean()), 3), 'within_ce': round(float(wi_f[mk].mean()), 3),
                                'pct_of_total_loss': round(float(100*contrib/float(tot_f.mean())), 1)}
    return res


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    V = 50257
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(S.reshape(-1)):
        if int(tid) < V: tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, len(CLASSES)).float()
    inductable = np.zeros((nb, SEQ-1), dtype=bool); firstment = np.zeros((nb, SEQ-1), dtype=bool)
    for r in range(nb):
        seen_tok = set(); seen_big = {}
        for pp in range(SEQ-1):
            cur = int(S[r, pp]); nxt = int(S[r, pp+1])
            firstment[r, pp] = nxt not in seen_tok
            if cur in seen_big and seen_big[cur] == nxt: inductable[r, pp] = True
            seen_big[cur] = nxt; seen_tok.add(cur)
    inductable = inductable.reshape(-1); firstment = firstment.reshape(-1) & ~inductable; other = ~inductable & ~firstment
    out = {'position_fractions': {'inductable': round(float(inductable.mean()), 3),
           'first_mention': round(float(firstment.mean()), 3), 'seen_other': round(float(other.mean()), 3)},
           'bilin18_reference_pct': {'inductable': 1.3, 'first_mention': 78.4, 'seen_other': 20.3}, 'models': {}}
    for mid in MODELS:
        print(f"loading {mid}...", flush=True)
        mdl = AutoModelForCausalLM.from_pretrained(mid).to(DEV).eval()
        r = budget_for(mdl, blocks, cidx, Cmat, V, inductable, firstment, other); del mdl; torch.cuda.empty_cache()
        out['models'][mid] = r
        b = r['buckets']
        print(f"{mid}: overall {r['overall_ce']} | first-mention {b['first_mention']['pct_of_total_loss']}% | seen-other {b['seen_other']['pct_of_total_loss']}% | inductable {b['inductable']['pct_of_total_loss']}%", flush=True)
    fm = [out['models'][mid]['buckets']['first_mention']['pct_of_total_loss'] for mid in MODELS]
    ind = [out['models'][mid]['buckets']['inductable']['pct_of_total_loss'] for mid in MODELS]
    out['pred_a_universal_shape'] = bool(all(f > 60 for f in fm) and all(i < 5 for i in ind))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"first-mention % of loss: {dict(zip(MODELS, fm))} (bilin18 78.4)", flush=True)
    print(f"(a) budget shape universal: {out['pred_a_universal_shape']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
