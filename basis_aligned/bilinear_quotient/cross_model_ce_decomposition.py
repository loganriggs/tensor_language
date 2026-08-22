"""IS THE LOSS SPLIT (grammar easy ~23%, lexical choice hard ~77%) UNIVERSAL? (§829/830 cross-
model capstone). bilin18: CE = class 0.75 (23%) + within-class 2.48 (77%). Test GPT-2 and
Pythia-410M: chain-rule split of the model CE into predicting the next grammatical class vs
choosing the word within it, using each model's own tokenizer for the POS class assignment.

REGISTERED PREDICTIONS:
  (0) SANITY: CE_class + CE_within == CE_total exactly per model;
  (a) UNIVERSAL: GPT-2 and Pythia also have within-class dominating (>= 0.6 of loss) and a small
      class component -> the "interpretable machinery does the easy grammatical quarter, the hard
      majority is lexical choice" split is not bilin18-specific;
  (b) report CE_class, CE_within, within-fraction per model."""
import json, time, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_model_ce_decomposition_results.json'
MODELS = ['gpt2', 'EleutherAI/pythia-410m']
SEQ = 128; NBLOCK = 120
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}


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


def get_text(nc=2000, npass=70):
    ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
    t = []
    for i, ex in enumerate(ds):
        t.append(ex['text'][:nc])
        if i >= npass: break
    return '\n\n'.join(t)


@torch.no_grad()
def run(model_id):
    tok = AutoTokenizer.from_pretrained(model_id); mdl = AutoModelForCausalLM.from_pretrained(model_id).to(DEV).eval()
    ids = tok(get_text(), return_tensors='pt')['input_ids'][0]; nb = min(NBLOCK, ids.shape[0]//SEQ)
    blocks = ids[:nb*SEQ].reshape(nb, SEQ)
    V = mdl.config.vocab_size; NC = len(CLASSES)
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(blocks.reshape(-1).cpu().numpy()):
        try: tok2cls[int(tid)] = CLASSES.index(classify(tok.decode([int(tid)])))
        except Exception: pass
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, NC).float()
    tc = tw = tt = 0.0; n = 0
    for i in range(0, nb, 4):
        b = blocks[i:i+4].to(DEV)
        lg = mdl(b).logits.float()[:, :-1, :]; tgt = b[:, 1:]
        logp = F.log_softmax(lg, -1); p = logp.exp(); pcl = p @ Cmat
        tgtf = tgt.reshape(-1); logpf = logp.reshape(-1, V); pcf = pcl.reshape(-1, NC); tgt_cls = cidx[tgtf]
        lp_tok = logpf[torch.arange(tgtf.shape[0], device=DEV), tgtf]
        lp_cls = (pcf[torch.arange(tgtf.shape[0], device=DEV), tgt_cls] + 1e-12).log()
        tc += float((-lp_cls).sum()); tw += float((-(lp_tok - lp_cls)).sum()); tt += float((-lp_tok).sum()); n += tgtf.shape[0]
    del mdl; torch.cuda.empty_cache()
    return {'ce_class': round(tc/n, 4), 'ce_within': round(tw/n, 4), 'ce_total': round(tt/n, 4),
            'within_fraction': round(tw/tt, 4), 'chain_check': round((tc+tw-tt)/n, 6)}


def main():
    t0 = time.time(); out = {}
    for mid in MODELS:
        r = run(mid); out[mid] = r
        print(f"{mid}: CE {r['ce_total']} = class {r['ce_class']} + within {r['ce_within']} | within-fraction {r['within_fraction']:.1%}", flush=True)
    out['bilin18_reference'] = {'ce_class': 0.747, 'ce_within': 2.480, 'within_fraction': 0.769}
    out['pred_a_universal'] = bool(all(out[m]['within_fraction'] >= 0.6 for m in MODELS))
    out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) grammar-easy/lexical-hard split universal (within>=60% all): {out['pred_a_universal']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']:.0f}s)")


if __name__ == '__main__':
    main()
