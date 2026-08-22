"""HOW MUCH OF THE LOSS IS PREDICTING THE CLASS vs CHOOSING THE WORD WITHIN THE CLASS?
(quantifies §828/§798). §828 showed the model predicts the next grammatical CLASS near-perfectly;
§798 said the residue is content. Decompose the model's cross-entropy exactly, by the chain rule:
  −log p(target) = −log p(class of target)  +  −log p(target | its class)
  CE_total       = CE_class                 +  CE_within
averaged over tokens. If class-prediction is the easy skeleton and specific-token the hard residue,
CE_within should be most of CE_total.

REGISTERED PREDICTIONS:
  (0) SANITY: CE_class + CE_within == CE_total exactly (chain rule);
  (a) WITHIN-CLASS DOMINATES: CE_within / CE_total is large (>= 0.6) — most of the model's loss is
      choosing which word inside the correct grammatical class, not getting the class;
  (b) report CE_class, CE_within, CE_total and the fraction; also per-current-class if informative."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ce_decomposition_results.json'
NEVAL = 300
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


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    V = int(m.lm_head.weight.shape[0])
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(rows[:, :257].reshape(-1).cpu().numpy()):
        tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); NC = len(CLASSES)
    Cmat = F.one_hot(cidx, NC).float()                       # (V, NC)
    tot_class = 0.0; tot_within = 0.0; tot_total = 0.0; n = 0
    by_cur = {c: [0.0, 0.0, 0] for c in CLASSES}             # class, within, count keyed by CURRENT token class
    for i in range(0, NEVAL, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); logp = F.log_softmax(lg, -1)   # (B,T,V)
        p = logp.exp(); pclass = p @ Cmat                                # (B,T,NC) prob of each class
        tgtf = tgt.reshape(-1); logpf = logp.reshape(-1, V); pcf = pclass.reshape(-1, NC)
        tgt_cls = cidx[tgtf]                                             # (N,)
        lp_tok = logpf[torch.arange(tgtf.shape[0], device=DEV), tgtf]    # log p(target)
        lp_cls = (pcf[torch.arange(tgtf.shape[0], device=DEV), tgt_cls] + 1e-12).log()  # log p(target's class)
        ce_class = -lp_cls; ce_within = -(lp_tok - lp_cls); ce_total = -lp_tok
        tot_class += float(ce_class.sum()); tot_within += float(ce_within.sum()); tot_total += float(ce_total.sum()); n += tgtf.shape[0]
        cur_cls = cidx[idx.reshape(-1)].cpu().numpy()
        for c in range(NC):
            mk = cur_cls == c
            if mk.any():
                mkt = torch.tensor(mk, device=DEV)
                by_cur[CLASSES[c]][0] += float(ce_class[mkt].sum()); by_cur[CLASSES[c]][1] += float(ce_within[mkt].sum()); by_cur[CLASSES[c]][2] += int(mk.sum())
    CEc = tot_class/n; CEw = tot_within/n; CEt = tot_total/n
    per = {c: {'ce_class': round(v[0]/v[2], 3), 'ce_within': round(v[1]/v[2], 3), 'n': v[2]} for c, v in by_cur.items() if v[2] >= 20}
    out = {'ce_class': round(CEc, 4), 'ce_within': round(CEw, 4), 'ce_total': round(CEt, 4),
           'chain_rule_check': round(CEc + CEw - CEt, 6), 'within_fraction': round(CEw/CEt, 4),
           'per_current_class': per, 'pred_a_within_dominates': bool(CEw/CEt >= 0.6), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'CE_total {CEt:.3f} = CE_class {CEc:.3f} + CE_within {CEw:.3f} (check {CEc+CEw-CEt:.2e})', flush=True)
    print(f'within-class fraction of loss: {CEw/CEt:.1%}', flush=True)
    print(f'(a) within-class choice dominates the loss (>=60%): {out["pred_a_within_dominates"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
