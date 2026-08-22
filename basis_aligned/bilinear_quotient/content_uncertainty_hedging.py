"""MECHANISTIC LINK: does the model HEDGE to a function word BECAUSE the CONTENT channel is uncertain? §973 showed
grammar-type errors are content-difficulty hedging. Test whether the model's own CONTENT uncertainty drives it.
Split the next-token predictive distribution (chain rule) into CLASS entropy (grammar uncertainty) and
WITHIN-CLASS entropy (content uncertainty). Bin positions by WITHIN-class (content) entropy and, per bin, measure:
top-1-is-function-word rate (hedging), accuracy, and — as a control — also bin by CLASS entropy. If hedging rises
with CONTENT entropy but not (much) with CLASS entropy, the hedging is driven by the content machine's uncertainty.

REGISTERED PREDICTIONS:
  (0) SANITY: accuracy falls as either entropy rises.
  (a) CONTENT-UNCERTAINTY DRIVES HEDGING: the top-1-function-word (hedge) rate RISES strongly with WITHIN-class
      (content) entropy; class entropy is a weaker/again-present driver -> the model hedges to a function word
      specifically when it is uncertain about the CONTENT word, mechanistically linking §973 hedging to the
      content machine;
  (b) report hedge-rate and accuracy across content-entropy bins and class-entropy bins + the corpus base."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_uncertainty_hedging_results.json'
NEVAL = 160; SEQ = 256
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
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); C = len(CLASSES)
    Cmat = F.one_hot(cidx, C).float()  # (V,C)
    is_func_cls = torch.tensor([CLASSES[c] in FUNCTION for c in range(C)], device=DEV)
    hedge = []; acc = []; cls_ent = []; con_ent = []
    for i in range(0, nb, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); p = lp.exp()
        pf = p.reshape(-1, V); tf = tgt.reshape(-1)
        # class distribution + entropy
        pcls = pf @ Cmat  # (n,C)
        cls_e = -(pcls.clamp_min(1e-12) * pcls.clamp_min(1e-12).log()).sum(1)
        # within-class entropy of the predicted class's tokens (content uncertainty within top class)
        top_cls = pcls.argmax(1)  # model's most-likely class
        # within-class token distribution for the top class: mask tokens of that class, renormalize
        clsmask = (cidx.unsqueeze(0) == top_cls.unsqueeze(1)).float()  # (n,V)
        pw = pf * clsmask; pw = pw / pw.sum(1, keepdim=True).clamp_min(1e-12)
        con_e = -(pw.clamp_min(1e-12) * pw.clamp_min(1e-12).log()).sum(1)
        pred = pf.argmax(1)
        hedge.append(is_func_cls[cidx[pred]].cpu().numpy())
        acc.append((pred == tf).cpu().numpy())
        cls_ent.append(cls_e.cpu().numpy()); con_ent.append(con_e.cpu().numpy())
    hedge = np.concatenate(hedge); acc = np.concatenate(acc); cls_ent = np.concatenate(cls_ent); con_ent = np.concatenate(con_ent)
    base_func = float(hedge.mean())
    def bin_report(ent, name):
        qs = np.quantile(ent, [0, 0.2, 0.4, 0.6, 0.8, 1.0]); rep = []
        for b in range(5):
            lo, hi = qs[b], qs[b+1]; msk = (ent >= lo) & (ent <= hi if b == 4 else ent < hi)
            if msk.sum() == 0: continue
            rep.append({'bin': b, 'hedge_rate': round(float(hedge[msk].mean()), 3), 'acc': round(float(acc[msk].mean()), 3), 'n': int(msk.sum())})
        return rep
    out = {'base_func_rate': round(base_func, 4), 'by_content_entropy': bin_report(con_ent, 'content'),
           'by_class_entropy': bin_report(cls_ent, 'class')}
    ce = out['by_content_entropy']; cl_ = out['by_class_entropy']
    out['content_hedge_rise'] = round(ce[-1]['hedge_rate'] - ce[0]['hedge_rate'], 3)
    out['class_hedge_rise'] = round(cl_[-1]['hedge_rate'] - cl_[0]['hedge_rate'], 3)
    out['pred_a_content_drives_hedge'] = bool(out['content_hedge_rise'] > 0.15)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print("by CONTENT entropy (low->high):", [(r['bin'], r['hedge_rate'], r['acc']) for r in ce], flush=True)
    print("by CLASS entropy   (low->high):", [(r['bin'], r['hedge_rate'], r['acc']) for r in cl_], flush=True)
    print(f"hedge-rate rise: content {out['content_hedge_rise']} vs class {out['class_hedge_rise']} (base func {base_func:.3f})", flush=True)
    print(f"(a) content uncertainty drives hedging: {out['pred_a_content_drives_hedge']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
