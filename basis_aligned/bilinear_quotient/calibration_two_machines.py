"""CALIBRATION of the two machines: is bilin18 well-calibrated, and is the GRAMMAR (class) prediction better
calibrated than the CONTENT (specific-token) prediction? Ties the uncertainty/hedging theme (§973-976) to a
standard calibration measure. For each position: (a) TOKEN calibration — bin by top-1 token probability, measure
top-1 accuracy per bin, compute ECE; (b) CLASS calibration — bin by the predicted-class marginal probability
(chain-rule class distribution), measure next-token-class accuracy per bin, ECE. Compare ECE and reliability.

REGISTERED PREDICTIONS:
  (0) SANITY: accuracy rises monotonically with confidence for both (positive reliability).
  (a) GRAMMAR BETTER CALIBRATED / MORE CONFIDENT: the CLASS prediction is higher-confidence and at-least-as-well
      calibrated as the TOKEN prediction (grammar is the solved machine); report both ECEs and mean confidences;
  (b) reliability tables (confidence-bin -> accuracy) for token and class + ECE."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'calibration_two_machines_results.json'
NEVAL = 160; SEQ = 256; NBIN = 10
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
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


def ece_table(conf, correct, nbin=NBIN):
    edges = np.linspace(0, 1, nbin+1); ece = 0.0; n = len(conf); tbl = []
    for b in range(nbin):
        lo, hi = edges[b], edges[b+1]; msk = (conf >= lo) & (conf < hi if b < nbin-1 else conf <= hi)
        if msk.sum() == 0: continue
        acc = float(correct[msk].mean()); c = float(conf[msk].mean()); w = int(msk.sum())
        ece += (w/n)*abs(acc - c); tbl.append({'bin': round((lo+hi)/2, 2), 'conf': round(c, 3), 'acc': round(acc, 3), 'n': w})
    return round(ece, 4), tbl


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, C).float()
    tok_conf=[]; tok_ok=[]; cls_conf=[]; cls_ok=[]
    for i in range(0, nb, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        p = F.softmax(forward_logits(idx).float(), -1).reshape(-1, V); tf = tgt.reshape(-1)
        tconf, tpred = p.max(1)
        tok_conf.append(tconf.cpu().numpy()); tok_ok.append((tpred == tf).cpu().numpy())
        pcls = p @ Cmat; cconf, cpred = pcls.max(1)
        cls_conf.append(cconf.cpu().numpy()); cls_ok.append((cpred == cidx[tf]).cpu().numpy())
    tok_conf=np.concatenate(tok_conf); tok_ok=np.concatenate(tok_ok); cls_conf=np.concatenate(cls_conf); cls_ok=np.concatenate(cls_ok)
    tok_ece, tok_tbl = ece_table(tok_conf, tok_ok); cls_ece, cls_tbl = ece_table(cls_conf, cls_ok)
    out = {'token': {'ece': tok_ece, 'mean_conf': round(float(tok_conf.mean()),3), 'mean_acc': round(float(tok_ok.mean()),3), 'reliability': tok_tbl},
           'class': {'ece': cls_ece, 'mean_conf': round(float(cls_conf.mean()),3), 'mean_acc': round(float(cls_ok.mean()),3), 'reliability': cls_tbl}}
    out['pred_a_grammar_better_calibrated'] = bool(out['class']['mean_conf'] > out['token']['mean_conf'] and out['class']['ece'] <= out['token']['ece'] + 0.02)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"TOKEN: ECE {tok_ece} mean-conf {out['token']['mean_conf']} mean-acc {out['token']['mean_acc']}", flush=True)
    print(f"CLASS: ECE {cls_ece} mean-conf {out['class']['mean_conf']} mean-acc {out['class']['mean_acc']}", flush=True)
    print(f"(a) grammar higher-confidence + at-least-as-calibrated: {out['pred_a_grammar_better_calibrated']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
