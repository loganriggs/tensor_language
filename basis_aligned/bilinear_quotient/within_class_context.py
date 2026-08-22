"""WHAT DRIVES THE HARD 77% (within-class lexical choice)? Is it context-reducible or
irreducible entropy? (§829 follow-up). §829 split the loss into class (0.75, easy) + within-class
(2.48, hard, 77%). Test whether the within-class part shrinks with CONTEXT LENGTH: bin tokens by
absolute position and measure class-CE and within-class-CE per bin. If context narrows the specific
word, within-class CE should DROP with position while class-CE stays roughly flat (grammar needs
little context). Also compare against a NO-CONTEXT baseline (position 0-2 tokens) to bound how much
context buys.

REGISTERED PREDICTIONS:
  (0) SANITY: class-CE + within-CE == total-CE per bin (chain rule);
  (a) CONTEXT-REDUCIBLE: within-class CE decreases with position (later tokens, more context ->
      easier word choice); class-CE roughly flat across position -> the hard 77% is substantially
      driven by context, not fixed entropy;
  (b) if within-class CE is flat across position, the lexical residue is mostly irreducible/high-
      entropy regardless of context;
  report class-CE and within-CE for position bins [0-8, 8-32, 32-96, 96-256]."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'within_class_context_results.json'
NEVAL = 300
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}
BINS = [(0, 8), (8, 32), (32, 96), (96, 256)]


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
    Cmat = F.one_hot(cidx, NC).float()
    acc = {b: [0.0, 0.0, 0] for b in range(len(BINS))}     # class-CE, within-CE, count
    for i in range(0, NEVAL, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        T = idx.shape[1]
        lg = forward_logits(idx).float(); logp = F.log_softmax(lg, -1); p = logp.exp()
        pclass = p @ Cmat
        pos = np.broadcast_to(np.arange(T), idx.shape).reshape(-1)
        tgtf = tgt.reshape(-1); logpf = logp.reshape(-1, V); pcf = pclass.reshape(-1, NC)
        tgt_cls = cidx[tgtf]
        lp_tok = logpf[torch.arange(tgtf.shape[0], device=DEV), tgtf]
        lp_cls = (pcf[torch.arange(tgtf.shape[0], device=DEV), tgt_cls] + 1e-12).log()
        ce_class = (-lp_cls).cpu().numpy(); ce_within = (-(lp_tok - lp_cls)).cpu().numpy()
        for bi, (lo, hi) in enumerate(BINS):
            mk = (pos >= lo) & (pos < hi)
            if mk.any():
                acc[bi][0] += float(ce_class[mk].sum()); acc[bi][1] += float(ce_within[mk].sum()); acc[bi][2] += int(mk.sum())
    bins_out = []
    for bi, (lo, hi) in enumerate(BINS):
        c, w, n = acc[bi]
        bins_out.append({'pos_range': [lo, hi], 'ce_class': round(c/n, 4), 'ce_within': round(w/n, 4), 'ce_total': round((c+w)/n, 4), 'n': n})
        print(f"pos [{lo},{hi}): class {c/n:.3f} | within {w/n:.3f} | total {(c+w)/n:.3f} (n {n})", flush=True)
    wfirst = bins_out[0]['ce_within']; wlast = bins_out[-1]['ce_within']
    cfirst = bins_out[0]['ce_class']; clast = bins_out[-1]['ce_class']
    out = {'bins': bins_out, 'within_drop_first_to_last': round(wfirst - wlast, 4),
           'class_drop_first_to_last': round(cfirst - clast, 4),
           'pred_a_context_reducible': bool((wfirst - wlast) > 0.3 and (wfirst - wlast) > 2*abs(cfirst - clast)),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nwithin-class CE drop (early->late) {wfirst-wlast:+.3f} | class-CE drop {cfirst-clast:+.3f}", flush=True)
    print(f"(a) hard 77% is context-reducible: {out['pred_a_context_reducible']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']:.0f}s)")


if __name__ == '__main__':
    main()
