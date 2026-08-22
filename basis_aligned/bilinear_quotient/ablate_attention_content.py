"""DOES CONTENT NEED ATTENTION's CONTEXT? Ablate ALL attention (keep MLPs), split CE into grammar (class)
vs content (within-class). §861 found attention's DIRECT logit-lens content gain is small (1.48 vs MLP
4.58), but attention may have a large INDIRECT role: it moves context between positions that the MLPs then
convert into content. If ablating all attention collapses CONTENT far more than GRAMMAR, attention's role
is context-for-content (indirect) — reconciling the small direct gain with a large indirect importance.

REGISTERED PREDICTIONS:
  (0) SANITY: full CE ~ class 0.75 / within 2.48;
  (a) attention is context-for-content: ablating all attention raises CONTENT (within) CE far more than
      GRAMMAR (class) CE (content/grammar increase ratio > 2) — content depends on attention's context
      even though attention's direct logit gain is small;
  (b) if grammar is hurt as much as content, attention is not content-specific."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ablate_attention_content_results.json'
NEVAL = 200
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}
ABL = {'on': False}


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


def hook(mo, i_, o_):
    if not ABL['on']: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; z = torch.zeros_like(y)
    return (z,) + tuple(o_[1:]) if isinstance(o_, tuple) else z


def reg_all():
    return [m.transformer.h[L].attn.register_forward_hook(hook) for L in range(18)]


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def split_ce(rows, cidx, Cmat, V):
    tc = tw = 0.0; n = 0
    for i in range(0, NEVAL, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); pcl = lp.exp() @ Cmat
        tgtf = tgt.reshape(-1); lpf = lp.reshape(-1, V); tcl = cidx[tgtf]
        lp_tok = lpf[torch.arange(tgtf.shape[0], device=DEV), tgtf]
        lp_cls = (pcl.reshape(-1, len(CLASSES))[torch.arange(tgtf.shape[0], device=DEV), tcl] + 1e-12).log()
        tc += float((-lp_cls).sum()); tw += float((-(lp_tok - lp_cls)).sum()); n += tgtf.shape[0]
    return tc/n, tw/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    V = int(m.lm_head.weight.shape[0])
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(rows[:, :257].reshape(-1).cpu().numpy()): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, len(CLASSES)).float()
    hs = reg_all()
    ABL['on'] = False; c0, w0 = split_ce(rows, cidx, Cmat, V)
    ABL['on'] = True; c1, w1 = split_ce(rows, cidx, Cmat, V)
    for h in hs: h.remove()
    out = {'full': {'class': round(c0, 3), 'within': round(w0, 3)},
           'attn_ablated': {'class': round(c1, 3), 'within': round(w1, 3)},
           'grammar_increase': round(c1-c0, 3), 'content_increase': round(w1-w0, 3),
           'content_over_grammar': round((w1-w0)/max(c1-c0, 1e-6), 2), 'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"full: class {c0:.3f} within {w0:.3f}", flush=True)
    print(f"ALL-ATTENTION ablated: class {c1:.3f} within {w1:.3f}", flush=True)
    print(f"grammar CE increase {c1-c0:+.3f} | content CE increase {w1-w0:+.3f} (content/grammar {out['content_over_grammar']})", flush=True)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
