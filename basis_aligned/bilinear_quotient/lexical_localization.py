"""WHICH COMPONENTS DO GRAMMAR vs LEXICAL WORK? (opens the next system after the class+position
program). §829/830: the loss = class/grammar (easy ~23%) + within-class/lexical (hard ~77%, partly
context-driven). The class+position (grammar) machinery is front+back (§812). Where is the LEXICAL
(within-class, context-based word-narrowing) work done? Per component, ablate its output and measure
the increase in CE_class (grammar) vs CE_within (lexical), via the chain-rule split. A component that
mostly raises CE_within specializes in lexical choice; one that mostly raises CE_class in grammar.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating a component raises CE_total = ΔCE_class + ΔCE_within (chain rule);
  (a) SPECIALIZATION: components differ in their grammar-vs-lexical mix; report the ratio
      ΔCE_within/ΔCE_total per component. Expect the big class+position MLPs (mlp0, back MLPs) to be
      relatively grammar-heavy and attention/context components to be relatively lexical-heavy;
  (b) report the components with the highest LEXICAL share and highest GRAMMAR share."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'lexical_localization_results.json'
NEVAL = 200
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}
ABL = {'name': None}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    name = (w, L)
    def hook(mo, i_, o_):
        if ABL['name'] != name: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; z = torch.zeros_like(y)
        return (z,) + tuple(o_[1:]) if isinstance(o_, tuple) else z
    return hook


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
def ce_split(rows, cidx, Cmat, V):
    tc = tw = 0.0; n = 0
    for i in range(0, NEVAL, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); logp = F.log_softmax(lg, -1); pcl = logp.exp() @ Cmat
        tgtf = tgt.reshape(-1); logpf = logp.reshape(-1, V); pcf = pcl.reshape(-1, len(CLASSES)); tgt_cls = cidx[tgtf]
        lp_tok = logpf[torch.arange(tgtf.shape[0], device=DEV), tgtf]
        lp_cls = (pcf[torch.arange(tgtf.shape[0], device=DEV), tgt_cls] + 1e-12).log()
        tc += float((-lp_cls).sum()); tw += float((-(lp_tok - lp_cls)).sum()); n += tgtf.shape[0]
    return tc/n, tw/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    V = int(m.lm_head.weight.shape[0]); NC = len(CLASSES)
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(rows[:, :257].reshape(-1).cpu().numpy()): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, NC).float()
    comps = [(w, L) for L in range(18) for w in ('attn', 'mlp')]
    hooks = [comp(w, L).register_forward_hook(mk_hook(w, L)) for w, L in comps]
    ABL['name'] = None; c0, w0 = ce_split(rows, cidx, Cmat, V)
    per = {}
    for w, L in comps:
        ABL['name'] = (w, L); c1, w1 = ce_split(rows, cidx, Cmat, V); ABL['name'] = None
        dc = c1 - c0; dw = w1 - w0; dt = dc + dw
        per[f'{w}{L}'] = {'d_class': round(dc, 3), 'd_within': round(dw, 3), 'd_total': round(dt, 3),
                          'lexical_share': round(dw/dt, 3) if dt > 0.01 else None}
    for h in hooks: h.remove()
    rel = {k: v for k, v in per.items() if v['d_total'] > 0.05}
    lex = sorted(rel.items(), key=lambda kv: -kv[1]['lexical_share'])[:6]
    gram = sorted(rel.items(), key=lambda kv: kv[1]['lexical_share'])[:6]
    out = {'base_class': round(c0, 3), 'base_within': round(w0, 3), 'per_component': per,
           'most_lexical': [(k, v['lexical_share'], v['d_total']) for k, v in lex],
           'most_grammatical': [(k, v['lexical_share'], v['d_total']) for k, v in gram], 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'base CE_class {c0:.3f} CE_within {w0:.3f}', flush=True)
    print('MOST LEXICAL (high within-share):', [(k, s) for k, s, _ in out['most_lexical']], flush=True)
    print('MOST GRAMMATICAL (low within-share):', [(k, s) for k, s, _ in out['most_grammatical']], flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
