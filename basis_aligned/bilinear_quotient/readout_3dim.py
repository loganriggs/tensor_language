"""WHAT ARE THE ~3 DIMENSIONS THE READOUT COLLAPSES TO? (§857: L17 token-mean eff-dim 2.8). The final
residual, per token, lives in a ~3-dim subspace. Name those axes: take the top-3 right-singular
directions of the L17 token-conditional-mean matrix and characterize each by its token loadings —
correlation with log-frequency, grammatical class, mean position, and the extreme tokens.

REGISTERED PREDICTIONS:
  (0) SANITY: top-3 singular values dominate (consistent with eff-dim ~3);
  (a) name each of the top-3 axes (candidates: a frequency/confidence axis, a class axis, a
      content/semantic axis) via |corr| with log-freq / class-separation / position and the top +/- tokens;
  (b) report how much variance the top-3 explain."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'readout_3dim_results.json'
NEVAL = 300; MINCOUNT = 10; NDIR = 4; NTOK = 12
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
def capture(rows):
    outs = []; toks = []; poss = []
    def h(mo, i_, o_): outs.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = m.transformer.h[17].register_forward_hook(h)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); poss.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    hh.remove(); return torch.cat(outs, 0), np.concatenate(toks), np.concatenate(poss)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    O, toks, poss = capture(rows)
    uniq, cnts = np.unique(toks, return_counts=True); keep = uniq[cnts >= MINCOUNT]
    means = []; freq = []; kcls = []; kpos = []
    for t in keep:
        mk = toks == t; means.append(O[mk].mean(0)); freq.append(int(mk.sum())); kcls.append(CLASSES.index(classify(d(int(t))))); kpos.append(float(poss[mk].mean()))
    M = torch.stack(means, 0); g = M.mean(0, keepdim=True); Mc = M - g
    U, S, Vh = torch.linalg.svd(Mc, full_matrices=False)
    load = (Mc @ Vh.T).cpu().numpy()                       # (ntok, D) token loadings on each dir
    s2 = (S**2).cpu().numpy(); var3 = float(s2[:3].sum()/s2.sum())
    logf = np.log(np.array(freq)); kcls = np.array(kcls); kpos = np.array(kpos)
    dirs = []
    for k in range(NDIR):
        c = load[:, k]
        cr_freq = abs(float(np.corrcoef(c, logf)[0, 1]))
        # class separation: F-like ratio (between-class var / within)
        cmean = np.array([c[kcls == j].mean() if (kcls == j).any() else 0 for j in range(len(CLASSES))])
        cls_sep = float(cmean.std()/(c.std()+1e-9))
        cr_pos = abs(float(np.corrcoef(c, kpos)[0, 1]))
        pos_tok = [repr(d(int(keep[j]))) for j in np.argsort(-c)[:NTOK]]
        neg_tok = [repr(d(int(keep[j]))) for j in np.argsort(c)[:NTOK]]
        dirs.append({'dir': k, 'sv': round(float(S[k]), 1), 'corr_logfreq': round(cr_freq, 2),
                     'class_separation': round(cls_sep, 2), 'corr_position': round(cr_pos, 2),
                     'top_pos_tokens': pos_tok, 'top_neg_tokens': neg_tok})
    out = {'n_tokens': len(keep), 'top3_var_frac': round(var3, 3), 'directions': dirs, 'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"L17 token-mean top-3 explain {var3:.1%} of variance", flush=True)
    for r in dirs:
        print(f"  dir{r['dir']} (sv {r['sv']}): |corr|logfreq {r['corr_logfreq']} class-sep {r['class_separation']} |corr|pos {r['corr_position']}", flush=True)
        print(f"    +{r['top_pos_tokens'][:8]}", flush=True)
        print(f"    -{r['top_neg_tokens'][:8]}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
