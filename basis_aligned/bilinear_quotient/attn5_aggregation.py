"""WHAT DOES attn5 AGGREGATE? (biggest uncharacterized component, 1.97 nats). §849/scan: attn5 collapses
the geometry (eff-dim 23) and does NOT carry token identity (token decode ~0.33) but encodes class (0.59)
— it aggregates rather than copies. Test whether attn5's output is a CONTEXT-POOLED summary: decode from
its output (i) the CURRENT token's class, (ii) the running CONTEXT class-distribution (causal mean of
class over the preceding tokens — what a pooling head would carry), (iii) absolute POSITION. Contrast
every decode with attn2 (a token-COPIER) to see what is distinctive to attn5.

REGISTERED PREDICTIONS:
  (0) SANITY: attn2 (copier) decodes current/prev token high; attn5 low (reproduces the scan);
  (a) AGGREGATOR: attn5 output predicts the running CONTEXT class-distribution (R^2) and POSITION much
      better than attn2 does, i.e. attn5 pools class/position over the context rather than copying a token;
  (b) report current-class acc, context-class-distribution R^2, position acc for attn5 vs attn2."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn5_aggregation_results.json'
NEVAL = 240; T = 256; WIN = 32
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
def capture(rows, layers):
    outs = {L: [] for L in layers}; seqs = []; hs = []
    for L in layers:
        def mk(L):
            def h(mo, i_, o_): outs[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].attn.register_forward_hook(mk(L)))
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    for h in hs: h.remove()
    return {L: torch.cat(outs[L], 0) for L in layers}, np.concatenate(seqs, 0)


def clf_acc(Ft, y, valid, ncls, seed=0):
    idx = np.where(valid)[0]; rng = np.random.RandomState(seed); rng.shuffle(idx)
    ntr = int(0.7*len(idx)); tr, te = idx[:ntr], idx[ntr:]
    Y = torch.zeros(len(tr), ncls, device=DEV); Y[torch.arange(len(tr)), torch.tensor(y[tr], device=DEV)] = 1.0
    A = Ft[tr].T @ Ft[tr] + 1e2*torch.eye(D, device=DEV); Wp = torch.linalg.solve(A, Ft[tr].T @ Y)
    return float(((Ft[te] @ Wp).argmax(1).cpu().numpy() == y[te]).mean())


def reg_r2(Ft, Yt, seed=0):
    n = Ft.shape[0]; rng = np.random.RandomState(seed); perm = rng.permutation(n); ntr = int(0.7*n)
    tr, te = perm[:ntr], perm[ntr:]
    A = Ft[tr].T @ Ft[tr] + 1e2*torch.eye(D, device=DEV); Wp = torch.linalg.solve(A, Ft[tr].T @ Yt[tr])
    pred = Ft[te] @ Wp; resid = ((Yt[te]-pred)**2).sum(); tot = ((Yt[te]-Yt[te].mean(0))**2).sum()
    return float(1 - resid/tot)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    outs, seqs = capture(rows, [2, 5]); nseq = seqs.shape[0]
    cur = seqs.reshape(-1)
    curcls = np.array([CLASSES.index(classify(d(int(t)))) for t in cur])
    # running context class-distribution (causal mean of class one-hot over preceding WIN tokens)
    cls_seq = curcls.reshape(nseq, T); nc = len(CLASSES)
    onehot = np.eye(nc)[cls_seq]                     # (nseq,T,nc)
    ctx = np.zeros_like(onehot)
    for i in range(T):
        lo = max(0, i-WIN); ctx[:, i] = onehot[:, lo:i+1].mean(1) if i > 0 else onehot[:, 0]
    ctx = torch.tensor(ctx.reshape(-1, nc), device=DEV, dtype=torch.float32)
    posbin = (np.broadcast_to(np.arange(T), (nseq, T)) // 32).reshape(-1); nposb = int(posbin.max())+1
    prev = np.full((nseq, T), -1); prev[:, 1:] = seqs[:, :-1]; prev = prev.reshape(-1)
    uniq, cnts = np.unique(cur, return_counts=True); topv = set(uniq[np.argsort(-cnts)[:200]].tolist())
    remap = {t: i for i, t in enumerate(sorted(topv))}; lbl = lambda a: np.array([remap.get(int(t), -1) for t in a])
    prev_l = lbl(prev)
    res = {}
    for L in [2, 5]:
        Ft = outs[L]
        res[f'attn{L}'] = {
            'current_class_acc': round(clf_acc(Ft, curcls, np.ones_like(curcls, bool), nc), 3),
            'prev_token_acc': round(clf_acc(Ft, prev_l, prev_l >= 0, 200), 3),
            'context_class_r2': round(reg_r2(Ft, ctx), 3),
            'position_acc': round(clf_acc(Ft, posbin, np.ones_like(posbin, bool), nposb), 3),
        }
        print(f"attn{L}: current-class {res[f'attn{L}']['current_class_acc']} | prev-token {res[f'attn{L}']['prev_token_acc']} | context-class R2 {res[f'attn{L}']['context_class_r2']} | position {res[f'attn{L}']['position_acc']}", flush=True)
    a2, a5 = res['attn2'], res['attn5']
    out = {'window': WIN, 'results': res,
           'attn5_aggregator': bool(a5['context_class_r2'] > a2['context_class_r2'] + 0.1 and a5['prev_token_acc'] < a2['prev_token_acc'] - 0.1),
           'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) attn5 is a CONTEXT AGGREGATOR (context-class R2 >> copier attn2, token-copy <<): {out['attn5_aggregator']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
