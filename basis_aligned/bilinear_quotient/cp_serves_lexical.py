"""DOES THE class+position SUBSPACE SERVE LEXICAL (within-class) PREDICTION, not just grammar?
(decisive test of the §832 reconciliation — reframes the whole program). §807: keeping only
class+position at every component recovers 0.78 of total benefit. §829: total benefit is 23% class
+ 77% within-class. Question: of the class+position keep's recovery, how much is CLASS-loss vs
WITHIN-CLASS-loss? If class+position keep recovers a large fraction of the WITHIN-class benefit (not
just the class benefit), then the interpretable low-rank representation drives LEXICAL prediction too
— "class+position = grammar" is too glib. Simultaneous centered keep (the 0.78 metric), CE split into
class vs within by chain rule under full / ablate-all / keep-class+position.

REGISTERED PREDICTIONS:
  (0) SANITY: keep-class+position total recovery ≈ 0.78 (reproduces §794/807);
  (a) SERVES LEXICAL: class+position keep recovers a LARGE fraction of the WITHIN-class benefit
      (>= 0.5), comparable to its class-benefit recovery -> the representation serves both; the
      grammar/lexical loss split is orthogonal to the class+position/remainder representation split;
  (b) if class+position keep recovers the CLASS benefit but little WITHIN benefit, then
      class+position ≈ grammar-only after all (would restore the glib reading);
  NULL: random same-rank simultaneous keep recovers far less of both."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cp_serves_lexical_results.json'
NEVAL = 200; MINCOUNT = 5; RTOK = 64; RPOS = 32
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}
SUBS = {}; MODE = {'op': None, 'rand': None}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    key = (w, L)
    def hook(mo, i_, o_):
        if MODE['op'] is None: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else: U = MODE['rand'] if MODE['op'] == 'keeprand' else SUBS[key]; v2 = (v @ U) @ U.T
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
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
def capture(rows, w, L):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(w, L).register_forward_hook(h)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    V = int(m.lm_head.weight.shape[0]); NC = len(CLASSES)
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(rows[:, :257].reshape(-1).cpu().numpy()): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, NC).float()
    comps = [(w, L) for L in range(18) for w in ('attn', 'mlp')]
    MODE['op'] = None
    for w, L in comps:
        O, toks, pos = capture(rows, w, L)
        Ut = mean_subspace(O, toks, RTOK); Up = mean_subspace(O, pos, RPOS)
        SUBS[(w, L)] = torch.linalg.svd(torch.cat([Ut, Up], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    g = torch.Generator(device=DEV).manual_seed(0); MODE['rand'] = torch.linalg.qr(torch.randn(D, RTOK+RPOS, generator=g, device=DEV))[0]
    hooks = [comp(w, L).register_forward_hook(mk_hook(w, L)) for w, L in comps]
    MODE['op'] = None; fc, fw = ce_split(rows, cidx, Cmat, V)
    MODE['op'] = 'ablate'; ac, aw = ce_split(rows, cidx, Cmat, V)
    MODE['op'] = 'keep'; kc, kw = ce_split(rows, cidx, Cmat, V)
    MODE['op'] = 'keeprand'; rc, rw = ce_split(rows, cidx, Cmat, V); MODE['op'] = None
    for h in hooks: h.remove()
    ben_c = ac - fc; ben_w = aw - fw; ben_t = ben_c + ben_w
    rec_c = (ac - kc)/max(ben_c, 1e-6); rec_w = (aw - kw)/max(ben_w, 1e-6); rec_t = ((ac+aw)-(kc+kw))/max(ben_t, 1e-6)
    rrec_c = (ac - rc)/max(ben_c, 1e-6); rrec_w = (aw - rw)/max(ben_w, 1e-6)
    out = {'full_class': round(fc, 3), 'full_within': round(fw, 3),
           'benefit_class': round(ben_c, 3), 'benefit_within': round(ben_w, 3),
           'cp_recovers_class': round(float(rec_c), 4), 'cp_recovers_within': round(float(rec_w), 4),
           'cp_recovers_total': round(float(rec_t), 4),
           'rand_recovers_class': round(float(rrec_c), 4), 'rand_recovers_within': round(float(rrec_w), 4),
           'pred_a_serves_lexical': bool(rec_w >= 0.5 and rec_w > 2*rrec_w), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'benefit: class {ben_c:.3f} within {ben_w:.3f}', flush=True)
    print(f'class+position keep RECOVERS: class {rec_c:.3f} | within {rec_w:.3f} | total {rec_t:.3f} (random: class {rrec_c:.3f} within {rrec_w:.3f})', flush=True)
    print(f'(a) class+position representation serves LEXICAL prediction too: {out["pred_a_serves_lexical"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
