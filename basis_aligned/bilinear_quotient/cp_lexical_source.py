"""IS THE LEXICAL-DRIVING PART OF class+position the FINE TOKEN-IDENTITY (bigram) or the COARSE
GRAMMATICAL CLASS? (precise follow-up to §833). §833: class+position keep recovers 76% of the
within-class/lexical benefit. But the "class+position" subspace is built from TOKEN-conditional means
(rank 64) — it captures specific token identity, not just the 8-way POS class. So is the lexical
recovery driven by fine token-identity (a bigram: current token -> next word) or by the coarse
grammatical class? Compare simultaneous keep of:
  - COARSE-8-POS-class + position  (rank ~8 class + 32 pos): only coarse grammar + position
  - FINE-token + position          (rank 64 token-cond means + 32 pos): the standard class+position
and split each recovery into class-benefit vs within-class(lexical)-benefit.

REGISTERED PREDICTIONS:
  (0) SANITY: fine-token+position reproduces §833 (class ~0.81, within ~0.76);
  (a) BIGRAM/TOKEN-IDENTITY drives lexical: coarse-8-class+position recovers the CLASS benefit
      decently but the WITHIN benefit POORLY; fine-token+position recovers within well -> the lexical
      driving is fine token-identity (bigram), not coarse grammatical class;
  (b) if coarse-8-class+position already recovers within well, then coarse grammar+position alone
      drives lexical choice (surprising);
  report class- and within-recovery for coarse8+pos, fine+pos, and position-only."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cp_lexical_source_results.json'
NEVAL = 200; MINCOUNT = 5; RTOK = 64; RPOS = 32
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}
SUBS = {}; MODE = {'op': None, 'key': 'fine'}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    key = (w, L)
    def hook(mo, i_, o_):
        if MODE['op'] is None: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else: U = SUBS[(MODE['key'], w, L)]; v2 = (v @ U) @ U.T
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
    k = min(r, M.shape[0])
    return torch.linalg.svd(M, full_matrices=False)[2][:k].T.contiguous()


def orth(*mats):
    C = torch.cat([x for x in mats if x is not None and x.shape[1] > 0], 1)
    return torch.linalg.svd(C, full_matrices=False)[0][:, :C.shape[1]].contiguous()


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
        posl = pos.astype(np.int64); coarse = tok2cls[toks.astype(np.int64)]     # 8-way POS label per position
        Ufine = mean_subspace(O, toks, RTOK); Ucoarse = mean_subspace(O, coarse, 8); Upos = mean_subspace(O, posl, RPOS)
        SUBS[('fine', w, L)] = orth(Ufine, Upos)
        SUBS[('coarse', w, L)] = orth(Ucoarse, Upos)
        SUBS[('pos', w, L)] = orth(Upos)
    hooks = [comp(w, L).register_forward_hook(mk_hook(w, L)) for w, L in comps]
    MODE['op'] = None; fc, fw = ce_split(rows, cidx, Cmat, V)
    MODE['op'] = 'ablate'; ac, aw = ce_split(rows, cidx, Cmat, V)
    ben_c = ac - fc; ben_w = aw - fw
    res = {}
    for key in ['coarse', 'fine', 'pos']:
        MODE['op'] = 'keep'; MODE['key'] = key; kc, kw = ce_split(rows, cidx, Cmat, V); MODE['op'] = None
        res[key] = {'rec_class': round(float((ac-kc)/max(ben_c, 1e-6)), 4), 'rec_within': round(float((aw-kw)/max(ben_w, 1e-6)), 4)}
        print(f'{key}+pos keep: class {res[key]["rec_class"]} | within/lexical {res[key]["rec_within"]}', flush=True)
    for h in hooks: h.remove()
    fine_w = res['fine']['rec_within']; coarse_w = res['coarse']['rec_within']; pos_w = res['pos']['rec_within']
    out = {'benefit_class': round(ben_c, 3), 'benefit_within': round(ben_w, 3), 'recovery': res,
           'token_identity_lexical_gain': round(fine_w - coarse_w, 4),
           'pred_a_bigram_drives_lexical': bool(fine_w - coarse_w > 0.25 and coarse_w < 0.5), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nfine-token adds {fine_w-coarse_w:+.3f} lexical recovery over coarse-8-class (coarse within {coarse_w}, fine {fine_w}, pos-only {pos_w})', flush=True)
    print(f'(a) fine token-identity (bigram) drives lexical, not coarse class: {out["pred_a_bigram_drives_lexical"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
