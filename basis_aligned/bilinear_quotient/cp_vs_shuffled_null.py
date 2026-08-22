"""DECISIVE: is the class+position keep magnitude (0.78) SPECIFIC or a construction artifact?
(§835 follow-up). §835 showed a rank-32 shuffled-position subspace recovers as much as real position
(construction artifact), so the right keep-only null is a SHUFFLED-LABEL subspace of matched rank/
construction — not random-orthonormal. Test the FULL class+position (rank 96 = 64 token-cond + 32
position-cond) against a matched null: rank 96 from SHUFFLED token labels + SHUFFLED position labels,
same construction. Simultaneous centered keep, CE split into class vs within. The SPECIFIC
class+position contribution = real − shuffled.

REGISTERED PREDICTIONS:
  (0) SANITY: real class+position reproduces §833 (class ~0.81, within ~0.76);
  (a) SPECIFIC: real class+position recovery >> shuffled-label recovery (gap >= 0.2 on class and/or
      within) -> the keep magnitude reflects genuine class+position structure, not just rank;
  (b) CONSTRUCTION-INFLATED: if real ≈ shuffled (gap < 0.1), the keep magnitude is mostly a rank/
      construction artifact and the SPECIFIC class+position contribution is small (class+position is
      still real per naming §825 + steering §823, but the keep number overstates it);
  report real and shuffled recovery for class and within, and the specific gaps."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cp_vs_shuffled_null_results.json'
NEVAL = 200; MINCOUNT = 5; RTOK = 64; RPOS = 32
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}
SUBS = {}; MODE = {'op': None, 'key': 'real'}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
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
    MODE['op'] = None; rng = np.random.RandomState(0)
    for w, L in comps:
        O, toks, pos = capture(rows, w, L)
        Ut = mean_subspace(O, toks, RTOK); Up = mean_subspace(O, pos.astype(np.int64), RPOS)
        SUBS[('real', w, L)] = orth(Ut, Up)
        stok = toks.copy(); rng.shuffle(stok); spos = pos.copy(); rng.shuffle(spos)
        Uts = mean_subspace(O, stok, RTOK); Ups = mean_subspace(O, spos.astype(np.int64), RPOS)
        SUBS[('shuf', w, L)] = orth(Uts, Ups)
    hooks = [comp(w, L).register_forward_hook(mk_hook(w, L)) for w, L in comps]
    MODE['op'] = None; fc, fw = ce_split(rows, cidx, Cmat, V)
    MODE['op'] = 'ablate'; ac, aw = ce_split(rows, cidx, Cmat, V)
    ben_c = ac - fc; ben_w = aw - fw
    res = {}
    for key in ['real', 'shuf']:
        MODE['op'] = 'keep'; MODE['key'] = key; kc, kw = ce_split(rows, cidx, Cmat, V); MODE['op'] = None
        res[key] = {'rec_class': round(float((ac-kc)/max(ben_c, 1e-6)), 4), 'rec_within': round(float((aw-kw)/max(ben_w, 1e-6)), 4)}
        print(f'{key} class+position (rank 96) keep: class {res[key]["rec_class"]} | within {res[key]["rec_within"]}', flush=True)
    for h in hooks: h.remove()
    gap_c = res['real']['rec_class'] - res['shuf']['rec_class']; gap_w = res['real']['rec_within'] - res['shuf']['rec_within']
    out = {'benefit_class': round(ben_c, 3), 'benefit_within': round(ben_w, 3), 'recovery': res,
           'specific_class_gap': round(gap_c, 4), 'specific_within_gap': round(gap_w, 4),
           'pred_a_specific': bool(gap_c >= 0.2 or gap_w >= 0.2), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nSPECIFIC (real − shuffled-label): class {gap_c:+.3f} | within {gap_w:+.3f}', flush=True)
    print(f'(a) class+position keep is SPECIFIC (not just rank/construction): {out["pred_a_specific"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
