"""DOES THE CLASS REPRESENTATION CAUSALLY DRIVE GRAMMATICAL SEQUENCING? (closes the loop between
§828 and §837, robust — steering-based, not the retracted keep-only). §828: the model predicts the
next grammatical CLASS well (matches the class-bigram). §837: class steering is class-specific. Now
combine: steer the representation toward class B at the front components and measure the predicted
next-CLASS distribution. If the class representation drives grammatical sequencing, steering toward B
should shift the predicted next-class toward what actually FOLLOWS B in the data (e.g. steer toward a
determiner -> predicted next-class shifts toward nouns/words).

REGISTERED PREDICTIONS:
  (0) SANITY: unsteered predicted next-class ~ matches the empirical class-bigram (reproduces §828);
  (a) CAUSAL SEQUENCING: steering toward class B shifts the predicted next-class distribution toward
      B's empirical next-class row (KL to B's-follows drops vs unsteered), for grammatically-clear B
      (determiner, preposition); a random-direction steer does not;
  (b) report, per steered class, the predicted next-class before/after and the top shift."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'steering_sequencing_results.json'
NEVAL = 200; MINCOUNT = 5; RTOK = 64; RPOS = 32
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}
SRC = [262, 257, 290, 13]        # the(det), a(det), and(conj), .(punct)
FRONT = list(range(0, 6)); ALPHA = 16.0
ST = {'on': False, 'mode': 'cp', 'delta': {}, 'rand': {}}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    key = (w, L)
    def hook(mo, i_, o_):
        if not ST['on']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        dd = ST['delta'][key] if ST['mode'] == 'cp' else ST['rand'][key]
        v2 = v + ALPHA * dd
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
def pred_nextclass(rows, cidx, Cmat, NC):
    ps = []
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        lg = forward_logits(idx).float(); p = F.softmax(lg, -1).reshape(-1, lg.shape[-1])
        ps.append((p @ Cmat).mean(0).cpu())
    return torch.stack(ps, 0).mean(0).numpy()


def kl(p, q):
    p = np.asarray(p)+1e-9; q = np.asarray(q)+1e-9; p = p/p.sum(); q = q/q.sum()
    return float((p*np.log(p/q)).sum())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    V = int(m.lm_head.weight.shape[0]); NC = len(CLASSES)
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(rows[:, :257].reshape(-1).cpu().numpy()): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, NC).float()
    # empirical next-class row for each source token B (what follows B)
    empB = {}
    for b in SRC:
        cnt = np.zeros(NC)
        for i in range(0, NEVAL, 4):
            bb = rows[i:i+4, :257].cpu().numpy(); cur = bb[:, :-1].reshape(-1); nxt = bb[:, 1:].reshape(-1)
            mk = cur == b
            for nc in tok2cls[nxt[mk]]: cnt[nc] += 1
        empB[b] = cnt/max(cnt.sum(), 1)
    # subspaces + deltas
    subs = {}; gm = {}; tmean = {}
    g_ = torch.Generator(device=DEV).manual_seed(0)
    for L in FRONT:
        for w in ('attn', 'mlp'):
            O, toks, pos = capture(rows, w, L); g = O.mean(0, keepdim=True)
            Ut = mean_subspace(O, toks, RTOK); Up = mean_subspace(O, pos.astype(np.int64), RPOS)
            subs[(w, L)] = torch.linalg.svd(torch.cat([Ut, Up], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous(); gm[(w, L)] = g
            for b in SRC:
                mk = toks == b
                if mk.sum() >= MINCOUNT: tmean[(b, w, L)] = O[mk].mean(0, keepdim=True).to(DEV)
            ST['rand'][(w, L)] = torch.zeros(1, D, device=DEV)
    # random directions (fixed, matched norm set per source below)
    hooks = [comp(w, L).register_forward_hook(mk_hook(w, L)) for L in FRONT for w in ('attn', 'mlp')]
    ST['on'] = False; base_nc = pred_nextclass(rows, cidx, Cmat, NC)
    out = {'classes': CLASSES, 'base_next_class': [round(float(x), 3) for x in base_nc], 'per_source': {}}
    for b in SRC:
        for (w, L) in [(w, L) for L in FRONT for w in ('attn', 'mlp')]:
            key = (w, L)
            if (b, w, L) in tmean:
                dev = tmean[(b, w, L)] - gm[key]; U = subs[key]; dcp = (dev @ U) @ U.T; ST['delta'][key] = dcp
                rd = torch.randn(1, D, generator=g_, device=DEV); ST['rand'][key] = rd/rd.norm()*dcp.norm()
            else: ST['delta'][key] = torch.zeros(1, D, device=DEV); ST['rand'][key] = torch.zeros(1, D, device=DEV)
        ST['on'] = True; ST['mode'] = 'cp'; nc_cp = pred_nextclass(rows, cidx, Cmat, NC)
        ST['mode'] = 'rand'; nc_rand = pred_nextclass(rows, cidx, Cmat, NC); ST['on'] = False
        kb_base = kl(base_nc, empB[b]); kb_cp = kl(nc_cp, empB[b]); kb_rand = kl(nc_rand, empB[b])
        out['per_source'][str(b)] = {'empirical_follows': [round(float(x), 3) for x in empB[b]],
                                     'steered_next_class': [round(float(x), 3) for x in nc_cp],
                                     'kl_to_Bfollows_base': round(kb_base, 4), 'kl_cp': round(kb_cp, 4), 'kl_rand': round(kb_rand, 4),
                                     'cp_moved_toward_Bfollows': round(kb_base - kb_cp, 4)}
        print(f'steer->{b}: KL(pred nextclass ‖ what-follows-{b}) base {kb_base:.3f} -> cp {kb_cp:.3f} (rand {kb_rand:.3f}) | moved {kb_base-kb_cp:+.3f}', flush=True)
    for h in hooks: h.remove()
    moves = [out['per_source'][str(b)]['cp_moved_toward_Bfollows'] for b in SRC]
    randmoves = [out['per_source'][str(b)]['kl_to_Bfollows_base'] - out['per_source'][str(b)]['kl_rand'] for b in SRC]
    out['mean_cp_move'] = round(float(np.mean(moves)), 4); out['mean_rand_move'] = round(float(np.mean(randmoves)), 4)
    out['pred_a_causal_sequencing'] = bool(np.mean(moves) > 0.1 and np.mean(moves) > np.mean(randmoves) + 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nmean cp move toward B-follows {out["mean_cp_move"]:+.3f} | random {out["mean_rand_move"]:+.3f}', flush=True)
    print(f'(a) class steering causally drives grammatical sequencing: {out["pred_a_causal_sequencing"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"] if "runtime_s" in out else round(time.time()-t0)}s)')


if __name__ == '__main__':
    main()
