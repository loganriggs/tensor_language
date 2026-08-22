"""WHERE is the CONTENT (within-class) prediction built across depth? (locates the content wall without
naming it). Logit-lens each layer's residual (apply the model's final readout to the intermediate stream)
and split the readable CE into GRAMMAR (predicting the next class) vs CONTENT (within-class word choice),
by the chain rule. Tracing both across depth shows WHERE grammar is solved and WHERE content is built —
complementing the geometry (§857) and the loss split (§829).

REGISTERED PREDICTIONS:
  (0) SANITY: final-layer logit-lens CE ≈ the true model CE (class ~0.75, within ~2.48);
  (a) GRAMMAR EARLY, CONTENT LATE: class-CE (grammar) drops to near-final by the front layers and is flat
      after; within-class-CE (content) keeps dropping through the mid/back layers — content is built late/
      gradually while grammar is settled early;
  (b) report class-CE and within-class-CE by layer (logit lens)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'logit_lens_content_results.json'
NEVAL = 200
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


def readout(x):
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def capture(rows):
    caps = {L: [] for L in range(18)}; seqs = []
    hs = []
    for L in range(18):
        def mk(L):
            def h(mo, i_, o_): caps[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float())
            return h
        hs.append(m.transformer.h[L].register_forward_hook(mk(L)))
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    for h in hs: h.remove()
    return {L: torch.cat([c.reshape(-1, D) for c in caps[L]], 0) for L in range(18)}, np.concatenate([s.reshape(-1) for s in seqs])


@torch.no_grad()
def split_ce_from_resid(resid, tgt, cidx, Cmat, V):
    tc = tw = 0.0; n = 0
    for i in range(0, resid.shape[0], 4096):
        r = resid[i:i+4096]; tg = tgt[i:i+4096]
        lg = readout(r).float(); logp = F.log_softmax(lg, -1); pcl = logp.exp() @ Cmat
        tgt_cls = cidx[tg]
        lp_tok = logp[torch.arange(r.shape[0], device=DEV), tg]
        lp_cls = (pcl[torch.arange(r.shape[0], device=DEV), tgt_cls] + 1e-12).log()
        tc += float((-lp_cls).sum()); tw += float((-(lp_tok - lp_cls)).sum()); n += r.shape[0]
    return tc/n, tw/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    reps, toks = capture(rows)
    V = int(m.lm_head.weight.shape[0])
    # targets = next token; align: residual at pos i predicts token i+1. reps/toks are over positions 0..T-2 (idx[:-1]).
    # build target = current-position's NEXT token. Reconstruct per-seq.
    T = 256
    seqs = toks.reshape(-1, T)
    tgt = np.full_like(seqs, -1); tgt[:, :-1] = seqs[:, 1:]; tgtf = tgt.reshape(-1)
    valid = tgtf >= 0
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(toks): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, len(CLASSES)).float()
    tgt_t = torch.tensor(np.where(valid, tgtf, 0), device=DEV)
    vmask = torch.tensor(valid, device=DEV)
    out = {'layers': {}}
    for L in range(18):
        r = reps[L][vmask]; tg = tgt_t[vmask]
        c, w = split_ce_from_resid(r, tg, cidx, Cmat, V)
        out['layers'][f'L{L}'] = {'class_ce': round(c, 3), 'within_ce': round(w, 3), 'total': round(c+w, 3)}
        print(f"L{L:>2}: class {c:.3f} | within {w:.3f} | total {c+w:.3f}", flush=True)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
