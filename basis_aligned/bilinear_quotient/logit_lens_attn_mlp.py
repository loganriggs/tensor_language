"""ATTENTION vs MLP: which builds content? logit-lens after each attn and after each mlp
"""  # orig:
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
OUT = PT + 'logit_lens_attn_mlp_results.json'
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
    # stream_after_attn[L] = input to mlp[L] (pre-hook on mlp); stream_after_mlp[L] = block[L] output
    saa = {L: [] for L in range(18)}; sam = {L: [] for L in range(18)}; seqs = []; hs = []
    for L in range(18):
        def mkpre(L):
            def pre(mo, a): saa[L].append(a[0].detach().float().reshape(-1, D))
            return pre
        def mkpost(L):
            def post(mo, i_, o_): sam[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return post
        hs.append(m.transformer.h[L].mlp.register_forward_pre_hook(mkpre(L)))
        hs.append(m.transformer.h[L].register_forward_hook(mkpost(L)))
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    for h in hs: h.remove()
    SAA = {L: torch.cat(saa[L], 0) for L in range(18)}; SAM = {L: torch.cat(sam[L], 0) for L in range(18)}
    return SAA, SAM, np.concatenate([s.reshape(-1) for s in seqs])


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
    SAA, SAM, toks = capture(rows); V = int(m.lm_head.weight.shape[0]); T = 256
    seqs = toks.reshape(-1, T); tgt = np.full_like(seqs, -1); tgt[:, :-1] = seqs[:, 1:]; tgtf = tgt.reshape(-1); valid = tgtf >= 0
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(toks): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, len(CLASSES)).float()
    tg = torch.tensor(np.where(valid,tgtf,0), device=DEV); vmask = torch.tensor(valid, device=DEV)
    out = {'layers': {}}
    prev_w = None
    for L in range(18):
        ca, wa = split_ce_from_resid(SAA[L][vmask], tg[vmask], cidx, Cmat, V)   # after attn L
        cm, wm = split_ce_from_resid(SAM[L][vmask], tg[vmask], cidx, Cmat, V)   # after mlp L
        # attn contribution = prev_mlp_within - after_attn_within; mlp contribution = after_attn_within - after_mlp_within
        attn_dw = (prev_w - wa) if prev_w is not None else None
        mlp_dw = wa - wm
        out['layers'][f'L{L}'] = {'within_after_attn': round(wa,3), 'within_after_mlp': round(wm,3),
                                  'attn_content_gain': (round(attn_dw,3) if attn_dw is not None else None), 'mlp_content_gain': round(mlp_dw,3)}
        print(f"L{L:>2}: after-attn within {wa:.3f} after-mlp {wm:.3f} | attn gain {attn_dw if attn_dw is None else round(attn_dw,3)} mlp gain {mlp_dw:+.3f}", flush=True)
        prev_w = wm
    tot_attn = sum(v['attn_content_gain'] for v in out['layers'].values() if v['attn_content_gain'] is not None)
    tot_mlp = sum(v['mlp_content_gain'] for v in out['layers'].values())
    out['total_attn_content_gain'] = round(tot_attn,3); out['total_mlp_content_gain'] = round(tot_mlp,3)
    out['runtime_s'] = round(time.time()-t0,1)
    json.dump(out, open(OUT,'w'), indent=1)
    print(f"\nTOTAL content (within-CE) reduction: attention {tot_attn:.2f} vs MLP {tot_mlp:.2f}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")

if __name__ == '__main__':
    main()
