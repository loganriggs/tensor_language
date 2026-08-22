"""ROBUSTNESS of the compositional double dissociation (§959) across QUERY POSITIONS. §899 flagged that fixed-QP
interchange tests can be fragile. Repeat the v3 class-controlled joint interchange at several query positions
(QP in {80,140,200,240}) and check the double dissociation holds everywhere: class-patch moves CLASS not topic,
content-patch moves TOPIC (class-controlled) not class, joint moves both, random neither.

REGISTERED PREDICTIONS:
  (0) SANITY: random patch moves neither at every QP.
  (a) ROBUST DISSOCIATION: at EVERY QP, class_only class->source > random+0.05 and content_only class-controlled
      topic_net > class_only's and > 0.05 -> the compositional double dissociation (§959) is robust to query
      position, not a QP=200 artifact;
  (b) report per-QP class-match and class-controlled topic_net for class_only/content_only/joint/random."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F
from collections import Counter

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'joint_interchange_qpsweep_results.json'
CONTENT_L = 15; NEVAL = 200; SEQ = 256; RTOK = 64; RPOS = 32; K = 12; RCLASS = 8; RCONTENT = 24
QPS = [80, 140, 200, 240]
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
PATCH = {'on': False, 'vec': None, 'U': None, 'qp': -1}


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


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def patch_hook(mo, i_, o_):
    if not PATCH['on']: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; y = y.clone(); U = PATCH['U']; qp = PATCH['qp']
    b = y[:, qp, :]; y[:, qp, :] = b - (b @ U) @ U.T + PATCH['vec']
    return (y,) + tuple(o_[1:]) if isinstance(o_, tuple) else y


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def capL(idx):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[CONTENT_L].register_forward_hook(h); forward_logits(idx); hh.remove(); return cap['r']


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed); c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    Rs = []
    for i in range(0, nb, 4): Rs.append(capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous()).reshape(-1, D))
    R = torch.cat(Rs, 0); toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    Uclass, _ = mean_subspace(R, nxtcls, RCLASS)
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic_lbl = kmeans(cn, K).cpu().numpy(); Utopic, _ = mean_subspace(content, topic_lbl, RCONTENT)
    Ujoint = torch.linalg.svd(torch.cat([Uclass, Utopic], 1), full_matrices=False)[0][:, :RCLASS+RCONTENT].contiguous()
    gd = torch.Generator(device=DEV).manual_seed(0); Urand = torch.linalg.qr(torch.randn(D, RCLASS+RCONTENT, generator=gd, device=DEV))[0]
    tgt = S[:, 1:].reshape(-1); base_ct = Counter(tgt[tgt >= 0]); Nn = int((tgt >= 0).sum())
    tl = topic_lbl.reshape(nb, SEQ-1)
    distinct = {}
    for j in range(K):
        mk = tl.reshape(-1) == j; nc = Counter()
        for t in np.unique(tgt[mk]):
            if t < 0: continue
            c = int((tgt[mk] == t).sum())
            if c < 4: continue
            nc[t] = (c/max(mk.sum(), 1))/((base_ct.get(t, 0)+1)/Nn)
        distinct[j] = set(int(t) for t, _ in nc.most_common(30))
    def dset(topic):
        s = distinct.get(topic, set()); return torch.tensor(sorted(s), device=DEV) if s else None
    seqR = capL(blocks[:, :SEQ].to(DEV)[:, :-1].contiguous())
    conds = {'class_only': Uclass, 'content_only': Utopic, 'joint': Ujoint, 'random': Urand}
    hh = m.transformer.h[CONTENT_L].register_forward_hook(patch_hook)
    pairs = [(i, (i + nb//2) % nb) for i in range(nb)]
    def predclass(lg): return classify(d(int(lg.argmax())))
    out = {'QPS': QPS, 'by_qp': {}}
    robust = True
    for QP in QPS:
        agg = {c: {'cls': 0, 'top': 0.0, 'n': 0} for c in conds}
        for (bi, si) in pairs:
            src_topic = int(tl[si, QP]); base_topic = int(tl[bi, QP])
            if src_topic == base_topic: continue
            Dsrc = dset(src_topic); Dbase = dset(base_topic)
            if Dsrc is None or Dbase is None: continue
            idx = blocks[bi:bi+1, :SEQ].to(DEV)[:, :-1].contiguous(); src = seqR[si, QP, :]
            src_cls = classify(d(int(S[si, QP+1]))) if QP+1 < SEQ else 'word'
            PATCH['on'] = False; PATCH['qp'] = QP; lp0 = F.log_softmax(forward_logits(idx).float()[0, QP], -1)
            b_s = float(lp0[Dsrc].mean()); b_b = float(lp0[Dbase].mean())
            for c, U in conds.items():
                PATCH['U'] = U; PATCH['vec'] = (src @ U) @ U.T; PATCH['on'] = True
                lg = forward_logits(idx).float()[0, QP]; PATCH['on'] = False
                lp = F.log_softmax(lg, -1)
                agg[c]['cls'] += int(predclass(lg) == src_cls)
                agg[c]['top'] += (float(lp[Dsrc].mean())-b_s) - (float(lp[Dbase].mean())-b_b)
                agg[c]['n'] += 1
        res = {c: {'class_to_source': round(agg[c]['cls']/max(agg[c]['n'],1), 3), 'topic_net': round(agg[c]['top']/max(agg[c]['n'],1), 4)} for c in conds}
        out['by_qp'][str(QP)] = res
        ok = (res['class_only']['class_to_source'] > res['random']['class_to_source']+0.05 and
              res['content_only']['topic_net'] > res['class_only']['topic_net'] and res['content_only']['topic_net'] > 0.05)
        robust = robust and ok
        print(f"QP{QP}: class_only cls {res['class_only']['class_to_source']} topic {res['class_only']['topic_net']:+.3f} | content_only cls {res['content_only']['class_to_source']} topic {res['content_only']['topic_net']:+.3f} | random cls {res['random']['class_to_source']} topic {res['random']['topic_net']:+.3f} | ok {ok}", flush=True)
    hh.remove()
    out['pred_a_robust_dissociation'] = bool(robust)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) double dissociation robust across QP: {robust}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
