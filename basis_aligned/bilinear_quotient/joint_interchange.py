"""CAPSTONE causal-abstraction test (Geiger): can we COMPOSITIONALLY control both machines? Patch the CLASS
subspace and the CONTENT/topic subspace at L15 base<-source (from a different source context) and verify the
prediction shifts toward the SOURCE's grammatical class AND the SOURCE's topic — jointly. Compare class-only patch,
content-only patch, joint patch, and a random-subspace patch (null). If the joint patch moves BOTH the predicted
class toward the source's class AND the topic toward the source's topic, we have compositional causal control of
the two named machines — the causal-abstraction ideal (§892 class + §894 topic, now JOINT).

REGISTERED PREDICTIONS:
  (0) SANITY: random-subspace patch of the same total rank barely moves class or topic.
  (a) COMPOSITIONAL CONTROL: the joint (class+content) patch shifts the predicted next-token CLASS toward the
      source's class (like class-only) AND the topic-distinctive-token mass toward the source's topic (like
      content-only) — i.e. the two controls compose without destroying each other; class-only moves class but not
      topic, content-only moves topic but not class (dissociation);
  (b) report class-match and topic-match shifts for class-only / content-only / joint / random patches."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F
from collections import Counter

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'joint_interchange_results.json'
CONTENT_L = 15; NEVAL = 200; SEQ = 256; RTOK = 64; RPOS = 32; K = 12; RCLASS = 8; RCONTENT = 24; QP = 200
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
PATCH = {'on': False, 'vec': None, 'U': None}


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
    y = o_[0] if isinstance(o_, tuple) else o_; y = y.clone()
    U = PATCH['U']; b = y[:, QP, :]
    y[:, QP, :] = b - (b @ U) @ U.T + PATCH['vec']   # remove base's component in U, insert source's
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
    # build class + content subspaces from all positions
    Rs = []
    for i in range(0, nb, 4): Rs.append(capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous()).reshape(-1, D))
    R = torch.cat(Rs, 0); toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    Uclass, gcl = mean_subspace(R, nxtcls, RCLASS)
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic_lbl = kmeans(cn, K).cpu().numpy(); Utopic, _ = mean_subspace(content, topic_lbl, RCONTENT)
    Ujoint = torch.linalg.svd(torch.cat([Uclass, Utopic], 1), full_matrices=False)[0][:, :RCLASS+RCONTENT].contiguous()
    gd = torch.Generator(device=DEV).manual_seed(0); Urand = torch.linalg.qr(torch.randn(D, RCLASS+RCONTENT, generator=gd, device=DEV))[0]
    # topic distinctive tokens (for topic-match metric)
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
    # capture per-sequence L15 at QP for source components
    seqR = capL(blocks[:, :SEQ].to(DEV)[:, :-1].contiguous())  # (nb, SEQ-1, D)
    qvec = seqR[:, QP, :]  # (nb, D) base/source residual at QP
    seq_topic = tl[:, QP]  # topic label at QP per sequence
    hh = m.transformer.h[CONTENT_L].register_forward_hook(patch_hook)
    rng = np.random.RandomState(0); pairs = [(i, (i+ nb//2) % nb) for i in range(nb)]
    def predclass(lg):
        t = int(lg.argmax()); return classify(d(t))
    conds = {'class_only': Uclass, 'content_only': Utopic, 'joint': Ujoint, 'random': Urand}
    agg = {c: {'class_to_source': 0, 'topic_to_source': 0, 'n': 0} for c in conds}
    clsrc_base = 0; tpsrc_base = 0
    for (bi, si) in pairs:
        idx = blocks[bi:bi+1, :SEQ].to(DEV)[:, :-1].contiguous()
        # source components
        src = qvec[si]
        src_cls_class = classify(d(int(S[si, QP+1]))) if QP+1 < SEQ else 'word'
        src_topic = int(seq_topic[si])
        PATCH['on'] = False; lg0 = forward_logits(idx).float()[0, QP]
        base_pred_cls = predclass(lg0)
        for c, U in conds.items():
            PATCH['U'] = U; PATCH['vec'] = (src @ U) @ U.T; PATCH['on'] = True
            lg = forward_logits(idx).float()[0, QP]; PATCH['on'] = False
            pc = predclass(lg)
            top10 = set(int(t) for t in torch.topk(lg, 10).indices.cpu().numpy())
            agg[c]['class_to_source'] += int(pc == src_cls_class)
            agg[c]['topic_to_source'] += len(top10 & distinct.get(src_topic, set()))
            agg[c]['n'] += 1
    hh.remove()
    out = {'QP': QP, 'conditions': {}}
    for c in conds:
        a = agg[c]; out['conditions'][c] = {'pred_class_matches_source': round(a['class_to_source']/a['n'], 3),
                                            'source_topic_tokens_in_top10': round(a['topic_to_source']/a['n'], 3)}
        print(f"{c:>13}: class->source {out['conditions'][c]['pred_class_matches_source']} | topic->source(top10 hits) {out['conditions'][c]['source_topic_tokens_in_top10']}", flush=True)
    j = out['conditions']['joint']; r = out['conditions']['random']
    out['pred_a_compositional'] = bool(j['pred_class_matches_source'] > r['pred_class_matches_source'] + 0.05 and
                                       j['source_topic_tokens_in_top10'] > r['source_topic_tokens_in_top10'] + 0.2)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) compositional causal control (joint moves BOTH class and topic > random): {out['pred_a_compositional']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
