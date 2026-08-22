"""WHAT does the topic aggregator COMPUTE — a content-word, order-invariant GIST? (mechanizes the NATURE
of the §870/§871 aggregation). §870: attention aggregates context into topic; §871: topic needs long
context. This asks what property of the context the topic depends on, by editing the context and re-reading:
  - FULL: unmodified context (ceiling).
  - FUNCTION-MASKED: replace all function-class tokens (det/prep/conj/pron/punct) with a neutral token,
    keeping content words -> does topic survive on content words alone?
  - CONTENT-MASKED: replace all content-class tokens (word/cap/number) with a neutral token, keeping
    function words -> control: topic should collapse (content carries it).
  - ORDER-SHUFFLED: keep the exact same context tokens but randomly permute their ORDER (query token fixed
    at the end) -> is topic order-invariant (a bag-of-words gist) while grammar needs order?
For each, decode the FULL-context §866 topic label from the query-position L15 residual, and measure grammar
(class) CE for the next token. Controls: shuffled-topic-label decode (chance); the content-masked condition
is the within-experiment control for function-masked.

REGISTERED PREDICTIONS:
  (0) SANITY: FULL topic-decode and class-CE match §871 at this C; shuffled-label ~ chance;
  (a) TOPIC = CONTENT-WORD GIST: FUNCTION-masked keeps most topic decodability (content words carry topic),
      CONTENT-masked collapses it toward chance -> topic is aggregated from content words;
  (b) TOPIC = ORDER-INVARIANT: ORDER-shuffled keeps most topic decodability while grammar (class) CE rises
      sharply -> the aggregator computes an order-insensitive gist, whereas grammar needs word order;
  (c) if function-masking hurts topic as much as content-masking, or order-shuffling destroys topic, the
      aggregator is not a content-word bag-of-words gist (report honestly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'topic_aggregation_nature_results.json'
CONTENT_L = 15; NEVAL = 220; RTOK = 64; RPOS = 32; K = 12; C = 128
QPOS = list(range(160, 256, 6))
NEUTRAL = 262   # ' the'
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
FUNCTION = {'det', 'prep', 'conj', 'pron', 'punct'}
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


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def forward_cap(idx):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[CONTENT_L].register_forward_hook(h)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    hh.remove()
    return cap['r'], readout(x)


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def decode_acc(Ftr, ytr, Fte, yte, ncls):
    Y = torch.zeros(len(ytr), ncls, device=DEV); Y[torch.arange(len(ytr)), torch.tensor(ytr, device=DEV)] = 1.0
    A = Ftr.T @ Ftr + 1e2*torch.eye(Ftr.shape[1], device=DEV); Wp = torch.linalg.solve(A, Ftr.T @ Y)
    return float(((Fte @ Wp).argmax(1).cpu().numpy() == yte).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    V = int(m.lm_head.weight.shape[0])
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(rows[:, :257].reshape(-1).cpu().numpy()): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, len(CLASSES)).float()
    is_func = np.array([CLASSES[tok2cls[t]] in FUNCTION for t in range(V)])
    # ---- full-context topic labels ----
    fullR = []; seqs = []
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :256].to(DEV).contiguous(); r, _ = forward_cap(idx); fullR.append(r.cpu()); seqs.append(idx.cpu().numpy())
    fullR = torch.cat(fullR, 0); S = np.concatenate(seqs, 0); Nseq = S.shape[0]
    allR = fullR.reshape(-1, D).to(DEV); toks = S.reshape(-1); pos = np.broadcast_to(np.arange(256), S.shape).reshape(-1)
    Utok, g = mean_subspace(allR, toks, RTOK); Upos, _ = mean_subspace(allR, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (allR-g) - ((allR-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic_all = kmeans(cn, K).cpu().numpy().reshape(Nseq, 256)
    qp = np.array(QPOS); rng = np.random.RandomState(0); order = rng.permutation(Nseq); ntr = int(0.7*Nseq); tr_s, te_s = order[:ntr], order[ntr:]
    ytr = topic_all[tr_s][:, qp].reshape(-1); yte = topic_all[te_s][:, qp].reshape(-1)
    tshuf = topic_all.reshape(-1).copy(); rng.shuffle(tshuf); tshuf = tshuf.reshape(Nseq, 256)

    def run_condition(edit):
        Rq = torch.zeros(Nseq, len(qp), D); tc = 0.0; ntok = 0
        for i in range(0, Nseq, 4):
            bb = rows[i:i+4, :256].to(DEV)
            for qi, q in enumerate(qp):
                lo = q-C+1
                win = bb[:, lo:q+1].clone()                       # (b, C), last col = query token q
                win = edit(win)
                r, lg = forward_cap(win.contiguous())
                Rq[i:i+bb.shape[0], qi] = r[:, -1].cpu()
                if q+1 < 256:
                    logp = F.log_softmax(lg[:, -1].float(), -1); tgt = bb[:, q+1]
                    pcl = logp.exp() @ Cmat; tcl = cidx[tgt]
                    lp_cls = (pcl[torch.arange(bb.shape[0], device=DEV), tcl] + 1e-12).log()
                    tc += float((-lp_cls).sum()); ntok += bb.shape[0]
        Ftr = Rq[tr_s].reshape(-1, D).to(DEV); Fte = Rq[te_s].reshape(-1, D).to(DEV)
        return round(decode_acc(Ftr, ytr, Fte, yte, K), 3), round(tc/max(ntok, 1), 3)

    def ed_full(win): return win
    def ed_funcmask(win):
        w = win.clone(); fmask = is_func[w.cpu().numpy()]; fmask[:, -1] = False   # keep query token
        w[torch.tensor(fmask, device=DEV)] = NEUTRAL; return w
    def ed_contmask(win):
        w = win.clone(); cmask = ~is_func[w.cpu().numpy()]; cmask[:, -1] = False
        w[torch.tensor(cmask, device=DEV)] = NEUTRAL; return w
    def ed_shuffle(win):
        w = win.clone(); b, L = w.shape
        for r_ in range(b):
            perm = torch.randperm(L-1, device=DEV)                 # shuffle context, keep query at end
            w[r_, :L-1] = w[r_, :L-1][perm]
        return w

    conds = {'full': ed_full, 'function_masked': ed_funcmask, 'content_masked': ed_contmask, 'order_shuffled': ed_shuffle}
    res = {}
    for name, ed in conds.items():
        acc, cce = run_condition(ed); res[name] = {'topic_decode': acc, 'class_ce': cce}
        print(f"{name:>16}: topic-decode {acc} | class-CE {cce}", flush=True)
    # shuffled-label chance on full
    Rq = torch.zeros(Nseq, len(qp), D)
    for i in range(0, Nseq, 4):
        bb = rows[i:i+4, :256].to(DEV)
        for qi, q in enumerate(qp):
            r, _ = forward_cap(bb[:, q-C+1:q+1].contiguous()); Rq[i:i+bb.shape[0], qi] = r[:, -1].cpu()
    chance = round(decode_acc(Rq[tr_s].reshape(-1, D).to(DEV), tshuf[tr_s][:, qp].reshape(-1), Rq[te_s].reshape(-1, D).to(DEV), tshuf[te_s][:, qp].reshape(-1), K), 3)
    f = res['full']['topic_decode']
    out = {'C': C, 'k': K, 'shuffled_label_decode': chance, 'conditions': res,
           'topic_retained_funcmask': round((res['function_masked']['topic_decode']-chance)/max(f-chance, 1e-6), 2),
           'topic_retained_contmask': round((res['content_masked']['topic_decode']-chance)/max(f-chance, 1e-6), 2),
           'topic_retained_shuffle': round((res['order_shuffled']['topic_decode']-chance)/max(f-chance, 1e-6), 2),
           'grammar_ce_rise_shuffle': round(res['order_shuffled']['class_ce']-res['full']['class_ce'], 3), 'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_content_word_gist'] = bool(out['topic_retained_funcmask'] > out['topic_retained_contmask'] + 0.3)
    out['pred_b_order_invariant'] = bool(out['topic_retained_shuffle'] > 0.6 and out['grammar_ce_rise_shuffle'] > 0.3)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nshuffled-label(chance) {chance}", flush=True)
    print(f"topic retained: func-masked {out['topic_retained_funcmask']} | content-masked {out['topic_retained_contmask']} | order-shuffled {out['topic_retained_shuffle']}", flush=True)
    print(f"grammar-CE rise under order-shuffle {out['grammar_ce_rise_shuffle']}", flush=True)
    print(f"(a) topic = content-word gist: {out['pred_a_content_word_gist']} | (b) order-invariant: {out['pred_b_order_invariant']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
