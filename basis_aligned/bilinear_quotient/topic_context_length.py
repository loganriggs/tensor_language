"""HOW MUCH CONTEXT does the TOPIC/content machine need? (mechanizes "attention aggregates context into
topic", §870, with grammar as the built-in control). If topic is aggregated from the surrounding passage,
topic decodability and content prediction should IMPROVE with more visible context, while grammar (next
class) — a local, context-free computation (front MLPs, §863) — should saturate almost immediately.

Method: for each context length C in a sweep, run the model on the LAST C tokens before each query
position (a sliding window) and measure at the query: (i) topic decodability — decode the full-context
§866 topic label from the truncated-context L15 residual (ridge probe); (ii) grammar CE (next-class) and
content CE (within-class) by the chain rule. Full-context (C=256) is the ceiling; short C is the floor.
Controls: shuffled-topic-label decode (chance); grammar CE is the within-experiment control that isolates
context-dependence to content.

REGISTERED PREDICTIONS:
  (0) SANITY: at full context, topic decode ~0.85 (§870), class CE ~0.75 / within CE ~2.48 (§829);
      shuffled-label decode ~ chance at all C;
  (a) CONTENT NEEDS CONTEXT, GRAMMAR DOESN'T: topic decodability and content(within) CE improve strongly
      with context length (monotone, still rising at C=64+), while grammar(class) CE is near-flat after a
      few tokens -> content is a context-aggregate, grammar is local. Quantify: content-CE drop from C=4 to
      C=256 >> grammar-CE drop;
  (b) if grammar improves as much as content with context, grammar is not local (report honestly); if topic
      decodability is flat in C, topic is not context-aggregated (contradicting §870)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'topic_context_length_results.json'
CONTENT_L = 15; NLAYER = 18
NEVAL = 220; RTOK = 64; RPOS = 32; K = 12
CTXS = [2, 4, 8, 16, 32, 64, 128, 256]
QPOS = list(range(160, 256, 6))   # query positions with enough left-context to vary
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


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def forward_full(idx):
    """returns (L15 residual, final logits) for the whole batch."""
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
    # ---- full-context pass: define topic labels + full-context residual at the query positions ----
    fullR = []; seqs = []
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :256].to(DEV).contiguous(); r, _ = forward_full(idx); fullR.append(r.cpu()); seqs.append(idx.cpu().numpy())
    fullR = torch.cat(fullR, 0); S = np.concatenate(seqs, 0)                 # (Nseq, 256, D), (Nseq, 256)
    Nseq = S.shape[0]
    allR = fullR.reshape(-1, D).to(DEV); toks = S.reshape(-1); pos = np.broadcast_to(np.arange(256), S.shape).reshape(-1)
    Utok, g = mean_subspace(allR, toks, RTOK); Upos, _ = mean_subspace(allR, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (allR-g) - ((allR-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic_all = kmeans(cn, K).cpu().numpy().reshape(Nseq, 256)               # full-context topic label per (seq,pos)
    # train the topic probe on FULL-context residuals at query positions (fixed probe, applied to all C)
    qp = np.array(QPOS)
    # ---- for each context length, sliding-window forward and collect query-position residual + logits ----
    rng = np.random.RandomState(0); order = rng.permutation(Nseq); ntr = int(0.7*Nseq); tr_s, te_s = order[:ntr], order[ntr:]
    res = {}
    # precompute train/test topic labels at query positions
    ytr = topic_all[tr_s][:, qp].reshape(-1); yte = topic_all[te_s][:, qp].reshape(-1)
    tshuf = topic_all.copy().reshape(-1); rng.shuffle(tshuf); tshuf = tshuf.reshape(Nseq, 256)
    for C in CTXS:
        Rq = torch.zeros(Nseq, len(qp), D)
        tc = tw = 0.0; ntok = 0
        for i in range(0, Nseq, 4):
            bb = rows[i:i+4, :256].to(DEV)
            for qi, q in enumerate(qp):
                lo = max(0, q-C+1); win = bb[:, lo:q+1].contiguous()          # last C tokens ending at q
                r, lg = forward_full(win)
                Rq[i:i+bb.shape[0], qi] = r[:, -1].cpu()                       # residual at the query token
                # CE for predicting token q+1 from position q (last position of window)
                if q+1 < 256:
                    logp = F.log_softmax(lg[:, -1].float(), -1); tgt = bb[:, q+1]
                    pcl = logp.exp() @ Cmat; tcl = cidx[tgt]
                    lp_tok = logp[torch.arange(bb.shape[0], device=DEV), tgt]
                    lp_cls = (pcl[torch.arange(bb.shape[0], device=DEV), tcl] + 1e-12).log()
                    tc += float((-lp_cls).sum()); tw += float((-(lp_tok - lp_cls)).sum()); ntok += bb.shape[0]
        Ftr = Rq[tr_s].reshape(-1, D).to(DEV); Fte = Rq[te_s].reshape(-1, D).to(DEV)
        acc = decode_acc(Ftr, ytr, Fte, yte, K)
        accs = decode_acc(Ftr, tshuf[tr_s][:, qp].reshape(-1), Fte, tshuf[te_s][:, qp].reshape(-1), K)
        res[C] = {'topic_decode': round(acc, 3), 'shuffled_decode': round(accs, 3),
                  'class_ce': round(tc/max(ntok, 1), 3), 'within_ce': round(tw/max(ntok, 1), 3)}
        print(f"C={C:>3}: topic-decode {res[C]['topic_decode']} (shuf {res[C]['shuffled_decode']}) | class-CE {res[C]['class_ce']} | within-CE {res[C]['within_ce']}", flush=True)
    lo, hi = CTXS[0], CTXS[-1]
    out = {'ctxs': CTXS, 'query_positions': QPOS, 'k': K, 'chance': round(1.0/K, 3), 'by_ctx': res,
           'grammar_ce_drop': round(res[lo]['class_ce'] - res[hi]['class_ce'], 3),
           'content_ce_drop': round(res[lo]['within_ce'] - res[hi]['within_ce'], 3),
           'topic_decode_gain': round(res[hi]['topic_decode'] - res[lo]['topic_decode'], 3), 'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_content_needs_context'] = bool(out['content_ce_drop'] > 2*max(out['grammar_ce_drop'], 1e-6) and out['topic_decode_gain'] > 0.1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\ngrammar-CE drop (C{lo}->C{hi}) {out['grammar_ce_drop']} | content-CE drop {out['content_ce_drop']} | topic-decode gain {out['topic_decode_gain']}", flush=True)
    print(f"(a) content needs context, grammar local: {out['pred_a_content_needs_context']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
