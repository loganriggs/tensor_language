"""IS ATTENTION'S CONTENT AGGREGATION JUST A CONTENT-WORD BAG-OF-EMBEDDINGS RUNNING MEAN? §929/§930: the content
machine (topic tracker) is built gradually by long-range attention (§862/§871), and §872 showed the topic gist
is a CONTENT-WORD, ORDER-INVARIANT thing. If so, the L15 content representation should be largely reconstructable
from a simple CAUSAL RUNNING MEAN of the preceding token EMBEDDINGS — i.e. attention is approximately averaging
context word-vectors. Test by decoding the L15 topic (K=32, defined from the real L15 content) from several
cheap running-mean features of the raw embeddings, and compare to the real-L15 ceiling and a null:
  - all-token running mean (bag of words), content-word-only running mean, function-word-only running mean,
    recency-weighted (exp decay) all-token mean, and current-token-embedding only (no aggregation).
This NAMES the aggregation mechanism if a content-word bag explains most of the topic.

REGISTERED PREDICTIONS:
  (0) SANITY: real-L15 content decodes topic ~0.85 (ceiling, §929); current-token-only and shuffled-label null
      are low; base rate ~0.14.
  (a) CONTENT-WORD BAG: a CONTENT-WORD running-mean of embeddings decodes L15 topic well above current-token-only
      and above the function-word running mean -> attention's content aggregation is approximately a content-word
      bag-of-embeddings average (order-invariant, §872);
  (b) RECENCY: recency-weighted mean is NOT much better than the flat running mean -> aggregation is long-range /
      order-invariant, not recency-dominated (§871/§872). Report topic-decode for each feature."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_bagofwords_results.json'
CONTENT_L = 15; NEVAL = 200; SEQ = 256; RTOK = 64; RPOS = 32; K = 32; RIDGE = 1e2; DECAY = 0.95
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
FUNCTION = {'det', 'prep', 'conj', 'pron', 'punct'}; CONTENTC = {'word', 'cap', 'number'}
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}


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


def forward_capL(idx):
    cap = {}
    def ch(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[CONTENT_L].register_forward_hook(ch)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    hh.remove(); return cap['r']


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


def acc(F_, y, ncls, tr, te):
    Y = torch.zeros(len(tr), ncls, device=DEV); Y[torch.arange(len(tr)), torch.tensor(y[tr], device=DEV)] = 1.0
    A = F_[tr].T @ F_[tr] + RIDGE*torch.eye(F_.shape[1], device=DEV); W = torch.linalg.solve(A, F_[tr].T @ Y)
    return float(((F_[te] @ W).argmax(1).cpu().numpy() == y[te]).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    idxfull = blocks[:, :-1].contiguous()  # (nb, SEQ-1)
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    # real L15 content + topic labels
    R = []
    for i in range(0, nb, 4): R.append(forward_capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous()).reshape(-1, D))
    R = torch.cat(R, 0)
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy()
    # embeddings (raw wte, rms-normed as the model's input) per position
    E = F.rms_norm(m.transformer.wte(idxfull.to(DEV)), (D,)).float()  # (nb, SEQ-1, D)
    T = SEQ-1
    # class per position for content/function masks
    cls = np.zeros((nb, T), np.int64)
    for r in range(nb):
        for p in range(T): cls[r, p] = CLASSES.index(classify(d(int(S[r, p]))))
    cmask = torch.tensor(np.isin(cls, [CLASSES.index(c) for c in CONTENTC]), device=DEV, dtype=E.dtype)  # (nb,T)
    fmask = torch.tensor(np.isin(cls, [CLASSES.index(c) for c in FUNCTION]), device=DEV, dtype=E.dtype)
    # causal running means (inclusive of current position)
    def running_mean(weight=None, mask=None):
        w = torch.ones(nb, T, device=DEV, dtype=E.dtype) if weight is None else weight
        if mask is not None: w = w * mask
        num = torch.cumsum(E * w[:, :, None], dim=1); den = torch.cumsum(w, dim=1).clamp_min(1e-6)[:, :, None]
        return (num/den).reshape(-1, D)
    feat_all = running_mean()
    feat_content = running_mean(mask=cmask)
    feat_function = running_mean(mask=fmask)
    decw = DECAY ** (T-1 - torch.arange(T, device=DEV, dtype=E.dtype))  # more weight to recent (relative), causal via cumsum trick below
    # recency: weight_p over past = DECAY^(current-p); implement via cumulative with normalization per position
    # approximate with exponential: use reversed cumulative — do a simple loop-free EMA
    ema = torch.zeros_like(E); acc_e = torch.zeros(nb, D, device=DEV, dtype=E.dtype); acc_w = torch.zeros(nb, 1, device=DEV, dtype=E.dtype)
    for p in range(T):
        acc_e = DECAY*acc_e + E[:, p, :]; acc_w = DECAY*acc_w + 1.0; ema[:, p, :] = acc_e/acc_w.clamp_min(1e-6)
    feat_recency = ema.reshape(-1, D)
    feat_current = E.reshape(-1, D)
    n = R.shape[0]; rng = np.random.RandomState(0); perm = rng.permutation(n); ntr = int(0.7*n); tr, te = perm[:ntr], perm[ntr:]
    base = float(np.bincount(topic, minlength=K).max()/len(topic))
    tsh = topic.copy(); rng.shuffle(tsh)
    feats = {'real_L15_content(ceiling)': content, 'all_token_bag': feat_all, 'content_word_bag': feat_content,
             'function_word_bag': feat_function, 'recency_weighted': feat_recency, 'current_token_only': feat_current}
    out = {'base_rate': round(base, 4), 'K': K, 'topic_decode': {}}
    for name, Fm in feats.items():
        a = acc(Fm, topic, K, tr, te); out['topic_decode'][name] = round(a, 4)
        print(f"{name:>28}: topic-decode {a:.4f}", flush=True)
    out['shuffled_null'] = round(acc(content, tsh, K, tr, te), 4)
    td = out['topic_decode']
    out['pred_a_content_word_bag'] = bool(td['content_word_bag'] > td['current_token_only'] + 0.05 and td['content_word_bag'] > td['function_word_bag'] + 0.03)
    out['pred_b_order_invariant'] = bool(td['recency_weighted'] < td['all_token_bag'] + 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"shuffled null {out['shuffled_null']:.4f} | base {base:.4f}", flush=True)
    print(f"(a) content-word bag explains topic: {out['pred_a_content_word_bag']} | (b) order-invariant (recency~=flat): {out['pred_b_order_invariant']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
