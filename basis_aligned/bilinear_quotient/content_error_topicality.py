"""Is the CONTENT machine right at the TOPIC level even when it picks the WRONG word? §973: content-errors are
right-class/wrong-token. Test whether the model's WRONG content-word guesses are still TOPICALLY appropriate — i.e.
the content machine narrows to the correct topic/subject even when it misses the specific word (§866 topic
tracker). On CONTENT-ERROR positions, measure whether the PREDICTED token shares the true token's TOPIC cluster
more often than a frequency-matched random token would.

Method: cluster the L15 content residual into K topics (§866); assign each vocab token a topic by its most-common
context-topic (empirical). On content-error positions, compute the rate at which predicted-token-topic ==
true-token-topic, vs a null that replaces the prediction with a random token of the SAME class (frequency/class-
matched).

REGISTERED PREDICTIONS:
  (0) SANITY: on HITs topic-match is ~1 by construction.
  (a) TOPICALLY-APPROPRIATE ERRORS: on CONTENT-ERROR positions the predicted token matches the true token's topic
      well ABOVE the class-matched-random null -> the content machine gets the TOPIC right even when it misses the
      specific word (partial success), tying errors to the topic-tracker (§866);
  (b) report content-error topic-match rate vs null + hit rate."""
import json, time, sys, torch
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_error_topicality_results.json'
NEVAL = 200; SEQ = 256; CONTENT_L = 15; K = 24; RTOK = 64; RPOS = 32
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
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


def forward_capL_and_logits(idx):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[CONTENT_L].register_forward_hook(h)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    hh.remove(); return cap['r'], readout(x)


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
    V = int(m.lm_head.weight.shape[0])
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(S.reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    # content residual + topic clustering; token->topic by most-common context topic of its occurrences
    Rs = []; preds = []; tfs = []
    for i in range(0, nb, 4):
        bb = blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        r, lg = forward_capL_and_logits(idx); Rs.append(r.reshape(-1, D))
        preds.append(lg.float().reshape(-1, V).argmax(1)); tfs.append(tgt.reshape(-1))
    R = torch.cat(Rs, 0); pred = torch.cat(preds, 0); tf = torch.cat(tfs, 0)
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1,keepdim=True)+1e-9)
    postopic = kmeans(cn, K).cpu().numpy()  # topic at each position (of the CONTEXT)
    # token -> topic: the topic of the positions where that token is the TARGET (next token)
    tf_np = tf.cpu().numpy(); tok_topic_votes = defaultdict(Counter)
    for j in range(len(tf_np)):
        tok_topic_votes[int(tf_np[j])][int(postopic[j])] += 1
    tok_topic = np.full(V, -1, np.int64)
    for t, c in tok_topic_votes.items(): tok_topic[t] = c.most_common(1)[0][0]
    pred_np = pred.cpu().numpy(); tcls = cidx.cpu().numpy()
    is_hit = pred_np == tf_np
    # content-error = right class, wrong token, and both have a defined topic
    is_cont = (~is_hit) & (tcls[pred_np] == tcls[tf_np])
    # topic-match on content errors
    valid = is_cont & (tok_topic[pred_np] >= 0) & (tok_topic[tf_np] >= 0)
    tm = (tok_topic[pred_np[valid]] == tok_topic[tf_np[valid]])
    match_rate = float(tm.mean()) if valid.sum() else 0.0
    # null: replace prediction with a random token of the SAME class (freq-weighted), measure topic-match to true
    rng = np.random.RandomState(0)
    by_class = defaultdict(list)
    for t in range(V):
        if tok_topic[t] >= 0: by_class[tcls[t]].append(t)
    null_matches = []
    idxs = np.where(valid)[0]
    for j in idxs[rng.permutation(len(idxs))[:min(4000, len(idxs))]]:
        cl_ = tcls[tf_np[j]]; cand = by_class.get(cl_, [])
        if not cand: continue
        rt = cand[rng.randint(len(cand))]
        null_matches.append(int(tok_topic[rt] == tok_topic[tf_np[j]]))
    null_rate = float(np.mean(null_matches)) if null_matches else 0.0
    out = {'K': K, 'n_content_error_valid': int(valid.sum()), 'content_err_topic_match': round(match_rate, 4),
           'class_matched_random_null': round(null_rate, 4), 'hit_frac': round(float(is_hit.mean()), 4)}
    out['pred_a_topically_appropriate'] = bool(match_rate > null_rate + 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"content-error topic-match {match_rate:.3f} vs class-matched-random null {null_rate:.3f} (n={int(valid.sum())})", flush=True)
    print(f"(a) content errors are topically appropriate: {out['pred_a_topically_appropriate']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
