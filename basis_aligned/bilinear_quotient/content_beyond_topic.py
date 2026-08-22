"""WHAT STRUCTURE lives INSIDE the high-rank content residual BEYOND topic? (content-frontier probe). The
content machine is high-rank and topic-organized (§866), but topic is only a slice (§908/§922). Probe what
OTHER, constructible structure the content residual carries: sentence-position (tokens since last sentence
end), is-repeat (entity/coreference — current token seen earlier), paren/quote parity (nesting state),
prev-token class (local syntax). Decode each from the content residual (R − class/pos projection), and ALSO
from content-with-the-topic-subspace-removed, to see which structure is NON-topic (survives topic removal).
vs shuffled-label null. This names more of the content machine beyond topic.

REGISTERED PREDICTIONS:
  (0) SANITY: each feature decodes above its base rate; shuffled-label ~ base rate;
  (a) CONTENT CARRIES NON-TOPIC STRUCTURE: sentence-position, is-repeat (entity), and nesting state are
      decodable from the content residual well above base rate AND largely SURVIVE removing the topic subspace
      (they are non-topic content structure) -> the content machine tracks more than topic: sentence/discourse
      position, entity repetition, nesting;
  (b) report decodability + topic-removal drop per feature."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_beyond_topic_results.json'
CONTENT_L = 15; NEVAL = 200; SEQ = 256; RTOK = 64; RPOS = 32; K = 12; RTOPIC = 24; RIDGE = 1e2
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


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capL(idx):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D)
    hh = m.transformer.h[CONTENT_L].register_forward_hook(h); forward_logits(idx); hh.remove(); return cap['r']


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed); c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def acc(F_, y, ncls, seed=0):
    n = F_.shape[0]; rng = np.random.RandomState(seed); idx = rng.permutation(n); ntr = int(0.7*n); a, b = idx[:ntr], idx[ntr:]
    Y = torch.zeros(len(a), ncls, device=DEV); Y[torch.arange(len(a)), torch.tensor(y[a], device=DEV)] = 1.0
    A = F_[a].T @ F_[a] + RIDGE*torch.eye(F_.shape[1], device=DEV); W = torch.linalg.solve(A, F_[a].T @ Y)
    return float(((F_[b] @ W).argmax(1).cpu().numpy() == y[b]).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    Rs = []
    for i in range(0, nb, 4): Rs.append(capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous()))
    R = torch.cat(Rs, 0); toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy(); Utopic, _ = mean_subspace(content, topic, RTOPIC)
    content_notopic = content - (content @ Utopic) @ Utopic.T
    # constructible features
    ss = S[:, :-1]; senti = d
    def sentpos(r):
        out = np.zeros(SEQ-1, np.int64); c = 0
        for p in range(SEQ-1):
            tk = d(int(ss[r, p])).strip(); out[p] = min(c, 7); c = 0 if (tk[:1] in '.!?' or tk == '') else c+1
        return out
    sent = np.concatenate([sentpos(r) for r in range(nb)])
    isrep = np.zeros((nb, SEQ-1), np.int64)
    for r in range(nb):
        seen = set()
        for p in range(SEQ-1):
            t = int(ss[r, p]); isrep[r, p] = 1 if t in seen else 0; seen.add(t)
    isrep = isrep.reshape(-1)
    def paren(r):
        out = np.zeros(SEQ-1, np.int64); depth = 0
        for p in range(SEQ-1):
            tk = d(int(ss[r, p])); depth += tk.count('(') - tk.count(')'); out[p] = min(max(depth, 0), 3)
        return out
    pdep = np.concatenate([paren(r) for r in range(nb)])
    prevcls = np.full((nb, SEQ-1), 7, np.int64)
    for r in range(nb):
        for p in range(1, SEQ-1): prevcls[r, p] = CLASSES.index(classify(d(int(ss[r, p-1]))))
    prevcls = prevcls.reshape(-1)
    feats = {'sentence_position': (sent, 8), 'is_repeat_entity': (isrep, 2), 'paren_depth': (pdep, 4), 'prev_token_class': (prevcls, 8)}
    rng = np.random.RandomState(0)
    out = {'features': {}}
    for name, (y, nc) in feats.items():
        base = float(np.bincount(y, minlength=nc).max()/len(y))
        a_full = acc(content, y, nc); a_notop = acc(content_notopic, y, nc)
        ysh = y.copy(); rng.shuffle(ysh); a_sh = acc(content, ysh, nc)
        out['features'][name] = {'decode_content': round(a_full, 3), 'decode_content_no_topic': round(a_notop, 3),
                                 'base_rate': round(base, 3), 'shuffled': round(a_sh, 3),
                                 'topic_removal_drop': round(a_full - a_notop, 3), 'above_base': round(a_full - base, 3)}
        print(f"{name:>18}: content {a_full:.3f} | no-topic {a_notop:.3f} | base {base:.3f} | shuf {a_sh:.3f} | topic-drop {a_full-a_notop:+.3f}", flush=True)
    out['pred_a_nontopic_structure'] = bool(all(out['features'][f]['above_base'] > 0.05 and out['features'][f]['decode_content_no_topic'] > out['features'][f]['base_rate'] + 0.03 for f in ['sentence_position', 'is_repeat_entity']))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\n(a) content carries NON-topic structure (survives topic removal): {out['pred_a_nontopic_structure']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
