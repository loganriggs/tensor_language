"""Are the two machines SEPARABLE or ENTANGLED? Attention is a shared context-engine building BOTH predictive
grammar (next-class) and topic, on mostly-shared heads (§918/§919). Do they live in ORTHOGONAL subspaces
(cleanly separable — you could edit one without the other) or OVERLAPPING ones (entangled)? Measure (a) the
subspace overlap between the next-class-grammar subspace and the topic subspace of the L15 residual, vs chance;
(b) causally: does INTERCHANGE-patching the topic subspace change the next-CLASS prediction (cross-talk), and
vice versa — patching the class subspace change the topic prediction?

REGISTERED PREDICTIONS:
  (0) SANITY: both subspaces decode their own target well;
  (a) SEPARABLE: grammar and topic subspaces are near-ORTHOGONAL (overlap ~ chance) AND cross-patching has
      little effect (patching topic barely moves next-class; patching class barely moves topic) -> the two
      machines are separable, sharing the attention engine but writing to orthogonal subspaces;
  (b) if overlap is high and cross-patching leaks, they are ENTANGLED (report which)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'grammar_topic_interference_results.json'
CONTENT_L = 15; NEVAL = 240; QP = 128; RTOK = 64; RPOS = 32; K = 12; RCLASS = 8
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
PATCH = {'on': False, 'U': None, 'src': None}


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
    y = o_[0] if isinstance(o_, tuple) else o_; U = PATCH['U']; b = y[:, QP, :]
    y = y.clone(); y[:, QP, :] = b - (b @ U) @ U.T + PATCH['src']
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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :257].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    Rs = []
    for i in range(0, nb, 4): Rs.append(capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous()).reshape(-1, D))
    R = torch.cat(Rs, 0); toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(256), (nb, 256)).reshape(-1)
    nxt = np.full_like(S[:, :-1], -1); nxt[:, :-1] = S[:, 1:-1]
    nxtlab = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxt.reshape(-1)])
    # class (next-class) subspace and topic subspace of R
    Uclass, gc = mean_subspace(R, nxtlab, RCLASS)
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy(); Utopic, _ = mean_subspace(content, topic, K-1)
    # subspace overlap
    r = min(Uclass.shape[1], Utopic.shape[1]); overlap = round(float((Uclass.T @ Utopic).pow(2).sum()/r), 3); chance = round((RCLASS)/D, 3)
    out = {'grammar_topic_subspace_overlap': overlap, 'chance': chance, 'runtime_s': None}
    print(f"grammar (next-class) vs topic subspace overlap = {overlap} (chance {chance})", flush=True)
    # cross-patch causality: patch topic -> does next-class prediction move? patch class -> does topic move?
    Rqp = R.reshape(nb, 256, D)[:, QP, :]
    idxb = blocks[:32, :-1].to(DEV).contiguous()
    hh = m.transformer.h[CONTENT_L].register_forward_hook(patch_hook)
    rng = np.random.RandomState(0); src = Rqp[torch.tensor(rng.permutation(nb)[:32], device=DEV)]
    def predclass_entropy(U, srcv):
        PATCH['U'] = U; PATCH['src'] = (srcv @ U) @ U.T; PATCH['on'] = True
        lg = forward_logits(idxb).float()[:, QP, :]; PATCH['on'] = False
        return lg
    PATCH['on'] = False; base = forward_logits(idxb).float()[:, QP, :]
    lg_topic = predclass_entropy(Utopic, src); lg_class = predclass_entropy(Uclass, src)
    hh.remove()
    # measure change in top-1 predicted token's CLASS distribution (grammar) vs the token's topic-ness
    def class_shift(lg):  # mean L1 change in class-marginal of the prediction
        V = int(m.lm_head.weight.shape[0]); tok2c = np.full(V, 7)
        for tid in np.unique(S.reshape(-1)): tok2c[int(tid)] = CLASSES.index(classify(d(int(tid))))
        C = F.one_hot(torch.tensor(tok2c, device=DEV), 8).float()
        pb = F.softmax(base, -1) @ C; pp = F.softmax(lg, -1) @ C
        return round(float((pp-pb).abs().sum(-1).mean()), 4)
    out['patch_topic_class_shift'] = class_shift(lg_topic)   # patching topic -> change in class prediction (cross-talk)
    out['patch_class_class_shift'] = class_shift(lg_class)    # patching class -> change in class prediction (own effect)
    out['crosstalk_ratio'] = round(out['patch_topic_class_shift']/max(out['patch_class_class_shift'], 1e-6), 2)
    out['runtime_s'] = round(time.time()-t0, 1)
    out['pred_a_separable'] = bool(overlap < 3*chance and out['crosstalk_ratio'] < 0.5)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"patch TOPIC -> class-prediction shift {out['patch_topic_class_shift']} (cross-talk) vs patch CLASS -> {out['patch_class_class_shift']} (own); ratio {out['crosstalk_ratio']}", flush=True)
    print(f"(a) machines separable (orthogonal subspaces + low cross-talk): {out['pred_a_separable']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
