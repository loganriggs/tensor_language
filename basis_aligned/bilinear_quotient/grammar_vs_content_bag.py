"""DOUBLE DISSOCIATION: grammar is LOCAL (current-token-driven), content is a BAG-of-words gist. Using the same
cheap probe as §932, decode the next-token CLASS (grammar target) and the L15 TOPIC (content target) each from
two features of the raw input embeddings: (i) the CURRENT-token embedding only (local), and (ii) a causal
BAG-of-words running mean of the context (order-invariant gist). The two-machine account (grammar low-rank /
local / token-driven vs content high-rank / long-range / bag) predicts a clean double dissociation.

REGISTERED PREDICTIONS:
  (0) SANITY: both targets decode above their base rate from at least one feature; shuffled-label nulls ~ base.
  (a) DOUBLE DISSOCIATION: next-CLASS is decoded well by the CURRENT-token embedding and the bag adds little
      (bag_class - current_class small or negative); TOPIC is decoded well by the BAG and poorly by the current
      token (bag_topic >> current_topic, per §932). Quantify with a dissociation index
      = (current_class - bag_class) + (bag_topic - current_topic) > 0.3;
  (b) report the 2x2 decode table (feature x target) + nulls."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'grammar_vs_content_bag_results.json'
CONTENT_L = 15; NEVAL = 200; SEQ = 256; RTOK = 64; RPOS = 32; K = 32; RIDGE = 1e2; RCLASS = 8
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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    idxfull = blocks[:, :-1].contiguous(); T = SEQ-1
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(T), (nb, T)).reshape(-1)
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    # L15 topic labels
    R = []
    for i in range(0, nb, 4): R.append(forward_capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous()).reshape(-1, D))
    R = torch.cat(R, 0)
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy()
    # features from raw embeddings
    E = F.rms_norm(m.transformer.wte(idxfull.to(DEV)), (D,)).float()
    feat_current = E.reshape(-1, D)
    num = torch.cumsum(E, dim=1); den = torch.arange(1, T+1, device=DEV, dtype=E.dtype).view(1, T, 1)
    feat_bag = (num/den).reshape(-1, D)
    n = R.shape[0]; rng = np.random.RandomState(0); perm = rng.permutation(n); ntr = int(0.7*n); tr, te = perm[:ntr], perm[ntr:]
    def acc(Feat, y, ncls, mask=None):
        yy = y.copy()
        trm = tr if mask is None else tr[mask[tr]]; tem = te if mask is None else te[mask[te]]
        Y = torch.zeros(len(trm), ncls, device=DEV); Y[torch.arange(len(trm)), torch.tensor(yy[trm], device=DEV)] = 1.0
        A = Feat[trm].T @ Feat[trm] + RIDGE*torch.eye(Feat.shape[1], device=DEV); W = torch.linalg.solve(A, Feat[trm].T @ Y)
        return float((Feat[tem] @ W).argmax(1).cpu().numpy().__eq__(yy[tem]).mean())
    clsmask = nxtcls >= 0
    topic_base = float(np.bincount(topic, minlength=K).max()/len(topic))
    class_base = float(np.bincount(nxtcls[clsmask], minlength=RCLASS).max()/clsmask.sum())
    tab = {
        'current_topic': round(acc(feat_current, topic, K), 4),
        'bag_topic': round(acc(feat_bag, topic, K), 4),
        'current_class': round(acc(feat_current, np.where(clsmask, nxtcls, 0), RCLASS, clsmask), 4),
        'bag_class': round(acc(feat_bag, np.where(clsmask, nxtcls, 0), RCLASS, clsmask), 4),
    }
    tsh = topic.copy(); rng.shuffle(tsh); csh = nxtcls.copy(); rng.shuffle(csh)
    nulls = {'topic_null': round(acc(feat_bag, tsh, K), 4), 'class_null': round(acc(feat_current, np.where(clsmask, csh, 0), RCLASS, clsmask), 4)}
    diss = (tab['current_class'] - tab['bag_class']) + (tab['bag_topic'] - tab['current_topic'])
    out = {'table': tab, 'nulls': nulls, 'topic_base': round(topic_base, 4), 'class_base': round(class_base, 4),
           'dissociation_index': round(diss, 4), 'pred_a_double_dissociation': bool(diss > 0.3 and tab['bag_topic'] > tab['current_topic'] and tab['current_class'] >= tab['bag_class'] - 0.02),
           'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"           TOPIC(base {topic_base:.3f})   CLASS(base {class_base:.3f})", flush=True)
    print(f"current:   {tab['current_topic']:.3f}          {tab['current_class']:.3f}", flush=True)
    print(f"bag:       {tab['bag_topic']:.3f}          {tab['bag_class']:.3f}", flush=True)
    print(f"nulls: {nulls} | dissociation index {diss:+.3f}", flush=True)
    print(f"(a) double dissociation (grammar=local, content=bag): {out['pred_a_double_dissociation']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
