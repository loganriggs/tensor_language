"""DO predictive-grammar and topic use the SAME attention heads or DIFFERENT ones? (§918: attention builds
BOTH predictive next-class grammar and topic; §875: topic is distributed across heads). Attribute each head's
additive residual contribution (c_proj(y_head)) and decode from it BOTH the next-token CLASS (predictive
grammar) and the TOPIC. Per head, is it a grammar head, a topic head, both, or neither? Correlate the per-head
grammar-score and topic-score across heads.

REGISTERED PREDICTIONS:
  (0) SANITY: full attention output decodes both next-class and topic above chance;
  (a) SPECIALIZATION vs SHARED: report the per-head (next-class decode, topic decode) and their correlation
      across heads. If correlation is HIGH, the same heads do both (shared context aggregation); if LOW/negative,
      heads SPECIALIZE (grammar heads vs topic heads);
  (b) name any strongly-specialized heads."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'grammar_vs_topic_heads_results.json'
CONTENT_L = 15; NEVAL = 200; RTOK = 64; RPOS = 32; K = 12; RIDGE = 1e2
LAYERS = [2, 5, 8, 11, 14]
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
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def acc(F_, y, valid, ncls):
    vi = np.where(valid)[0]; rng = np.random.RandomState(1); rng.shuffle(vi); ntr = int(0.7*len(vi)); a, b = vi[:ntr], vi[ntr:]
    Y = torch.zeros(len(a), ncls, device=DEV); Y[torch.arange(len(a)), torch.tensor(y[a], device=DEV)] = 1.0
    A = F_[a].T @ F_[a] + RIDGE*torch.eye(F_.shape[1], device=DEV); W = torch.linalg.solve(A, F_[a].T @ Y)
    return float(((F_[b] @ W).argmax(1).cpu().numpy() == y[b]).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    nh = m.transformer.h[0].attn.n_head; hd = m.transformer.h[0].attn.head_dim
    ybuf = {L: [] for L in LAYERS}; c15 = []; seqs = []; hs = []
    for L in LAYERS:
        def mkpre(L):
            def pre(mo, a_): ybuf[L].append(a_[0].detach().float().reshape(-1, nh*hd))
            return pre
        hs.append(m.transformer.h[L].attn.c_proj.register_forward_pre_hook(mkpre(L)))
    def hc(mo, i_, o_): c15.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hs.append(m.transformer.h[CONTENT_L].register_forward_hook(hc))
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    for h in hs: h.remove()
    Y = {L: torch.cat(ybuf[L], 0) for L in LAYERS}; R15 = torch.cat(c15, 0)
    S = np.concatenate(seqs, 0); toks = S.reshape(-1); pos = np.broadcast_to(np.arange(S.shape[1]), S.shape).reshape(-1)
    nxt = np.full_like(S, -1); nxt[:, :-1] = S[:, 1:]; nxtlab = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxt.reshape(-1)]); validn = nxtlab >= 0
    Utok, g = mean_subspace(R15, toks, RTOK); Upos, _ = mean_subspace(R15, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R15-g) - ((R15-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9); topic = kmeans(cn, K).cpu().numpy()
    gscores = []; tscores = []; out = {'n_head': nh, 'layers': {}}
    for L in LAYERS:
        Wp = m.transformer.h[L].attn.c_proj.weight.detach().float(); y = Y[L]; row = []
        for hh in range(nh):
            yh = torch.zeros_like(y); yh[:, hh*hd:(hh+1)*hd] = y[:, hh*hd:(hh+1)*hd]; contr = yh @ Wp.T
            gsc = acc(contr, nxtlab, validn, len(CLASSES)); tsc = acc(contr, topic, np.ones(len(topic), bool), K)
            row.append({'head': hh, 'grammar_nextclass': round(gsc, 3), 'topic': round(tsc, 3)}); gscores.append(gsc); tscores.append(tsc)
        out['layers'][f'L{L}'] = row
        print(f"L{L}: " + " ".join(f"h{r['head']}(g{r['grammar_nextclass']:.2f}/t{r['topic']:.2f})" for r in row), flush=True)
    corr = float(np.corrcoef(gscores, tscores)[0, 1])
    out['grammar_topic_head_corr'] = round(corr, 3)
    out['pred_shared'] = bool(corr > 0.5); out['pred_specialized'] = bool(corr < 0.2)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nper-head grammar vs topic decode correlation = {corr:.3f} (high=same heads do both; low=specialized)", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
