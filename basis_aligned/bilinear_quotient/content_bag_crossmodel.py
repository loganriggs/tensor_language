"""IS THE BAG-OF-WORDS CONTENT MECHANISM UNIVERSAL? §932/§934: in bilin18 the late-layer topic is ~an
order-invariant BAG-of-word-embeddings running mean of context (bag decodes topic ~0.66 vs current-token ~0.22),
while grammar is local. Is that a general LM property (like the loss budget §880, the 23/77 split §831, and the
grammar⊥topic separability §925)? For GPT-2 and GPT-2-large, at a late layer define topics from the content
residual (token+pos stripped), then decode the topic from (i) the current-token embedding vs (ii) a causal
bag-of-words running mean of the input embeddings.

CAVEAT: GPT-2 is WebText-trained (slightly OOD on FineWeb); the geometry test (a within-model fact) is robust.

REGISTERED PREDICTIONS:
  (0) SANITY: topic decodes above base from the bag in each model; shuffled-label null ~ base.
  (a) UNIVERSAL BAG: in BOTH GPT-2 models the BAG decodes topic well above the CURRENT-token embedding
      (bag - current > 0.15) -> content is an order-invariant bag-of-words gist in general LMs, as in bilin18;
  (b) report bag vs current topic-decode per model + the bilin18 reference (0.655 vs 0.216)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import DEV
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_bag_crossmodel_results.json'
NEVAL = 160; SEQ = 256; K = 32; RTOK = 64; RPOS = 32; RIDGE = 1e2
MODELS = ['gpt2', 'gpt2-large']


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
    return float((F_[te] @ W).argmax(1).cpu().numpy().__eq__(y[te]).mean())


@torch.no_grad()
def run(mid, blocks, S):
    mdl = AutoModelForCausalLM.from_pretrained(mid).to(DEV).eval()
    Dm = mdl.config.n_embd; L = int(0.8*mdl.config.n_layer)
    reps = []
    def h(mo, i_, o_): reps.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, Dm))
    hh = mdl.transformer.h[L].register_forward_hook(h)
    embs = []
    for i in range(0, blocks.shape[0], 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1]
        mdl(idx); embs.append(mdl.transformer.wte(idx).detach().float())
    hh.remove(); R = torch.cat(reps, 0)
    T = SEQ-1; nb = blocks.shape[0]
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(T), (nb, T)).reshape(-1)
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy()
    E = torch.cat(embs, 0)  # (nb,T,Dm)
    feat_current = E.reshape(-1, Dm)
    num = torch.cumsum(E, 1); den = torch.arange(1, T+1, device=DEV, dtype=E.dtype).view(1, T, 1)
    feat_bag = (num/den).reshape(-1, Dm)
    n = R.shape[0]; rng = np.random.RandomState(0); perm = rng.permutation(n); ntr = int(0.7*n); tr, te = perm[:ntr], perm[ntr:]
    base = float(np.bincount(topic, minlength=K).max()/len(topic))
    tsh = topic.copy(); rng.shuffle(tsh)
    r = {'layer': L, 'base': round(base, 4), 'bag': round(acc(feat_bag, topic, K, tr, te), 4),
         'current': round(acc(feat_current, topic, K, tr, te), 4), 'null': round(acc(feat_bag, tsh, K, tr, te), 4)}
    r['bag_minus_current'] = round(r['bag'] - r['current'], 4)
    del mdl; torch.cuda.empty_cache(); return r


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy()
    out = {'bilin18_ref': {'bag': 0.655, 'current': 0.216}, 'models': {}}
    for mid in MODELS:
        print(f"loading {mid}...", flush=True); r = run(mid, blocks, S); out['models'][mid] = r
        print(f"{mid} L{r['layer']}: bag {r['bag']:.3f} vs current {r['current']:.3f} (base {r['base']:.3f}, null {r['null']:.3f}) -> bag-current {r['bag_minus_current']:+.3f}", flush=True)
    out['pred_a_universal_bag'] = bool(all(out['models'][mid]['bag_minus_current'] > 0.15 for mid in MODELS))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) bag-of-words content universal (bag>>current in all): {out['pred_a_universal_bag']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
