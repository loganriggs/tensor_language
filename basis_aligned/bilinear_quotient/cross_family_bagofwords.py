"""Does the CONTENT MECHANISM — topic ~ a bag-of-word-embeddings running mean of context (§932) — generalize
across the model FAMILY? §937 confirmed it in GPT-2; test siblings swiglu18 and bilin12 (+ bilin18 ref). Decode
the late-layer topic (K=32) from a causal bag-of-words running mean of the input embeddings vs the current-token
embedding. Residual-only (no output transform), low-risk. Pairs with cross_family_separability (structure) to
show the two-machine account's MECHANISM also travels across the family.

REGISTERED PREDICTIONS:
  (0) SANITY: topic decodes above base from the bag in each model.
  (a) FAMILY-WIDE BAG: in swiglu18 and bilin12 the bag decodes topic well above the current-token embedding
      (bag - current > 0.15), as in bilin18 (§932) and GPT-2 (§937) -> content is an order-invariant bag-of-words
      gist across the family;
  (b) report bag vs current topic-decode per model + bilin18 ref."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
import census_lib as cl
from bilin18_joint_removal import m as BILIN, DEV
from tier2_model import load_elriggs
import torch.nn.functional as F

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_family_bagofwords_results.json'
NEVAL = 140; SEQ = 256; K = 32; RTOK = 64; RPOS = 32; RIDGE = 1e2


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
def cap_late(mdl, idx, Dm, L):
    reps = {}
    def h(mo, i_, o_): reps['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = mdl.transformer.h[L].register_forward_hook(h)
    x = F.rms_norm(mdl.transformer.wte(idx), (Dm,)); x0 = x; v1 = None
    for blk in mdl.transformer.h: x, v1 = blk(x, v1, x0)
    hh.remove(); return reps['r']  # (b,T,Dm)


@torch.no_grad()
def run(mdl, blocks, S, Dm, nlayer):
    L = int(0.8*nlayer); nb = blocks.shape[0]; T = SEQ-1
    reps = []; embs = []
    for i in range(0, nb, 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous()
        reps.append(cap_late(mdl, idx, Dm, L).reshape(-1, Dm))
        embs.append(F.rms_norm(mdl.transformer.wte(idx), (Dm,)).float())
    R = torch.cat(reps, 0)
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
    bag = acc(feat_bag, topic, K, tr, te); cur = acc(feat_current, topic, K, tr, te)
    return {'layer': L, 'base': round(base, 4), 'bag': round(bag, 4), 'current': round(cur, 4), 'bag_minus_current': round(bag-cur, 4)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy()
    out = {'bilin18_ref_932': {'bag': 0.655, 'current': 0.216}, 'gpt2_ref_937': {'bag_minus_current': 0.247}, 'models': {}}
    r = run(BILIN, blocks, S, 1152, 18); out['models']['bilin18'] = r; print(f"bilin18: {r}", flush=True)
    for short in ['swiglu18', 'bilin12']:
        try:
            mdl, cfg = load_elriggs(short); Dm = cfg.get('n_embd'); nl = cfg.get('n_layer')
            r = run(mdl, blocks, S, Dm, nl); out['models'][short] = r; print(f"{short}: {r}", flush=True)
            del mdl; torch.cuda.empty_cache()
        except Exception as e:
            out['models'][short] = {'error': f'{type(e).__name__}: {e}'}; print(f"{short} ERROR {e}", flush=True)
    ok = all('bag_minus_current' in out['models'][k] and out['models'][k]['bag_minus_current'] > 0.15 for k in ['swiglu18', 'bilin12'] if k in out['models'])
    out['pred_a_family_bag'] = bool(ok)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) content-as-bag family-wide (bag>>current all): {ok}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
