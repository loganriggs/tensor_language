"""DOES THE CONTENT PIPELINE = FRONT-MLP PER-TOKEN ENCODING -> ATTENTION BAG-AVERAGE? §932: a bag-of-RAW-EMBEDDING
running mean recovers 72% of L15 topic-decode-above-base (0.655 vs ceiling 0.844). §933: the front MLPs write
the residual substrate. Hypothesis: attention pools not raw embeddings but the FRONT-MLP-ENCODED per-token
features; so a bag of POST-FRONT-MLP per-token features should recover MORE topic than the raw-embedding bag,
approaching the ceiling. Build a per-token feature TABLE by forwarding each unique token id as a length-1
sequence (no context -> its own nonlinear encoding) and capturing the residual after block L, for several L; then
causally bag-average those features over each context and decode the L15 topic (K=32, fixed from the real run).

REGISTERED PREDICTIONS:
  (0) SANITY: L=0 (embedding) feature bag reproduces §932's raw-embedding bag (~0.655); real L15 ceiling ~0.84.
  (a) FRONT-MLP ENCODING HELPS: a bag of POST-FRONT-MLP per-token features (L=2-4) recovers MORE topic than the
      raw-embedding bag (L=0) and moves toward the ceiling -> the content pipeline is front-MLP per-token
      encoding followed by attention bag-averaging;
  (b) report topic-decode of the feature-bag at each L vs raw-embed bag vs ceiling; note the L where it saturates.
  NULL: a bag of the per-token features with the token->feature map SHUFFLED recovers ~base."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_bag_mlpfeatures_results.json'
CONTENT_L = 15; NEVAL = 200; SEQ = 256; RTOK = 64; RPOS = 32; K = 32; RIDGE = 1e2
FEAT_LAYERS = [0, 2, 4, 7]


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def forward_capL(idx, capL=CONTENT_L):
    cap = {}
    def ch(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[capL].register_forward_hook(ch)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    hh.remove(); return cap['r']


@torch.no_grad()
def per_token_features(unique_ids, layers):
    """Forward each unique token id as a length-1 sequence; capture residual after each block in `layers`.
    Returns dict L -> tensor (U, D) plus the embedding (L=-1 handled by caller)."""
    U = unique_ids.shape[0]; out = {L: torch.zeros(U, D, device=DEV) for L in layers}
    B = 4096
    for s in range(0, U, B):
        idx = unique_ids[s:s+B].to(DEV).view(-1, 1)  # (b,1)
        caps = {}
        hooks = []
        for L in layers:
            def mk(L):
                def h(mo, i_, o_): caps[L] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()[:, 0, :]
                return h
            hooks.append(m.transformer.h[L].register_forward_hook(mk(L)))
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
        for h in hooks: h.remove()
        for L in layers: out[L][s:s+idx.shape[0]] = caps[L]
    return out


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
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]; T = SEQ-1
    idxctx = S[:, :-1]  # (nb, T)
    toks = idxctx.reshape(-1); pos = np.broadcast_to(np.arange(T), (nb, T)).reshape(-1)
    # real L15 content + topic labels (ceiling)
    R = []
    for i in range(0, nb, 4): R.append(forward_capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous()).reshape(-1, D))
    R = torch.cat(R, 0)
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy()
    # per-token feature tables
    uids = torch.tensor(np.unique(idxctx), dtype=torch.long)
    id2row = {int(t): i for i, t in enumerate(uids.tolist())}
    feats = per_token_features(uids, FEAT_LAYERS)
    # raw embedding feature (L=-1)
    emb_table = F.rms_norm(m.transformer.wte(uids.to(DEV)), (D,)).float()  # (U,D)
    rowidx = torch.tensor([id2row[int(t)] for t in toks], device=DEV).view(nb, T)
    def bag_decode(table):
        featmap = table[rowidx]  # (nb,T,D)
        num = torch.cumsum(featmap, dim=1); den = torch.arange(1, T+1, device=DEV, dtype=featmap.dtype).view(1, T, 1)
        Feat = (num/den).reshape(-1, D)
        n = Feat.shape[0]; rng = np.random.RandomState(0); perm = rng.permutation(n); ntr = int(0.7*n); tr, te = perm[:ntr], perm[ntr:]
        Y = torch.zeros(len(tr), K, device=DEV); Y[torch.arange(len(tr)), torch.tensor(topic[tr], device=DEV)] = 1.0
        A = Feat[tr].T @ Feat[tr] + RIDGE*torch.eye(D, device=DEV); W = torch.linalg.solve(A, Feat[tr].T @ Y)
        return float((Feat[te] @ W).argmax(1).cpu().numpy().__eq__(topic[te]).mean())
    # ceiling: decode topic from real content
    n = content.shape[0]; rng = np.random.RandomState(0); perm = rng.permutation(n); ntr = int(0.7*n); tr, te = perm[:ntr], perm[ntr:]
    base = float(np.bincount(topic, minlength=K).max()/len(topic))
    Yc = torch.zeros(len(tr), K, device=DEV); Yc[torch.arange(len(tr)), torch.tensor(topic[tr], device=DEV)] = 1.0
    Ac = content[tr].T @ content[tr] + RIDGE*torch.eye(D, device=DEV); Wc = torch.linalg.solve(Ac, content[tr].T @ Yc)
    ceiling = float((content[te] @ Wc).argmax(1).cpu().numpy().__eq__(topic[te]).mean())
    out = {'base_rate': round(base, 4), 'ceiling_realL15': round(ceiling, 4), 'K': K, 'bag_decode': {}}
    out['bag_decode']['raw_embed(L-1)'] = round(bag_decode(emb_table), 4)
    print(f"raw-embed bag: {out['bag_decode']['raw_embed(L-1)']:.4f} (ceiling {ceiling:.4f}, base {base:.4f})", flush=True)
    for L in FEAT_LAYERS:
        a = bag_decode(feats[L]); out['bag_decode'][f'postblock_L{L}'] = round(a, 4)
        print(f"post-block L{L} feature bag: {a:.4f}", flush=True)
    # shuffled token->feature map null (use best layer)
    bestL = max(FEAT_LAYERS, key=lambda L: out['bag_decode'][f'postblock_L{L}'])
    permrows = torch.randperm(feats[bestL].shape[0], generator=torch.Generator(device=DEV).manual_seed(1), device=DEV)
    out['null_shuffled_map'] = round(bag_decode(feats[bestL][permrows]), 4)
    raw = out['bag_decode']['raw_embed(L-1)']; best = max(out['bag_decode'][f'postblock_L{L}'] for L in FEAT_LAYERS)
    out['best_postblock'] = round(best, 4); out['gain_over_raw'] = round(best - raw, 4)
    out['pred_a_frontmlp_helps'] = bool(best > raw + 0.02 and out['null_shuffled_map'] < base + 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"best post-block {best:.4f} vs raw {raw:.4f} (gain {best-raw:+.4f}); shuffled-map null {out['null_shuffled_map']:.4f}", flush=True)
    print(f"(a) front-MLP per-token encoding helps the bag: {out['pred_a_frontmlp_helps']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
