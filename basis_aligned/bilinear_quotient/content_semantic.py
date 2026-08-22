"""IS THE CONTENT MACHINE SEMANTIC? (first genuine attempt to NAME the content, §863 frontier). The
content machine predicts the specific next word from context; all prior probes were grammatical and
blind to it. Test a SEMANTIC hypothesis: does the readout-input residual predict the next token's
SEMANTIC CLUSTER (k-means on token embeddings) BEYOND its grammatical class? Compare, with the §836 null
lesson (proper null = shuffled-label matched-rank subspace, not random-orthonormal):
  - decode next-token semantic-cluster from class+position projection only (grammar baseline)
  - from the FULL residual
  - from a matched-rank SHUFFLED-label subspace (rank-artifact null)
If FULL >> class+position AND >> shuffled-null, the content residual carries next-token SEMANTICS —
naming the content machine as semantic next-token prediction.

REGISTERED PREDICTIONS:
  (0) SANITY: semantic clusters are coherent (report a few); class+position predicts grammar-correlated
      semantic structure at some baseline;
  (a) SEMANTIC: full residual predicts next-token semantic-cluster well ABOVE class+position AND above the
      matched shuffled-label null -> content machine encodes next-token semantics (a name for the content);
  (b) if full ≈ class+position or ≈ shuffled-null, the content is NOT captured by embedding-semantic
      clusters either — the wall is deeper (report honestly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_semantic_results.json'
NEVAL = 260; NCLUST = 32; RTOK = 64; RPOS = 32; READ_L = 15   # residual after L15 = readout input


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        d = torch.cdist(X, c); a = d.argmin(1)
        for j in range(k):
            m_ = a == j
            if m_.any(): c[j] = X[m_].mean(0)
    return a, c


@torch.no_grad()
def capture(rows):
    outs = []; seqs = []
    def h(mo, i_, o_): outs.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = m.transformer.h[READ_L].register_forward_hook(h)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    hh.remove(); return torch.cat(outs, 0), np.concatenate([s.reshape(-1) for s in seqs]), np.concatenate([s for s in seqs], 0) if False else np.stack([s for s in seqs]).reshape(-1, 256) if False else None


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def acc(Ft, y, valid, ncls, seed=0):
    idx = np.where(valid)[0]; rng = np.random.RandomState(seed); rng.shuffle(idx)
    ntr = int(0.7*len(idx)); tr, te = idx[:ntr], idx[ntr:]
    Y = torch.zeros(len(tr), ncls, device=DEV); Y[torch.arange(len(tr)), torch.tensor(y[tr], device=DEV)] = 1.0
    A = Ft[tr].T @ Ft[tr] + 1e2*torch.eye(Ft.shape[1], device=DEV); Wp = torch.linalg.solve(A, Ft[tr].T @ Y)
    return float(((Ft[te] @ Wp).argmax(1).cpu().numpy() == y[te]).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    # residual after L15 + tokens (sequence-structured for next-token)
    outs = []; seqs = []
    def h(mo, i_, o_): outs.append((o_[0] if isinstance(o_, tuple) else o_).detach().float())
    hh = m.transformer.h[READ_L].register_forward_hook(h)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    hh.remove()
    R = torch.cat([o.reshape(-1, D) for o in outs], 0)               # (N, D) residual after L15
    S = np.concatenate(seqs, 0); T = S.shape[1]
    cur = S.reshape(-1); tgt = np.full_like(S, -1); tgt[:, :-1] = S[:, 1:]; tgt = tgt.reshape(-1)
    pos = np.broadcast_to(np.arange(T), S.shape).reshape(-1)
    # semantic clusters from token EMBEDDINGS (input identity semantics)
    wte = m.transformer.wte.weight.detach().float()
    a_all, cen = kmeans(wte, NCLUST)
    tok2clust = a_all.cpu().numpy()
    nxt_clust = np.where(tgt >= 0, tok2clust[np.where(tgt >= 0, tgt, 0)], -1)
    valid = tgt >= 0
    # subspaces on the L15 residual
    Utok, g = mean_subspace(R, cur, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    rng = np.random.RandomState(0); scur = cur.copy(); rng.shuffle(scur)
    Ush, _ = mean_subspace(R, scur, RTOK+RPOS)                       # shuffled-label matched-rank null
    proj_cp = (R - g) @ Ucp                                          # (N, 96)
    proj_sh = (R - g) @ Ush
    res = {
        'full_residual': round(acc(R, nxt_clust, valid, NCLUST), 4),
        'class_position_proj': round(acc(proj_cp, nxt_clust, valid, NCLUST), 4),
        'shuffled_matched_proj': round(acc(proj_sh, nxt_clust, valid, NCLUST), 4),
    }
    # sanity: a few clusters
    exs = {}
    try:
        import tiktoken; enc = tiktoken.get_encoding('gpt2')
        for j in range(4):
            mem = np.where(tok2clust == j)[0][:8]; exs[j] = [repr(enc.decode([int(t)])) for t in mem]
    except Exception: pass
    out = {'n_clusters': NCLUST, 'read_layer': READ_L, 'decode_next_token_semantic_cluster': res,
           'full_over_classpos': round(res['full_residual'] - res['class_position_proj'], 4),
           'sample_clusters': exs, 'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"decode next-token semantic-cluster (of {NCLUST}) from L{READ_L} residual:", flush=True)
    print(f"  full {res['full_residual']} | class+pos {res['class_position_proj']} | shuffled-matched {res['shuffled_matched_proj']}", flush=True)
    print(f"  full over class+pos: {out['full_over_classpos']:+}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
