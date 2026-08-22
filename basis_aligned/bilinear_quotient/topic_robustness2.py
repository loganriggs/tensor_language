"""DOES THE CONTINUOUS TOPIC GEOMETRY REPLICATE across a disjoint data split? (proper retest after §873,
which showed the DISCRETE 12-cluster labels don't replicate — as expected for an arbitrary partition of a
continuum). The claim to test is not "the same 12 buckets" but "the same continuous topic SUBSPACE /
directions emerge in independent data." Metrics that don't depend on discrete labels or rare-token
fingerprints:
  - SUBSPACE OVERLAP: top-r PCA of the content residual on split A vs split B; mean squared cosine of
    principal angles = ||P_A P_B||_F^2 / r (1.0 = identical subspace, r/D = random). Do the dominant
    content directions coincide across splits?
  - CENTROID COSINE: k-means centroids on A vs B (K=12), greedily matched by cosine; mean matched cosine.
    Do the same cluster CENTERS (as directions) recur, even if the hard label boundaries are arbitrary?
Both vs a SHUFFLED-CONTENT null (permute content across token positions, destroying the token<->content
link). More data than §873 (better topic coverage per split).

REGISTERED PREDICTIONS:
  (0) SANITY: shuffled-content subspace overlap ~ r/D (chance), centroid cosine low;
  (a) CONTINUOUS TOPIC GEOMETRY REPLICATES: content-subspace overlap A<->B is HIGH (>0.5) and WELL ABOVE
      the shuffled null, and matched-centroid cosine >> null -> the topic structure is a robust continuous
      property of the model (vindicating §866-872 as a continuous structure, not the discrete labels);
  (b) if subspace overlap ~ null, the content geometry itself does not replicate -> the topic finding is a
      one-split artifact after all (retract §866-872). Report plainly."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'topic_robustness2_results.json'
CONTENT_L = 15; NEVAL = 420; RTOK = 64; RPOS = 32; K = 12; RPCA = 24


def forward_cap(idx):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[CONTENT_L].register_forward_hook(h)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    hh.remove()
    return cap['r']


def kmeans_centroids(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return c


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def pca_basis(X, r):
    Xc = X - X.mean(0, keepdim=True)
    V = torch.linalg.svd(Xc, full_matrices=False)[2][:r].T.contiguous()   # (D, r) orthonormal
    return V


def subspace_overlap(Ua, Ub):
    """mean squared cosine of principal angles = ||Ua^T Ub||_F^2 / r."""
    r = Ua.shape[1]
    return float((Ua.T @ Ub).pow(2).sum() / r)


def matched_centroid_cosine(Ca, Cb):
    Ca = Ca/(Ca.norm(dim=1, keepdim=True)+1e-9); Cb = Cb/(Cb.norm(dim=1, keepdim=True)+1e-9)
    Sim = (Ca @ Cb.T).cpu().numpy(); used = set(); scores = []
    order = np.argsort(-Sim.max(1))
    for a in order:
        row = Sim[a].copy()
        for b in used: row[b] = -2
        b = int(row.argmax()); used.add(b); scores.append(float(Sim[a, b]))
    return float(np.mean(scores))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    Rs = []; seqs = []
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :256].to(DEV).contiguous(); Rs.append(forward_cap(idx).cpu()); seqs.append(idx.cpu().numpy())
    R = torch.cat(Rs, 0); S = np.concatenate(seqs, 0); Nseq = S.shape[0]
    allR = R.reshape(-1, D).to(DEV); toks = S.reshape(-1); pos = np.broadcast_to(np.arange(256), S.shape).reshape(-1)
    Utok, g = mean_subspace(allR, toks, RTOK); Upos, _ = mean_subspace(allR, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (allR-g) - ((allR-g)@Ucp)@Ucp.T
    content = content/(content.norm(dim=1, keepdim=True)+1e-9)
    rng = np.random.RandomState(0); so = rng.permutation(Nseq); A_seq = set(so[:Nseq//2].tolist())
    seq_of = np.repeat(np.arange(Nseq), 256); inA = np.array([s in A_seq for s in seq_of])
    A = content[torch.tensor(inA, device=DEV)]; B = content[torch.tensor(~inA, device=DEV)]
    perm = rng.permutation(content.shape[0]); cs = content[perm]
    As = cs[torch.tensor(inA, device=DEV)]; Bs = cs[torch.tensor(~inA, device=DEV)]
    # subspace overlap
    Ua = pca_basis(A, RPCA); Ub = pca_basis(B, RPCA); ov = subspace_overlap(Ua, Ub)
    Uas = pca_basis(As, RPCA); Ubs = pca_basis(Bs, RPCA); ov_null = subspace_overlap(Uas, Ubs)
    chance_ov = RPCA/D
    # centroid cosine
    Ca = kmeans_centroids(A, K, seed=1); Cb = kmeans_centroids(B, K, seed=2); cc = matched_centroid_cosine(Ca, Cb)
    Cas = kmeans_centroids(As, K, seed=1); Cbs = kmeans_centroids(Bs, K, seed=2); cc_null = matched_centroid_cosine(Cas, Cbs)
    out = {'r_pca': RPCA, 'k': K, 'n_tokens': int(content.shape[0]),
           'subspace_overlap': round(ov, 3), 'subspace_overlap_null': round(ov_null, 3), 'subspace_overlap_chance': round(chance_ov, 3),
           'centroid_cosine': round(cc, 3), 'centroid_cosine_null': round(cc_null, 3),
           'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_continuous_replicates'] = bool(ov > 0.5 and ov > 3*max(ov_null, 1e-6) and cc > 2*max(cc_null, 1e-6))
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"content-SUBSPACE overlap A<->B: {ov:.3f} | shuffled-null {ov_null:.3f} | chance(r/D) {chance_ov:.3f}", flush=True)
    print(f"matched-CENTROID cosine A<->B: {cc:.3f} | shuffled-null {cc_null:.3f}", flush=True)
    print(f"(a) continuous topic geometry replicates: {out['pred_a_continuous_replicates']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
