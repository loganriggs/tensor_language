"""RSPD CLUSTER LOWRANK (user idea): is a HIGH-rank layer actually a UNION
of LOW-rank per-cluster circuits? Cluster the tokens by how the layer's
decoder treats them, then measure each cluster's OWN recovery rank. If the
per-cluster ranks are far below the global rank, the high global rank =
many low-rank special cases (different clusters use different subspaces).
This tests the open 699 question (why mlp1/mlp2 are high-rank) and
implements the user's 'cluster then per-cluster low rank' suggestion.

To stay comparable and avoid the effective_rank pitfall (588), everything
is measured in ONE reconstruction metric: recovery rank = smallest r that
recovers >=80% of the RESPONSE ENERGY (via RSPD's own subset_asvd_losses /
per_datum_truncation_losses). NOTE: this is response-energy rank, a
reconstruction quantity -- related to but not identical to the CE-priced
r80 (660: energy basis != functional basis); it is the right space for the
per-cluster-vs-global comparison since CE cannot be measured per cluster.

Clustering: k-means on the L2-normalized per-token response (X@W.T)_i --
groups tokens the decoder maps in similar directions. Components: mlp1.Down
(high-rank, r80=128) primary; mlp0.Down (low-rank, r80=8) control.

REGISTERED PREDICTIONS:
  (0) SANITY: global response-energy recovery rank orders like r80 --
      mlp1 global rank >> mlp0 global rank;
  (a) HYPOTHESIS (mlp1): mean per-cluster recovery rank <= 0.5 x the global
      recovery rank -- the high-rank layer dissolves into lower-rank
      clusters (union of low-rank special-case circuits);
  (b) report global rank, per-cluster ranks + token content, for both;
  NULL: SHUFFLED clusters (random token assignment, same cluster sizes)
      do NOT reduce rank -- their mean per-cluster rank stays near the
      global rank (the reduction is real structure, not just smaller N)."""
import json, time, sys, torch
import numpy as np
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from rspd.circuit_isolation import subset_asvd_losses
from rspd.mrank import per_datum_truncation_losses
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_cluster_lowrank_results.json'
NCAP = 12          # ~3072 tokens
K = 8              # clusters
FRAC = 0.80


def recon_rank(L, frac=FRAC):
    """L: (R+1, n) per-datum truncation losses. Smallest r recovering
    >=frac of the aggregate response energy."""
    agg = L.sum(1) if hasattr(L, 'sum') else np.asarray(L).sum(1)
    agg = np.asarray(agg.cpu() if torch.is_tensor(agg) else agg, dtype=np.float64)
    tot = agg[0] + 1e-12
    for r in range(1, len(agg)):
        if 1 - agg[r] / tot >= frac:
            return r
    return len(agg) - 1


def kmeans(Xn, k, iters=25, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    idx = torch.randperm(Xn.shape[0], generator=g)[:k]
    C = Xn[idx].clone()
    for _ in range(iters):
        d = torch.cdist(Xn, C)
        a = d.argmin(1)
        for j in range(k):
            m_ = a == j
            if m_.any():
                C[j] = Xn[m_].mean(0)
    return a


@torch.no_grad()
def capture(mod, rows, n, in_dim):
    cap = []; toks = []
    h = mod.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, in_dim)))
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); toks.append(idx.reshape(-1).cpu())
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0), torch.cat(toks, 0)


def d1(t):
    try: return cl.d1(int(t))
    except Exception: return f'<{t}>'


@torch.no_grad()
def analyze(name, mod, in_dim, rows):
    W = mod.weight.data.float().to(DEV)
    X, toks = capture(mod, rows, NCAP, in_dim); tk = toks.numpy()
    N = X.shape[0]
    # global recovery rank (energy)
    Lg = per_datum_truncation_losses(X, W)
    g_rank = recon_rank(Lg)
    # cluster on normalized response
    Y = X @ W.T
    Yn = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-9)
    assign = kmeans(Yn.cpu(), K).numpy()
    clusters = []
    per_ranks = []
    for j in range(K):
        idxj = np.where(assign == j)[0]
        if len(idxj) < 20:
            continue
        _, _, Lc = subset_asvd_losses(X, W, idxj)
        rj = recon_rank(Lc)
        per_ranks.append(rj)
        toptoks = [d1(t) for t, _ in Counter(tk[idxj].tolist()).most_common(6)]
        clusters.append({'j': int(j), 'n': int(len(idxj)), 'rank': int(rj),
                         'top_tokens': toptoks})
    # shuffled null: same cluster sizes, random assignment
    rng = np.random.default_rng(0)
    perm = rng.permutation(N); sizes = [c['n'] for c in clusters]
    null_ranks = []; pos = 0
    for sz in sizes:
        idxj = perm[pos:pos + sz]; pos += sz
        _, _, Lc = subset_asvd_losses(X, W, idxj)
        null_ranks.append(recon_rank(Lc))
    mean_pc = float(np.mean(per_ranks)); mean_null = float(np.mean(null_ranks))
    print(f'\n[{name}] global rank {g_rank}  mean per-cluster {mean_pc:.1f}  '
          f'(shuffled null {mean_null:.1f})  N={N}', flush=True)
    for c in sorted(clusters, key=lambda c: c['rank']):
        print(f'   cluster n={c["n"]:4d} rank={c["rank"]:4d}  {c["top_tokens"]}', flush=True)
    return {'global_rank': int(g_rank), 'mean_per_cluster_rank': round(mean_pc, 2),
            'mean_shuffled_rank': round(mean_null, 2), 'clusters': clusters,
            'per_cluster_ranks': per_ranks, 'shuffled_ranks': null_ranks, 'N': int(N)}


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NCAP)
    out = {}
    out['mlp1'] = analyze('mlp1.Down', m.transformer.h[1].mlp.Down, 4608, rows)
    out['mlp0'] = analyze('mlp0.Down', m.transformer.h[0].mlp.Down, 4608, rows)

    m1 = out['mlp1']
    p0 = m1['global_rank'] > out['mlp0']['global_rank']
    pa = m1['mean_per_cluster_rank'] <= 0.5 * m1['global_rank']
    null_ok = m1['mean_shuffled_rank'] >= 0.8 * m1['global_rank']
    print(f'\n(0) mlp1 global rank {m1["global_rank"]} > mlp0 '
          f'{out["mlp0"]["global_rank"]}: {p0}', flush=True)
    print(f'(a) mlp1 clusters low-rank (mean {m1["mean_per_cluster_rank"]} <= 0.5*'
          f'{m1["global_rank"]}): {pa}', flush=True)
    print(f'NULL shuffled stays high ({m1["mean_shuffled_rank"]} >= 0.8*'
          f'{m1["global_rank"]}): {null_ok}', flush=True)
    out.update({'pred_0': bool(p0), 'pred_a_clusters_lowrank': bool(pa),
                'null_ok': bool(null_ok), 'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
