"""TOY for co-activation grouping + token-semantic subspace (sanity-checks the
code behind 766/semantic_subspace AND clarifies the concepts). Plants KNOWN
structure so we can verify the detectors recover it, and demonstrates the
"subset/redundant vs genuinely co-activating" distinction the correlation-based
grouping cannot see.

GROUND TRUTH:
  - D=64 output dims; P=48 atoms with random unit decoder directions.
  - 6 co-activation GROUPS of 8 atoms. Each datapoint activates each group w.p. 0.4;
    when a group is active its member atoms fire (they CO-ACTIVATE).
    * groups 0-4 COMPLEMENTARY: all members fire ~symmetrically when the group is on.
    * group 5 SUBSET/REDUNDANT: nested firing -- member 0 (parent) fires whenever the
      group is on, member k fires only a 2^-k fraction of those (children subset the
      parent). Symmetric CO-ACTIVATION correlation still lumps them into one group,
      but a DIRECTIONAL containment P(i active | j active) reveals parent!=child.
  - SEMANTIC subspace: an r_sem=8 orthonormal basis Sem; each of T=20 tokens has a
    fixed point mu[t] in Sem, so the token-conditional MEAN of the output lives in
    Sem by construction (group activations are token-independent -> average out).
  - Output O = Sem @ mu[token] + sum_a code[a]*decoder[:,a] + noise.

CHECKS (with ground truth):
  (a) GROUP RECOVERY: correlation-clustering of codes recovers the planted groups
      (adjusted-Rand >= 0.7 for the complementary groups);
  (b) SUBSET vs CO-ACTIVATING: within the subset group, Pearson correlation is high
      (looks like a normal group) yet containment is ASYMMETRIC (parent|child ~1,
      child|parent low); within a complementary group containment is ~symmetric.
      -> correlation cannot distinguish redundancy from complementarity; containment can;
  (c) SEMANTIC RECOVERY: token-mean SVD subspace overlaps the planted Sem (>=0.9),
      and is data-split stable (>=0.9), while a random subspace is not.
Emits a figure: co-activation correlation heatmap (block structure) + containment
asymmetry for a complementary vs the subset group."""
import json, time, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE, BLUES, DIVERGING
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'toy_group_semantic_results.json'; FIG = PT + 'toy_group_semantic.png'
D = 64; P = 48; G = 6; GSIZE = 8; RSEM = 8; T = 20; N = 8000; NOISE = 0.15


def plant(seed=0):
    rng = np.random.RandomState(seed)
    dec = rng.randn(D, P); dec /= np.linalg.norm(dec, axis=0, keepdims=True)
    Sem = np.linalg.qr(rng.randn(D, RSEM))[0]                     # semantic basis (D, r_sem)
    mu = rng.randn(T, RSEM) * 1.2                                 # token means in Sem
    groups = [list(range(g*GSIZE, (g+1)*GSIZE)) for g in range(G)]
    tok = rng.randint(0, T, N)
    code = np.zeros((N, P))
    gact = rng.rand(N, G) < 0.4                                   # group active mask
    for g in range(G):
        mem = groups[g]; on = gact[:, g]
        if g < G-1:                                               # COMPLEMENTARY
            # members co-fire but PROBABILISTICALLY (p=0.8 each when group on) ->
            # containment symmetric but < 1 (a real circuit, not perfect lockstep)
            for a in mem:
                fires = on & (rng.rand(N) < 0.8)
                code[fires, a] = np.abs(rng.randn(fires.sum()))
        else:                                                     # SUBSET / redundant
            for k, a in enumerate(mem):
                fires = on & (rng.rand(N) < 0.5**k)               # nested subsets
                code[fires, a] = np.abs(rng.randn(fires.sum())) + 0.5
    O = mu[tok] @ Sem.T + code @ dec.T + NOISE*rng.randn(N, D)
    return O, tok, code, dec, Sem, groups


def cluster_codes(code, g):
    active = np.where((code > 1e-6).mean(0) > 0)[0]
    C = np.corrcoef(code[:, active].T); C = np.nan_to_num(C)
    dist = 1 - C; np.fill_diagonal(dist, 0)
    lab = fcluster(linkage(squareform(dist, checks=False), 'average'), g, 'maxclust')
    full = -np.ones(code.shape[1], int); full[active] = lab
    return full, C, active


def adjusted_rand(a, b):
    from itertools import combinations
    n = len(a);
    def idx(x):
        d = {};
        for i, v in enumerate(x): d.setdefault(v, []).append(i)
        return d
    A, B = idx(a), idx(b)
    import math
    def comb2(k): return k*(k-1)//2
    cont = {}
    for ka, ia in A.items():
        for kb, ib in B.items():
            cont[(ka, kb)] = len(set(ia) & set(ib))
    sum_ij = sum(comb2(v) for v in cont.values())
    sum_a = sum(comb2(len(v)) for v in A.values()); sum_b = sum(comb2(len(v)) for v in B.values())
    exp = sum_a*sum_b/comb2(n); mx = 0.5*(sum_a+sum_b)
    return (sum_ij - exp)/(mx - exp + 1e-12)


def token_mean_dirs(O, tok, mincount=5):
    g = O.mean(0, keepdims=True); rows = []; wt = []
    for t in np.unique(tok):
        m = tok == t
        if m.sum() < mincount: continue
        rows.append(O[m].mean(0) - g[0]); wt.append(np.sqrt(m.sum()))
    M = np.array(rows) * np.array(wt)[:, None]
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    return Vh, S


def sub_overlap(A, B):
    return float(np.linalg.svd(A.T @ B, compute_uv=False).mean())


def containment(code, members):
    # P(i active | j active) matrix within a group
    act = code[:, members] > 1e-6; m = len(members); C = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            denom = act[:, j].sum()
            C[i, j] = (act[:, i] & act[:, j]).sum()/max(denom, 1)
    return C


def main():
    t0 = time.time()
    O, tok, code, dec, Sem, groups = plant(0)
    planted = -np.ones(P, int)
    for g, mem in enumerate(groups):
        for a in mem: planted[a] = g

    # (a) group recovery
    lab, C, active = cluster_codes(code, G)
    ari = adjusted_rand(planted[active], lab[active])
    # complementary-only ARI (exclude subset group members)
    comp_mask = np.array([planted[a] < G-1 for a in active])
    ari_comp = adjusted_rand(planted[active][comp_mask], lab[active][comp_mask])
    print(f'(a) group recovery ARI all {ari:.3f}  complementary-only {ari_comp:.3f}', flush=True)

    # (b) subset vs co-activating: containment asymmetry
    comp_C = containment(code, groups[0])          # complementary group
    sub_C = containment(code, groups[G-1])         # subset group
    def asym(Cm): return float(np.abs(Cm - Cm.T)[np.triu_indices(len(Cm), 1)].mean())
    corr_comp = float(np.corrcoef(code[:, groups[0]].T)[np.triu_indices(GSIZE, 1)].mean())
    corr_sub = float(np.corrcoef(code[:, groups[G-1]].T)[np.triu_indices(GSIZE, 1)].mean())
    print(f'(b) complementary: mean|corr| {corr_comp:.2f} containment-asym {asym(comp_C):.2f}  |  '
          f'subset: mean|corr| {corr_sub:.2f} containment-asym {asym(sub_C):.2f}', flush=True)

    # (c) semantic recovery + data stability
    Vh, S = token_mean_dirs(O, tok)
    sem_ov = sub_overlap(Vh[:RSEM].T, Sem)
    rng = np.random.RandomState(1); Rr = np.linalg.qr(rng.randn(D, RSEM))[0]
    rand_ov = sub_overlap(Rr, Sem)
    h = N//2; Va, _ = token_mean_dirs(O[:h], tok[:h]); Vb, _ = token_mean_dirs(O[h:], tok[h:])
    data_ov = sub_overlap(Va[:RSEM].T, Vb[:RSEM].T)
    print(f'(c) semantic recovery overlap {sem_ov:.3f} (random {rand_ov:.3f})  data-stable {data_ov:.3f}', flush=True)

    # figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6)); fig.patch.set_facecolor(SURFACE)
    order = np.argsort(planted[active]); Cs = C[order][:, order]
    im = axs[0].imshow(Cs, cmap=BLUES, vmin=0, vmax=1); axs[0].set_title('co-activation correlation\n(atoms ordered by planted group -> blocks)', color=INK, fontsize=11, loc='left')
    axs[0].set_xlabel('atom'); axs[0].set_ylabel('atom'); fig.colorbar(im, ax=axs[0], fraction=0.046)
    im1 = axs[1].imshow(comp_C, cmap=BLUES, vmin=0, vmax=1); axs[1].set_title('COMPLEMENTARY group\nP(row active | col active) ~ symmetric', color=INK, fontsize=11, loc='left')
    axs[1].set_xlabel('member'); fig.colorbar(im1, ax=axs[1], fraction=0.046)
    im2 = axs[2].imshow(sub_C, cmap=BLUES, vmin=0, vmax=1); axs[2].set_title('SUBSET/redundant group\nP(row|col) ASYMMETRIC (parent vs child)', color=INK, fontsize=11, loc='left')
    axs[2].set_xlabel('member'); fig.colorbar(im2, ax=axs[2], fraction=0.046)
    for ax in axs:
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
    fig.suptitle('Toy: co-activation groups (blocks) + subset-vs-complementary (containment asymmetry)', fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(FIG, dpi=150, facecolor=SURFACE); print('wrote', FIG, flush=True)

    pa = ari_comp >= 0.7
    pb = asym(sub_C) > 3*asym(comp_C)
    pc = sem_ov >= 0.9 and data_ov >= 0.9 and rand_ov < 0.5
    out = {'ari_all': round(ari, 3), 'ari_complementary': round(ari_comp, 3),
           'corr_complementary': round(corr_comp, 3), 'corr_subset': round(corr_sub, 3),
           'containment_asym_complementary': round(asym(comp_C), 3), 'containment_asym_subset': round(asym(sub_C), 3),
           'semantic_overlap': round(sem_ov, 3), 'random_overlap': round(rand_ov, 3), 'data_stable': round(data_ov, 3),
           'pred_a_group_recovery': bool(pa), 'pred_b_subset_distinguished': bool(pb), 'pred_c_semantic': bool(pc),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) groups recovered: {pa}; (b) subset distinguished by containment: {pb}; (c) semantic recovered+stable: {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
