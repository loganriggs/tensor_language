# HOW MANY DISTINCT MECHANISMS DOES THE 70-CIRCUIT CENSUS ACTUALLY HAVE?
#
# §2056 found a8's five circuits 0.894-parallel and non-selective: they are one rank-1 mechanism seen five
# ways. a16's three are 0.797-parallel. That was eight circuits. The census has seventy, and the same
# question applies to all of them: how many DISTINCT things has it found?
#
# §2056 compared directions inside one component, which only works within a component. To compare all
# seventy, this uses a common space every circuit writes into: the FINAL residual stream before the
# unembedding. Each circuit's direction there is the mean over its members minus the mean off its slice.
# Two circuits with near-parallel final-stream directions push the output the same way, whatever component
# they were localised to.
#
# REGISTERED PREDICTIONS (before running):
#   pred_a  The a8 five fall in one cluster and the a16 three in one cluster, at the 0.8 cosine threshold.
#           This is the validation: §2056 established that collapse independently, in a different space and
#           by a different method, so if the clustering does not recover it the method is not measuring
#           what it claims and nothing else here should be believed.
#   pred_b  Clustering at |cos| >= 0.8 yields FEWER THAN 40 clusters from 70 circuits -- the census
#           over-counts by at least a third. Registered against the possibility that a8 and a16 are
#           unusual and the rest of the census is genuinely diverse.
#   pred_c  The largest cluster holds at least 8 circuits. If the collapse were confined to a8 and a16 the
#           biggest group would be five.
#
# Writes circuits/COLLAPSE.json. Read-only with respect to circuit files.
import json
import time

import torch
import torch.nn.functional as F

import census_lib as C

C.use_state('census_state_diverse.pt')
nflat = C.nflat()
# RESTRICT TO THE CURATED CIRCUITS. The census state holds 311 tree nodes; circuits/ holds the 70 that
# were curated and certified, and those are what "the census" means. The first run clustered all 311.
import glob as _glob
CURATED = set()
for _f in _glob.glob('circuits/*.json'):
    if _f.split('/')[-1] in ('INVENTORY.json', 'LOCALISATION.json', 'INTERCHANGE.json',
                             'DIGEST.json', 'SUBSPACE.json', 'COLLAPSE.json'):
        continue
    try:
        _c = json.load(open(_f))
    except Exception:
        continue
    if isinstance(_c, dict) and _c.get('tag'):
        CURATED.add(_c['tag'])
tags = [t for t in C.all_tags() if t in CURATED]
print(f'grid {nflat}, {len(CURATED)} curated circuits, {len(tags)} of them in this state', flush=True)

masks = {}
for t in tags:
    try:
        lf = C.leaf(t)
        mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
        sl = torch.zeros(nflat, dtype=torch.bool); sl[lf['slice']] = True
        # ROOT NODES HAVE NO OFF-SLICE. r.0 through r.17 cover the whole grid, so ~sl is empty, the
        # direction is a mean over nothing, and the NaN propagates through every cosine and chains the
        # clustering into one blob. The first run produced a 288-member cluster and a NaN mean for
        # exactly this reason, and pred_a -- the validation gate -- caught it.
        if int(mm.sum()) >= 20 and int((~sl).sum()) >= 20:
            masks[t] = (mm, sl)
    except Exception:
        pass
tags = sorted(masks)
print(f'{len(tags)} circuits with >=20 members', flush=True)


@torch.no_grad()
def final_stream():
    """The residual stream entering the unembedding, for every grid position."""
    R = C.rows(); out = []
    for i in range(0, R.shape[0], 4):
        bb = R[i:i + 4, :257].to(C.DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(C.m.transformer.wte(idx), (C.D,)); x0 = x; v1 = None
        for blkm in C.m.transformer.h:
            x, v1 = blkm(x, v1, x0)
        out.append(F.rms_norm(x, (C.D,)).detach().float().reshape(-1, C.D).cpu())
    return torch.cat(out)


t0 = time.time()
S = final_stream()
print(f'captured final stream {tuple(S.shape)} ({time.time()-t0:.0f}s)', flush=True)

D = torch.stack([(lambda mm, sl: (S[mm].mean(0) - S[~sl].mean(0)))(*masks[t]) for t in tags])
D = D / D.norm(dim=1, keepdim=True)
Cos = (D @ D.T).numpy()

THRESH = 0.8
n = len(tags)
parent = list(range(n))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i


# SINGLE LINKAGE CHAINS. Two circuits land together if a path of pairwise-0.8 links joins them, so one
# promiscuous circuit merges everything. The first run's 288-member cluster was that. Complete linkage
# instead: a group is only a group if EVERY pair inside it clears the threshold.
order = sorted(range(n), key=lambda i: -float(abs(Cos[i]).sum()))
assigned = {}
comp_clusters = []
for i in order:
    placed = False
    for ci, cl in enumerate(comp_clusters):
        if all(abs(Cos[i][j]) >= THRESH for j in cl):
            cl.append(i); assigned[i] = ci; placed = True; break
    if not placed:
        assigned[i] = len(comp_clusters); comp_clusters.append([i])
for i in range(n):
    parent[i] = comp_clusters[assigned[i]][0]
groups = {}
for i, t in enumerate(tags):
    groups.setdefault(find(i), []).append(t)
clusters = sorted(groups.values(), key=len, reverse=True)

A8 = ['r.11.1.1', 'r.11.1.2', 'r.11.3.1', 'r.23.2.1', 'r.23.2.3']
A16 = ['r.3.0', 'r.3.0.2', 'r.4.1.1']
def same(group_list, want):
    have = [c for c in clusters if any(w in c for w in want)]
    return len(have) == 1 and all(w in have[0] for w in want if w in tags)

off = [abs(Cos[i][j]) for i in range(n) for j in range(i + 1, n)
       if Cos[i][j] == Cos[i][j]]                      # drop any NaN rather than let it poison the mean
rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude, circuit task',
       'method': 'per-circuit direction in the FINAL residual stream (mean over members minus mean off '
                 'slice), unit-normalised; single-linkage clustering at |cos| >= 0.8',
       'threshold': THRESH, 'n_circuits': n,
       'n_clusters': len(clusters),
       'mean_pairwise_abs_cos': round(float(sum(off) / len(off)), 4),
       'largest_cluster_size': len(clusters[0]),
       'a8_five_in_one_cluster': same(clusters, A8),
       'a16_three_in_one_cluster': same(clusters, A16),
       'clusters': [{'size': len(c), 'tags': sorted(c)} for c in clusters]}
json.dump(rep, open('circuits/COLLAPSE.json', 'w'), indent=1)
print(f"\n{n} circuits -> {len(clusters)} clusters at |cos|>={THRESH}")
print(f"  mean pairwise |cos| {rep['mean_pairwise_abs_cos']}   largest cluster {len(clusters[0])}")
print(f"  a8 five in one cluster: {rep['a8_five_in_one_cluster']}")
print(f"  a16 three in one cluster: {rep['a16_three_in_one_cluster']}")
for c in clusters[:8]:
    print(f"   size {len(c):2d}: {sorted(c)[:9]}")
print(f'({time.time()-t0:.0f}s)', flush=True)
