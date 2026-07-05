"""Generalized graph-tracing data: uniform random walks over token-labeled graphs
from many families, one graph per document.

Train families: ring, directed ring (≡ the cycle task), grid, cylinder, random
tree, random 3-regular. Held out entirely: torus and Erdős–Rényi graphs, plus
larger sizes of the train families.

Graph structures are pre-sampled into per-family pools at import time (cheap);
token labelings and walks are fresh per batch.
"""

import torch

from geodata import neighbor_table as lattice_neighbors

N_VOCAB = 100
N_CTX = 256
TAIL = N_CTX // 2
MAX_NODES = 36
MAX_DEG = 8


def pad(neighbors: torch.Tensor, degree: torch.Tensor):
    n, d = neighbors.shape
    out_n = torch.full((MAX_NODES, MAX_DEG), -1, dtype=torch.long)
    out_n[:n, :d] = neighbors
    out_d = torch.ones(MAX_NODES, dtype=torch.long)          # pad nodes: degree 1 (never visited)
    out_d[:n] = degree
    return out_n, out_d


def ring(n: int, directed: bool = False):
    idx = torch.arange(n)
    nb = torch.stack([(idx + 1) % n], 1) if directed else torch.stack([(idx - 1) % n, (idx + 1) % n], 1)
    return pad(nb, torch.full((n,), 1 if directed else 2))


def lattice(shape, topology):
    nb, deg = lattice_neighbors(shape, topology)
    return pad(nb, deg)


def from_edges(n: int, edges: set):
    lists = [[] for _ in range(n)]
    for a, b in edges:
        lists[a].append(b)
        lists[b].append(a)
    if max(len(l) for l in lists) > MAX_DEG or min(len(l) for l in lists) == 0:
        return None
    nb = torch.full((n, MAX_DEG), -1, dtype=torch.long)
    for v, l in enumerate(lists):
        nb[v, : len(l)] = torch.tensor(l)
    if not connected(lists, n):
        return None
    return pad(nb, torch.tensor([len(l) for l in lists]))


def connected(lists, n: int) -> bool:
    seen, stack = {0}, [0]
    while stack:
        for u in lists[stack.pop()]:
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return len(seen) == n


def tree(n: int, generator):
    """Uniform random tree via a Prüfer sequence."""
    seq = torch.randint(0, n, (n - 2,), generator=generator).tolist()
    degree = [1] * n
    for v in seq:
        degree[v] += 1
    edges = set()
    for v in seq:
        leaf = min(u for u in range(n) if degree[u] == 1)
        edges.add((min(leaf, v), max(leaf, v)))
        degree[leaf] -= 1
        degree[v] -= 1
    last = [u for u in range(n) if degree[u] == 1]
    edges.add((min(last), max(last)))
    return from_edges(n, edges)


def k_regular(n: int, k: int, generator):
    """Random simple k-regular graph via the configuration model (retry on clash)."""
    stubs = torch.arange(n).repeat_interleave(k)
    stubs = stubs[torch.randperm(len(stubs), generator=generator)]
    edges = set()
    for a, b in zip(stubs[0::2].tolist(), stubs[1::2].tolist()):
        if a == b or (min(a, b), max(a, b)) in edges:
            return None
        edges.add((min(a, b), max(a, b)))
    return from_edges(n, edges)


def erdos_renyi(n: int, generator):
    p = 2.5 / n
    upper = torch.rand(n, n, generator=generator) < p
    edges = {(a, b) for a in range(n) for b in range(a + 1, n) if upper[a, b]}
    return from_edges(n, edges)


def build_pool(builder, args_list, n_each: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    pool = []
    for args in args_list:
        made = 0
        while made < n_each:
            g = builder(*args, generator)
            if g is not None:
                pool.append(g)
                made += 1
    return pool


TRAIN_POOLS = {
    "dring": [ring(n, directed=True) for n in range(5, 21)],
    "ring": [ring(n) for n in range(5, 21)],
    "grid": [lattice(s, "grid") for s in ((3, 3), (3, 4), (4, 4), (4, 5))],
    "cylinder": [lattice(s, "cylinder") for s in ((3, 3), (3, 4), (4, 4), (4, 5))],
    "tree": build_pool(tree, [(n,) for n in range(8, 17)], 40, seed=100),
    "kreg": build_pool(k_regular, [(n, 3) for n in (10, 12, 14, 16)], 90, seed=200),
}
TRAIN_FAMILIES = tuple(TRAIN_POOLS)

OOD_POOLS = {
    "torus (unseen family)": [lattice(s, "torus") for s in ((4, 4), (4, 5))],
    "ER graph (unseen family)": build_pool(erdos_renyi, [(n,) for n in (12, 14, 16)], 60, seed=300),
    "ring n=30 (unseen size)": [ring(30)],
    "dring n=27 (unseen size)": [ring(27, directed=True)],
    "grid 6x6 (unseen size)": [lattice((6, 6), "grid")],
    "tree n=24 (unseen size)": build_pool(tree, [(24,)], 60, seed=400),
    "kreg n=24 (unseen size)": build_pool(k_regular, [(24, 3)], 60, seed=500),
}


def walk_pool(pool, n_seq: int, generator: torch.Generator | None = None):
    """Sample structures + labelings + walks. Returns tokens, node ids, perms, structure ids."""
    pick = torch.randint(0, len(pool), (n_seq,), generator=generator)
    nb = torch.stack([pool[i][0] for i in pick.tolist()])
    deg = torch.stack([pool[i][1] for i in pick.tolist()])
    n_nodes = (nb[:, :, 0] >= 0).sum(1)

    perm = torch.rand(n_seq, N_VOCAB, generator=generator).argsort(1)[:, :MAX_NODES]
    position = torch.randint(0, MAX_NODES, (n_seq,), generator=generator) % n_nodes
    trail = [position]
    rows = torch.arange(n_seq)
    for _ in range(N_CTX - 1):
        choice = (torch.rand(n_seq, generator=generator) * deg[rows, position]).long()
        position = nb[rows, position, choice]
        trail.append(position)
    nodes = torch.stack(trail, 1)
    return perm.gather(1, nodes), nodes, perm, pick


def legal_tokens(nodes, perm, nb):
    """(n_seq, N_CTX, N_VOCAB) mask of graph-neighbor tokens of the current node."""
    nbr = nb[torch.arange(len(nodes))[:, None], nodes]           # (B, ctx, MAX_DEG)
    nbr_tok = perm.gather(1, nbr.clamp(min=0).flatten(1)).view_as(nbr)
    nbr_tok = nbr_tok.masked_fill(nbr < 0, N_VOCAB)
    mask = torch.zeros(*nodes.shape, N_VOCAB + 1, dtype=torch.bool)
    mask.scatter_(2, nbr_tok, True)
    return mask[..., :N_VOCAB]


def train_batch(n_seq: int, generator: torch.Generator | None = None):
    split = n_seq // len(TRAIN_FAMILIES)
    return torch.cat([walk_pool(TRAIN_POOLS[f], split, generator)[0] for f in TRAIN_FAMILIES])


def eval_sets(n_seq: int = 64, seed: int = 1234):
    generator = torch.Generator().manual_seed(seed)
    sets = {}
    for name, pool in list(TRAIN_POOLS.items()) + list(OOD_POOLS.items()):
        tokens, nodes, perm, pick = walk_pool(pool, n_seq, generator)
        nb = torch.stack([pool[i][0] for i in pick.tolist()])
        sets[name] = (tokens, legal_tokens(nodes, perm, nb))
    return sets


def widening_rings(sizes=(4, 8, 16)):
    """Concentric rings (sizes doubling outward) with radial spokes: node j of
    ring r connects to nodes 2j and 2j+1 of ring r+1. User-suggested structure."""
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    edges = set()
    for r, s in enumerate(sizes):
        for j in range(s):
            a = offsets[r] + j
            edges.add((min(a, offsets[r] + (j + 1) % s), max(a, offsets[r] + (j + 1) % s)))
            if r + 1 < len(sizes):
                for child in (2 * j, 2 * j + 1):
                    b = offsets[r + 1] + child
                    edges.add((min(a, b), max(a, b)))
    return from_edges(sum(sizes), edges)
