"""Random walks on token-labeled lattices: grid (open), cylinder (columns wrap),
torus (both wrap). Each document assigns random distinct tokens to the nodes of
one lattice and records a uniform random walk over its edges.

The next token is stochastic (uniform over the current node's neighbors), so
models are scored on the *distribution*: legal-move rate (argmax is a neighbor)
and probability mass on neighbors, rather than exact-match accuracy.
"""

import torch

N_VOCAB = 100
N_CTX = 256
TOPOLOGIES = ("grid", "cylinder", "torus")
TRAIN_SHAPES = ((3, 3), (3, 4), (4, 4), (4, 5))     # 9-20 nodes
OOD_SHAPES = ((5, 5), (6, 6))                        # never seen in training
TAIL = N_CTX // 2                                    # summary metrics use positions >= TAIL


def neighbor_table(shape: tuple[int, int], topology: str):
    """(n_nodes, 4) neighbor ids padded with -1, plus per-node degree."""
    m, n = shape
    row = torch.arange(m).repeat_interleave(n)
    col = torch.arange(n).repeat(m)

    candidates = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = row + dr, col + dc
        r = r % m if topology == "torus" else r
        c = c % n if topology in ("cylinder", "torus") else c
        valid = (r >= 0) & (r < m) & (c >= 0) & (c < n)
        candidates.append(torch.where(valid, r * n + c, -1))

    neighbors = torch.stack(candidates, 1)
    neighbors = neighbors.gather(1, (neighbors < 0).long().argsort(stable=True, dim=1))  # valid first
    return neighbors, (neighbors >= 0).sum(1)


def walk_batch(n_seq: int, shape: tuple[int, int], topology: str, generator: torch.Generator | None = None):
    """tokens (n_seq, N_CTX), node ids (n_seq, N_CTX), token assignment perm (n_seq, n_nodes)."""
    n_nodes = shape[0] * shape[1]
    neighbors, degree = neighbor_table(shape, topology)
    perm = torch.rand(n_seq, N_VOCAB, generator=generator).argsort(1)[:, :n_nodes]

    position = torch.randint(0, n_nodes, (n_seq,), generator=generator)
    trail = [position]
    for _ in range(N_CTX - 1):
        pick = (torch.rand(n_seq, generator=generator) * degree[position]).long()
        position = neighbors[position, pick]
        trail.append(position)

    nodes = torch.stack(trail, 1)
    return perm.gather(1, nodes), nodes, perm


def train_batch(n_seq: int, topology: str, generator: torch.Generator | None = None):
    split = n_seq // len(TRAIN_SHAPES)
    return torch.cat([walk_batch(split, shape, topology, generator)[0] for shape in TRAIN_SHAPES])


def legal_tokens(nodes, perm, shape: tuple[int, int], topology: str):
    """Boolean (n_seq, N_CTX, N_VOCAB): which tokens are graph-neighbors of the current node."""
    neighbors, _ = neighbor_table(shape, topology)
    nbr = neighbors[nodes]                                   # (n_seq, n_ctx, 4)
    nbr_tok = perm.gather(1, nbr.clamp(min=0).flatten(1)).view_as(nbr)
    nbr_tok = nbr_tok.masked_fill(nbr < 0, N_VOCAB)          # park invalid slots off the end
    mask = torch.zeros(*nodes.shape, N_VOCAB + 1, dtype=torch.bool)
    mask.scatter_(2, nbr_tok, True)
    return mask[..., :N_VOCAB]


def eval_sets(topology: str, n_seq: int = 96, seed: int = 1234):
    """Fresh documents per shape: {'in': train shapes pooled, 'ood 5x5': ..., 'ood 6x6': ...}."""
    generator = torch.Generator().manual_seed(seed)
    sets = {}
    pooled = [walk_batch(n_seq, s, topology, generator) for s in TRAIN_SHAPES]
    sets["in"] = (torch.cat([t for t, _, _ in pooled]),
                  torch.cat([legal_tokens(n, p, s, topology) for (_, n, p), s in zip(pooled, TRAIN_SHAPES)]))
    for shape in OOD_SHAPES:
        tokens, nodes, perm = walk_batch(2 * n_seq, shape, topology, generator)
        sets[f"ood {shape[0]}x{shape[1]}"] = (tokens, legal_tokens(nodes, perm, shape, topology))
    return sets
