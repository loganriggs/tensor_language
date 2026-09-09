"""Integer-exact candidate operation; this does not predict trained-model logits.

An edge u->v is matrix |v><u|. When it arrives, complete paths using a
distinct earlier edge in either orientation. Source-write removals and original
fact removals differ: the latter must recompute downstream join writes.
"""
import itertools
import torch


def execute(edges):
    prefix = torch.zeros_like(edges[..., 0, :, :])
    joined = torch.zeros_like(prefix)
    writes = []
    for edge in edges.unbind(-3):
        write = edge @ prefix + prefix @ edge
        writes.append(write)
        joined = joined+write
        prefix = prefix+edge
    return prefix, joined, torch.stack(writes, -3)


def controls():
    n = 3
    maps = torch.tensor(list(itertools.product(range(n), repeat=n)))
    edges = torch.zeros(len(maps), n, n, n, dtype=torch.int64)
    for b, fmap in enumerate(maps):
        for key, value in enumerate(fmap):
            edges[b, key, value, key] = 1
    checks = {}; cases = 0
    for order in itertools.permutations(range(n)):
        ordered = edges[:, order]
        adjacency, join, writes = execute(ordered)
        diagonal = sum(edge @ edge for edge in ordered.unbind(1))
        checks['square_minus_same_edge_'+str(order)] = bool(torch.equal(join, adjacency @ adjacency-diagonal))
        # Fact edits recompute prefix updates; certify every subset exactly.
        for retained in itertools.product((0, 1), repeat=n):
            mask = torch.tensor(retained)[None, :, None, None]
            edited = ordered*mask
            a, j, _ = execute(edited)
            checks['fact_subset_'+str(order)+str(retained)] = bool(torch.equal(j, a @ a-sum(e @ e for e in edited.unbind(1))))
            cases += len(maps)
        # Independent source-write removals preserve the remaining completed paths.
        for removed in ((0,), (1,), (0, 1)):
            kept = [i for i in range(n) if i not in removed]
            actual = writes[:, kept].sum(1)
            expected = join-sum(writes[:, i] for i in removed)
            checks['source_cut_'+str(order)+str(removed)] = bool(torch.equal(actual, expected))
    cycle = torch.zeros(1, 3, 3, 3, dtype=torch.int64)
    cycle[0, 0, 1, 0] = cycle[0, 1, 2, 1] = cycle[0, 2, 0, 2] = 1
    adjacency, join, writes = execute(cycle)
    checks['nonloop_cycle_square'] = bool(torch.equal(join, adjacency @ adjacency))
    identity_edges = torch.zeros_like(cycle)
    for i in range(3):
        identity_edges[0, i, i, i] = 1
    checks['isolated_self_loops_no_distinct_edge_join'] = not bool(execute(identity_edges)[1].any())
    # Deleting a fact affects joins stored at later sources, not just its own write.
    fact_cut = cycle.clone(); fact_cut[:, 0] = 0
    checks['fact_removal_differs_from_source_write_removal'] = not torch.equal(execute(fact_cut)[1], join-writes[:, 0])
    return {'passed': all(checks.values()), 'checks': checks, 'exhaustive_function_maps': len(maps),
            'orders': 6, 'fact_subset_cases': cases, 'arithmetic': 'int64 exact', 'trained_model_calls': 0}
