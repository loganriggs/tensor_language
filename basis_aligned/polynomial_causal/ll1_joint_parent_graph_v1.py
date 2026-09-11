"""Executable overlapping-parent graph with one joint quadratic core per group.

Discovery uses weights only. Reader incidence is frozen from the subspace census.
Unlike a sum of single-parent projections, all shared/shared interactions occur
once. Private quadratic directions are truncated to keep each group rank fixed.
"""
import json
import hashlib
from pathlib import Path
import torch
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner


@torch.no_grad()
def build(a, s, c, readers, nodes):
    groups = []
    pairs = []
    conditions = []
    for g in range(len(a)):
        ids = [i for i, node in enumerate(nodes) if g in node['consumers']]
        k, rank = len(ids), a.shape[1]
        if k > rank:
            raise ValueError('More proposed parents than the registered group rank')
        rr = readers[ids]
        if k:
            u, change = torch.linalg.qr(rr.T, mode='reduced')
            condition = float(torch.linalg.cond(change))
            if condition > 1e6:
                raise ValueError('Dependent shared readers require a new proposal')
            inv = torch.linalg.inv(change)
            residual_a = a[g] - (a[g] @ u) @ u.T
        else:
            u = a.new_zeros(a.shape[-1], 0)
            inv = a.new_zeros(0, 0)
            residual_a = a[g]
            condition = 1.
        conditions.append(condition)
        # Diagonalize the private restriction in at most rank coordinates.
        private_span, small_a = torch.linalg.qr(residual_a.T, mode='reduced')
        small_q = (small_a * s[g][None]) @ small_a.T
        values, vectors = torch.linalg.eigh(small_q)
        selected = values.abs().argsort(descending=True)[:rank-k]
        private = (private_span @ vectors[:, selected]).T
        lam = values[selected]
        au, av = a[g] @ u, a[g] @ private.T
        shared = inv @ ((au.T * s[g][None]) @ au) @ inv.T
        cross = inv @ ((au.T * s[g][None]) @ av)
        pair_ids, coeff = [], []
        for i in range(k):
            for j in range(i, k):
                pair = (ids[i], ids[j])
                if pair not in pairs:
                    pairs.append(pair)
                pair_ids.append(pairs.index(pair))
                coeff.append(shared[i, j] * (1 if i == j else 2))
        groups.append(dict(parent_ids=torch.tensor(ids, dtype=torch.long),
                           pair_ids=torch.tensor(pair_ids, dtype=torch.long),
                           shared_coeff=torch.stack(coeff) if coeff else a.new_zeros(0),
                           private=private, lam=lam, cross=2*cross, writer=c[g]))
    return dict(readers=readers, pairs=torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
                groups=groups, parent_conditions=conditions)


def execute(graph, x):
    z = x @ graph['readers'].T
    pairs = graph['pairs']
    products = z[..., pairs[:, 0]] * z[..., pairs[:, 1]]
    result = x.new_zeros(*x.shape[:-1], graph['groups'][0]['writer'].numel())
    for group in graph['groups']:
        y = x @ group['private'].T
        scalar = (y.square() * group['lam']).sum(-1)
        scalar = scalar + products[..., group['pair_ids']] @ group['shared_coeff']
        scalar = scalar + (z[..., group['parent_ids']] * (y @ group['cross'].T)).sum(-1)
        result = result + scalar[..., None] * group['writer']
    return result


def factors(graph):
    """Equivalent signed-square bank for coefficient scoring, not DAG pricing."""
    aa, ss, cc = [], [], []
    for group in graph['groups']:
        k = len(group['parent_ids'])
        l = len(group['lam'])
        core = group['private'].new_zeros(k+l, k+l)
        offset = 0
        for i in range(k):
            for j in range(i, k):
                core[i, j] = core[j, i] = group['shared_coeff'][offset] / (1 if i == j else 2)
                offset += 1
        core[:k, k:] = group['cross']/2
        core[k:, :k] = group['cross'].T/2
        core[k:, k:] = torch.diag(group['lam'])
        values, vectors = torch.linalg.eigh(core)
        basis = torch.cat((graph['readers'][group['parent_ids']], group['private']))
        aa.append(vectors.T @ basis)
        ss.append(values)
        cc.append(group['writer'])
    return torch.stack(aa), torch.stack(ss), torch.stack(cc)


def price(graph, original):
    a, s, c = original
    floats = graph['readers'].numel()
    ints = graph['pairs'].numel()
    linear_readers = len(graph['readers'])
    variable_products = len(graph['pairs'])
    for g in graph['groups']:
        floats += sum(g[key].numel() for key in ('shared_coeff', 'private', 'lam', 'cross', 'writer'))
        ints += g['parent_ids'].numel() + g['pair_ids'].numel()
        linear_readers += len(g['lam'])
        variable_products += len(g['lam']) + len(g['parent_ids'])
    old = sum(v.numel() for v in original)
    return dict(old_floats=old, graph_floats=floats, saved_float_fraction=1-floats/old,
                int64_indices=ints, graph_numeric_bytes_fp64=8*(floats+ints),
                old_numeric_bytes_fp64=8*old, old_linear_readers=a.shape[0]*a.shape[1],
                graph_linear_readers=linear_readers, old_variable_products=s.numel(),
                graph_variable_products=variable_products,
                coefficient_multiplications=floats,
                note='Logical arrays only; serialization/container overhead excluded. Bias, output whitener/native U, RMS, residual and tail are common background and remain required. Dense coefficient multiply count uses each stored floating coefficient once; variable products are counted separately.')


def control():
    from chunked_bilinear_coefficient_v1 import dense
    torch.manual_seed(2601)
    d, rank, groups = 12, 4, 3
    basis = torch.linalg.qr(torch.randn(d, d)).Q
    readers = torch.stack((basis[:, 0], .2*basis[:, 0]+.98*basis[:, 1]))
    readers /= readers.norm(dim=1, keepdim=True)
    nodes = [dict(consumers=[0, 1, 2]), dict(consumers=[0, 2])]
    aa, ss, cc = [], [], []
    for g in range(groups):
        ids = [i for i, n in enumerate(nodes) if g in n['consumers']]
        private = basis[:, 2+3*g:2+3*g+rank-len(ids)].T
        b = torch.cat((readers[ids], private))
        h = torch.randn(rank, rank); h = (h+h.T)/2
        s, v = torch.linalg.eigh(h)
        aa.append(v.T @ b); ss.append(s); cc.append(torch.randn(5))
    original = (torch.stack(aa), torch.stack(ss), torch.stack(cc))
    graph = build(*original, readers, nodes)
    before, after = dense(*cp(*original)), dense(*cp(*factors(graph)))
    coefficient_error = float((before-after).norm()/before.norm())
    x = torch.randn(9, d)
    expected = torch.einsum('oij,ni,nj->no', before, x, x)
    executor_error = float((execute(graph, x)-expected).norm()/expected.norm())
    result = dict(coefficient_error=coefficient_error, executor_error=executor_error,
                  nonorthogonal_parent_cosine=float(readers[0]@readers[1]),
                  max_parent_condition=max(graph['parent_conditions']),
                  held=max(coefficient_error, executor_error)<=1e-10)
    assert result['held'], result
    return result


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    root = Path(__file__).parent
    small_control = control()
    print(json.dumps(dict(control=small_control)), flush=True)
    proposals = torch.load(root/'LL1_SUBSPACE_PARENTS_V1_PROPOSALS.pt', weights_only=True)
    # Native tensor metric is represented implicitly, including the entire U.
    checkpoint = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
    weights = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
    rows, graphs = [], {}
    for label in ('spectral', 'native'):
        saved = torch.load(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt', weights_only=True, map_location='cpu')
        original = tuple(v.double() for v in saved['parts'])
        proposal = proposals[label]
        graph = build(*original, proposal['readers'], proposal['nodes'])
        graph['output_whitener'] = saved['output_whitener'].double()
        graph['bias'] = weights['transformer.h.17.mlp.Down_bias'].clone()
        graph['source_sha256'] = hashlib.sha256(Path(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt').read_bytes()).hexdigest()
        graphs[label] = graph
        candidate = cp(*factors(graph)); before = cp(*original)
        # Check the graph executor independently of its coefficient conversion.
        torch.manual_seed(2602); x = torch.randn(11, original[0].shape[-1])
        a, b, w = candidate
        reference = ((x@a.T)*(x@b.T))@w.T
        replay = float((execute(graph, x)-reference).norm()/reference.norm())
        left, right, down = (weights[f'transformer.h.17.mlp.{key}.weight'].double()
                             for key in ('Left', 'Right', 'Down'))
        target = (left, right, saved['output_whitener'].double() @ down)
        total = 99245061353.47293
        delta = (torch.cat((candidate[0], before[0])), torch.cat((candidate[1], before[1])),
                 torch.cat((candidate[2], -before[2]), dim=1))
        change = float(inner(delta, delta)/total)
        before_capture = float((2*inner(target, before)-inner(before, before))/total)
        capture = float((2*inner(target, candidate)-inner(candidate, candidate))/total)
        costs = price(graph, original)
        row = dict(label=label, price=costs, max_parent_condition=max(graph['parent_conditions']),
                   executor_error=replay, squared_change_over_native=change,
                   before_capture=before_capture, graph_capture=capture,
                   capture_loss=before_capture-capture,
                   pred_a=replay<=1e-9 and small_control['held'],
                   pred_b=costs['saved_float_fraction']>=.01,
                   pred_c=change<=1e-3 and before_capture-capture<=1e-3)
        rows.append(row); print(json.dumps(row), flush=True)
    torch.save(graphs, root/'LL1_JOINT_PARENT_GRAPH_V1.pt')
    result = dict(control=small_control, rows=rows,
                  scope='Weight-only approximate graph of frozen unconverged LL1 pilot. Exact graph execution does not imply exact native reconstruction or identified behavioral circuits.')
    (root/'LL1_JOINT_PARENT_GRAPH_V1_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
