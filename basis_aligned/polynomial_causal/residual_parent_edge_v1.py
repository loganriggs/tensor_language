"""One residual-aware consumer insertion for frozen spectral shared parent 1.

Preserve all old parents and group rank. The private-space proposal uses the
quadratic marginal, not a globally optimal constrained subspace solve. Each
proposed core is the exact conditional minimizer in that proposed basis.
"""
import copy
import hashlib
import json
from pathlib import Path

import torch
from ll1_joint_core_solve_v1 import coordinates, System, solve, install
from ll1_joint_parent_graph_v2 import execute, factors, price
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner


def restricted_residual(z, group, bases, writers, cores, target):
    left, right = target[0] @ z, target[1] @ z
    weights = writers[group] @ target[2]
    raw = (left.T*weights) @ right
    residual = (raw+raw.T)/2
    cross = torch.einsum('hdi,dj->hij', bases, z)
    contracted = cross.transpose(1, 2) @ cores @ cross
    coefficients = writers @ writers[group]
    coefficients[group] = 0
    return residual-(contracted*coefficients[:, None, None]).sum(0)


def node_fraction(core, shared, parent_index):
    """Named-node deletion in actual nonorthogonal reader coordinates."""
    k = shared.shape[1]
    transform = torch.eye(len(core), dtype=core.dtype)
    transform[:k, :k] = shared.T
    inverse = torch.linalg.inv(transform)
    actual = inverse.T @ core @ inverse
    deleted = actual.clone()
    deleted[parent_index, :] = 0
    deleted[:, parent_index] = 0
    difference = transform.T @ (actual-deleted) @ transform
    return float(difference.square().sum()/core.square().sum())


def propose(graph, g, parent, bases, writers, cores, target, penalty):
    old = graph['groups'][g]
    ids = sorted(old['parent_ids'].tolist()+[parent])
    rank = bases.shape[-1]
    rr = graph['readers'][ids]
    shared, change = torch.linalg.qr(rr.T, mode='reduced')
    if float(torch.linalg.cond(change)) > 1e6:
        return None
    residual = bases[g]-shared@(shared.T@bases[g])
    vectors, singular, _ = torch.linalg.svd(residual, full_matrices=False)
    private_span = vectors[:, singular>singular.max()*1e-10]
    size = rank-len(ids)
    if size < 0 or private_span.shape[1] < size:
        return None
    z = torch.cat((shared, private_span), 1)
    s = restricted_residual(z, g, bases, writers, cores, target)
    # Marginal includes shared/private terms; this is still a proposal rule.
    marginal = (s@s)[len(ids):, len(ids):]
    _, eig = torch.linalg.eigh(marginal)
    selected = eig[:, -size:] if size else eig[:, :0]
    private = private_span @ selected
    new_basis = torch.cat((shared, private), 1)
    embedding = z.T @ new_basis
    rhs = embedding.T @ s @ embedding
    core = rhs/(1+penalty)
    old_embedding = z.T @ bases[g]
    old_rhs = old_embedding.T @ s @ old_embedding
    old_phi = (1+penalty)*cores[g].square().sum()-2*(old_rhs*cores[g]).sum()
    baseline_phi = -old_rhs.square().sum()/(1+penalty)
    new_phi = -rhs.square().sum()/(1+penalty)
    return dict(group=g, ids=ids, private=private.T, core=core,
                conditional_delta=new_phi-old_phi,
                delta_vs_polish=new_phi-baseline_phi,
                polish_gain=old_phi-baseline_phi,
                node_energy_fraction=node_fraction(core, change, ids.index(parent)),
                orthogonality=float((new_basis.T@new_basis-torch.eye(rank)).abs().max()),
                old_span_replay=float((bases[g]-z@(z.T@bases[g])).norm()/bases[g].norm()),
                core_normal_residual=float(((1+penalty)*core-rhs).norm()/rhs.norm()),
                union_dimension=z.shape[1])


def install_edge(graph, proposal, cores, writers):
    result = copy.deepcopy(graph)
    group = result['groups'][proposal['group']]
    ids = proposal['ids']
    pairs = [tuple(row) for row in result['pairs'].tolist()]
    pair_ids = []
    for i, a in enumerate(ids):
        for b in ids[i:]:
            pair = (a, b)
            if pair not in pairs:
                pairs.append(pair)
            pair_ids.append(pairs.index(pair))
    result['pairs'] = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    group['parent_ids'] = torch.tensor(ids, dtype=torch.long)
    group['pair_ids'] = torch.tensor(pair_ids, dtype=torch.long)
    group['private'] = proposal['private']
    group['lam'] = torch.zeros(len(group['private']))
    group['cross'] = torch.zeros(len(ids), len(group['private']))
    group['shared_coeff'] = torch.zeros(len(pair_ids))
    next_cores = cores.clone()
    next_cores[proposal['group']] = proposal['core']
    return install(result, next_cores, writers)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    root = Path(__file__).parent
    out = root/'RESIDUAL_PARENT_EDGE_V1.json'
    assert not out.exists()
    source = root/'SHARED_READER_JOINT_FIT_V1_SPECTRAL_GRAPH.pt'
    graph = torch.load(source, weights_only=True, map_location='cpu')
    binding = json.loads((root/'SHARED_NODE_PARENT1_FINEWEB_V1_BINDING.json').read_text())['files']
    checkpoint = next(name for name in binding if name.endswith('/pytorch_model.bin'))
    weights = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
    l, r, d = [weights[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left', 'Right', 'Down')]
    target = (l, r, graph['output_whitener'] @ d)
    total, penalty, parent = 99245061353.47293, .01, 1
    bases, writers, cores = coordinates(graph)
    old_parts = factors(graph)
    old_price = price(graph, old_parts)
    proposals, rows = [], []
    for g, group in enumerate(graph['groups']):
        if parent in group['parent_ids'].tolist():
            continue
        proposal = propose(graph, g, parent, bases, writers, cores, target, penalty)
        if proposal is None:
            continue
        row = {key: value for key, value in proposal.items() if key not in ('core', 'private')}
        for key in ('conditional_delta', 'delta_vs_polish', 'polish_gain'):
            row[key] = float(row[key]/total)
        proposals.append(proposal)
        rows.append(row)
    assert proposals
    eligible = [i for i, row in enumerate(rows) if row['node_energy_fraction']>=.01]
    selected = min(eligible or range(len(rows)), key=lambda i: rows[i]['delta_vs_polish'])
    proposal = proposals[selected]
    candidate = install_edge(graph, proposal, cores, writers)
    candidate_parts = factors(candidate)
    new_price = price(candidate, old_parts)
    saved_floats = old_price['graph_floats']-new_price['graph_floats']
    before, after = cp(*old_parts), cp(*candidate_parts)
    delta = (torch.cat((after[0], before[0])), torch.cat((after[1], before[1])),
             torch.cat((after[2], -before[2]), 1))
    # Use only the changed group for independent coefficient-difference scoring.
    g = proposal['group']
    one_before = cp(*(part[g:g+1] for part in old_parts))
    one_after = cp(*(part[g:g+1] for part in candidate_parts))
    small_delta = (torch.cat((one_after[0], one_before[0])),
                   torch.cat((one_after[1], one_before[1])),
                   torch.cat((one_after[2], -one_before[2]), 1))
    measured_delta = (-2*inner(target, small_delta)+2*inner(before, small_delta)+
                      inner(small_delta, small_delta)+penalty*(proposal['core'].square().sum()-cores[g].square().sum()))/total
    x = torch.randn(8, 1152, generator=torch.Generator().manual_seed(5201))
    expected = ((x@after[0].T)*(x@after[1].T))@after[2].T
    replay = float((execute(candidate, x)-expected).norm()/expected.norm())
    actual_delta = execute(candidate, x)-execute(graph, x)
    small_expected = ((x@small_delta[0].T)*(x@small_delta[1].T))@small_delta[2].T
    unchanged_error = float((actual_delta-small_expected).norm()/execute(graph, x).norm())
    errors = dict(objective_delta=abs(float(measured_delta)-rows[selected]['conditional_delta']),
                  executor=replay, unchanged_groups=unchanged_error,
                  orthogonality=max(row['orthogonality'] for row in rows),
                  old_span=max(row['old_span_replay'] for row in rows),
                  conditional_normal=max(row['core_normal_residual'] for row in rows))
    print(json.dumps(dict(selected=rows[selected], saved_floats=saved_floats, errors=errors)), flush=True)
    fit_rows, final_graph = [], None
    for name, item in [('baseline', graph), ('new_edge', candidate)]:
        bb, cc, hh = coordinates(item)
        system = System(bb, cc, target, penalty)
        fitted, stats = solve(system, hh, tolerance=1e-11, seconds=120)
        objective = float(1+(-2*(system.rhs*fitted).sum()+(fitted*system.apply(fitted)).sum())/total)
        fit_rows.append(dict(name=name, objective=objective, **stats))
        if name=='new_edge':
            final_graph = install(item, fitted, cc)
            ids = final_graph['groups'][g]['parent_ids'].tolist()
            change = torch.linalg.qr(item['readers'][ids].T, mode='reduced').R
            final_fraction = node_fraction(fitted[g], change, ids.index(parent))
    delta_after = fit_rows[1]['objective']-fit_rows[0]['objective']
    errors['joint_core_normal'] = max(row['normal_residual'] for row in fit_rows)
    valid = all(value<=1e-9 for value in errors.values())
    held_b = valid and rows[selected]['node_energy_fraction']>=.01 and rows[selected]['delta_vs_polish']<=1e-6 and saved_floats>=1000
    artifact = root/'RESIDUAL_PARENT_EDGE_V1.pt'
    torch.save(final_graph, artifact)
    result = dict(pred_a=valid, pred_b=held_b,
                  pred_c=held_b and delta_after<=1e-6 and final_fraction>=.01,
                  errors=errors, parent=parent, selected=rows[selected], candidates=rows,
                  candidate_count=len(rows), effective_candidates=len(eligible),
                  saved_floats=saved_floats, old_price=old_price, new_price=new_price,
                  matched_core_solves=fit_rows, objective_delta_after_joint_solve=delta_after,
                  final_new_consumer_node_energy_fraction=final_fraction,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                  scope='One residual-aware edge insertion into a frozen graph. No topology convergence, globally optimal private space, data fitting, native behavioral or circuit-identification claim.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key!='candidates'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
