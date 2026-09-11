"""Exact conditional weight problem for a closed set of graph consumers.

Outside groups are frozen. Selected writers vary only within their common
original span. The legacy objective's leading 1 is retained, so its value is
shifted by a known constant from the full graph objective; do not call its
reported `capture` whole-model coefficient capture.
"""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from ll1_joint_parent_graph_v2 import factors, execute
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner
from shared_reader_variable_projection_v2 import Objective

GROUPS = [10, 11, 16, 19, 25]
PARENTS = [1, 8, 9]
TOTAL = 99245061353.47293
PENALTY = .01


def extract(graph, output_basis):
    ids = {g: i for i, g in enumerate(PARENTS)}
    pairs, groups = [], []
    for index in GROUPS:
        group = copy.deepcopy(graph['groups'][index])
        group['parent_ids'] = torch.tensor([ids[int(i)] for i in group['parent_ids']], dtype=torch.long)
        new_pair_ids = []
        for pair_id in group['pair_ids']:
            pair = tuple(ids[int(i)] for i in graph['pairs'][pair_id])
            if pair not in pairs:
                pairs.append(pair)
            new_pair_ids.append(pairs.index(pair))
        group['pair_ids'] = torch.tensor(new_pair_ids, dtype=torch.long)
        group['writer'] = output_basis.T @ group['writer']
        groups.append(group)
    return dict(readers=graph['readers'][PARENTS].clone(), groups=groups,
                pairs=torch.tensor(pairs, dtype=torch.long).reshape(-1, 2))


def inject(background, local, output_basis):
    result = copy.deepcopy(background)
    for index, group in enumerate(background['groups']):
        if index not in GROUPS:
            assert not (set(group['parent_ids'].tolist()) & set(PARENTS))
    result['readers'][PARENTS] = local['readers']
    pairs = [tuple(pair) for pair in result['pairs'].tolist()]
    for index, source in zip(GROUPS, local['groups']):
        group = copy.deepcopy(source)
        group['parent_ids'] = torch.tensor([PARENTS[int(i)] for i in group['parent_ids']], dtype=torch.long)
        pair_ids = []
        for pair_id in group['pair_ids']:
            pair = tuple(PARENTS[int(i)] for i in local['pairs'][pair_id])
            if pair not in pairs:
                pairs.append(pair)
            pair_ids.append(pairs.index(pair))
        group['pair_ids'] = torch.tensor(pair_ids, dtype=torch.long)
        group['writer'] = output_basis @ group['writer']
        result['groups'][index] = group
    result['pairs'] = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    # Construction-time pilot diagnostics are not diagnostics of the new graph.
    for key in ('parent_conditions', 'minimum_joint_memberships', 'nodes'):
        result.pop(key, None)
    return result


def prepare(root, device='cpu'):
    source = root/'SHARED_READER_JOINT_FIT_V1_SPECTRAL_GRAPH.pt'
    background = torch.load(source, weights_only=True, map_location='cpu')
    changed = torch.load(root/'RESIDUAL_PARENT_EDGE_V2.pt', weights_only=True, map_location='cpu')
    writers = torch.stack([background['groups'][i]['writer'] for i in GROUPS])
    output_basis, change = torch.linalg.qr(writers.T, mode='reduced')
    assert float(torch.linalg.cond(change))<1e6
    locals_ = {name: extract(graph, output_basis) for name, graph in [('original', background), ('new_edge', changed)]}
    parts = factors(background)
    outside = [i for i in range(len(background['groups'])) if i not in GROUPS]
    frozen = cp(*(part[outside] for part in parts))
    binding = json.loads((root/'SHARED_NODE_PARENT1_FINEWEB_V1_BINDING.json').read_text())['files']
    checkpoint = next(name for name in binding if name.endswith('/pytorch_model.bin'))
    weights = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
    left, right, down = [weights[f'transformer.h.17.mlp.{key}.weight'].double().to(device) for key in ('Left', 'Right', 'Down')]
    native_writer = background['output_whitener'].to(device) @ down
    basis = output_basis.to(device)
    target = (torch.cat((left, frozen[0].to(device))), torch.cat((right, frozen[1].to(device))),
              torch.cat((basis.T@native_writer, -basis.T@frozen[2].to(device)), 1))
    return background, output_basis, locals_, target, (left, right, native_writer)


def difference(a, b):
    return torch.cat((a[0], b[0])), torch.cat((a[1], b[1])), torch.cat((a[2], -b[2]), 1)


def component_energy(graph):
    a, s, c = factors(graph)
    overlaps = a @ a.transpose(1, 2)
    return ((overlaps.square()*s[:, :, None]*s[:, None, :]).sum((1, 2))*c.square().sum(1)).sum()


def preflight():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    root = Path(__file__).parent
    out = root/'CLOSED_COMPONENT_REFIT_V1_PREFLIGHT.json'
    assert not out.exists()
    background, basis, local, target, native = prepare(root)
    before = cp(*factors(background))
    old_energy = component_energy(background)
    torch.manual_seed(5301)
    x = torch.randn(7, 1152)
    rows, shifted_reference = [], None
    source_objective = json.loads((root/'SHARED_READER_JOINT_FIT_V1_SPECTRAL_GRAPH.json').read_text())['optimization']['final']
    for name, graph in local.items():
        start = time.perf_counter()
        objective = Objective(target, graph, TOTAL, PENALTY)
        value, gradient = objective.evaluate(objective.initial)
        elapsed = time.perf_counter()-start
        fitted = objective.physical(objective.initial)
        full = inject(background, fitted, basis)
        if name=='original':
            shifted_reference = value
        new = cp(*factors(full))
        delta = difference(new, before)
        full_delta = (-2*inner(native, delta)+2*inner(before, delta)+inner(delta, delta)+
                      PENALTY*(component_energy(full)-old_energy))/TOTAL
        expected = execute(background, x)+execute(fitted, x)@basis.T-execute(local['original'], x)@basis.T
        injection_error = float((execute(full, x)-expected).norm()/expected.norm())
        direction = gradient/np.linalg.norm(gradient)
        epsilon = 1e-5
        plus = objective.evaluate(objective.initial+epsilon*direction)[0]
        minus = objective.evaluate(objective.initial-epsilon*direction)[0]
        analytic = float(gradient@direction)
        fd = (plus-minus)/(2*epsilon)
        fd_error = abs(fd-analytic)/max(abs(analytic), 1e-12)
        row = dict(name=name, shifted_initial_objective=value,
                   whole_initial_objective=source_objective+float(full_delta),
                   whole_local_delta_error=abs(float(full_delta)-(value-shifted_reference)),
                   injection_relative_error=injection_error, directional_fd_relative_error=fd_error,
                   gradient_inf=float(np.abs(gradient).max()), initial_evaluation_cpu_seconds=elapsed,
                   reduced_identity_error=objective.last['reduced_identity_error'])
        rows.append(row)
    numeric = max(max(r['whole_local_delta_error'], r['injection_relative_error'], r['reduced_identity_error']) for r in rows)
    fd = max(r['directional_fd_relative_error'] for r in rows)
    result = dict(pred_a=numeric<=1e-9 and fd<=1e-5, rows=rows,
                  full_objective_offset=source_objective-shifted_reference,
                  groups=GROUPS, parents=PARENTS, native_output_span=basis.shape[1],
                  source_sha256=hashlib.sha256((root/'SHARED_READER_JOINT_FIT_V1_SPECTRAL_GRAPH.pt').read_bytes()).hexdigest(),
                  scope='Exact conditional local/full objective differences, fixed outside graph and fixed output span. Shifted local capture is not whole-graph capture. No optimizer convergence or circuit claim.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    preflight()
