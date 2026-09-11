"""Managed matched fits of one closed graph component, full native weight residual."""
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
P = ROOT/'basis_aligned/polynomial_causal'
sys.path[:0] = [str(P), str(ROOT)]
import torch
from closed_component_refit_v1 import prepare, inject, GROUPS, PARENTS, TOTAL, PENALTY
from bounded_shared_reader_fit_v1 import fit
from shared_reader_variable_projection_v2 import Objective
from shared_reader_joint_fit_v1 import cpu, digest
from ll1_joint_parent_graph_v2 import execute, factors, price
from ll1_joint_core_solve_v1 import coordinates
from symmetric_ll1_objective_v1 import cp
from residual_parent_edge_v1 import node_fraction

PREFIX = 'CLOSED_COMPONENT_REFIT_V1'


def aggregate():
    paths = [P/f'{PREFIX}_{name.upper()}.json' for name in ('original', 'new_edge')]
    if not all(path.exists() for path in paths):
        return
    original, changed = [json.loads(path.read_text()) for path in paths]
    delta = changed['whole_objective']-original['whole_objective']
    saving = original['price']['graph_floats']-changed['price']['graph_floats']
    roles = changed['parent1_consumer_energy_fractions']
    a = bool(original['pred_a'] and changed['pred_a'])
    b = bool(original['pred_b'] and changed['pred_b'])
    c = a and delta<=1e-6 and saving>=1000 and all(roles[str(i)]>=.01 for i in (11, 16, 25))
    result = dict(pred_a=a, pred_b=b, pred_c=c,
                  new_edge_minus_original_objective=delta, saved_floats=saving,
                  parent1_consumer_energy_fractions=roles,
                  scope='Matched local weight refits with fixed outside graph and common output span. A numerics, B both local convergence, C objective/price/effective-consumer comparison. No behavioral adoption or global recovery.')
    with (P/f'{PREFIX}_AGGREGATE.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(aggregate=result)), flush=True)


def main(name):
    assert name in ('original', 'new_edge')
    binding = json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path, sha in binding.items())
    preflight = json.loads((P/f'{PREFIX}_PREFLIGHT.json').read_text())
    assert preflight['pred_a']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False, gpu_accessed=False, body_forwards=0,
                              corpus_access=False, name=name, groups=GROUPS, parents=PARENTS,
                              output_span=5, rank=16, seconds=1200, penalty=PENALTY,
                              reencode_every=200, fresh_gradient_bar=1e-7, progress_bar=1e-6)))
        return
    out = P/f'{PREFIX}_{name.upper()}.json'
    artifact = out.with_suffix('.pt')
    assert not out.exists() and not artifact.exists()
    signal.alarm(1800)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    background, basis, local, target, _ = prepare(P, device='cuda')
    graph = local[name]
    expected = next(row for row in preflight['rows'] if row['name']==name)['shifted_initial_objective']
    start = time.perf_counter()
    check = Objective(target, graph, TOTAL, PENALTY)
    initial, _ = check.evaluate(check.initial)
    torch.cuda.synchronize()
    gpu_initial_seconds = time.perf_counter()-start
    initial_error = abs(initial-expected)
    assert initial_error<=1e-8
    del check
    fitted, stats = fit(target, graph, TOTAL, penalty=PENALTY, seconds=1200,
                        max_cycles=1000, iterations_per_cycle=200)
    with torch.no_grad():
        bank = cp(*factors(fitted))
        torch.manual_seed(5302)
        x = torch.randn(9, 1152, device='cuda')
        expected_values = ((x@bank[0].T)*(x@bank[1].T))@bank[2].T
        replay = float((execute(fitted, x)-expected_values).norm()/expected_values.norm())
        fitted_cpu = cpu(fitted)
        full = inject(background, fitted_cpu, basis)
        xx = x.cpu()
        expected_full = execute(background, xx)+execute(fitted_cpu, xx)@basis.T-execute(local['original'], xx)@basis.T
        injection = float((execute(full, xx)-expected_full).norm()/expected_full.norm())
        bb, cc, hh = coordinates(full)
        roles = {}
        for index in (11, 16, 25):
            ids = full['groups'][index]['parent_ids'].tolist()
            if 1 in ids:
                change = torch.linalg.qr(full['readers'][ids].T, mode='reduced').R
                roles[str(index)] = node_fraction(hh[index], change, ids.index(1))
            else:
                roles[str(index)] = 0.
        cosine = float(abs(full['readers'][1]@background['readers'][1]) /
                       full['readers'][1].norm()/background['readers'][1].norm())
    details = stats['details']
    numeric = (max(initial_error, replay, injection, details['reduced_identity_error'],
                   stats['maximum_reencoding_objective_error'], stats['maximum_objective_increase'])<=1e-8
               and details['inner']['normal_residual']<=1e-10
               and stats['maximum_raw_entry']<=1+1e-10)
    whole = stats['final']+preflight['full_objective_offset']
    # Preserve legacy quantities with explicit names preventing whole-capture claims.
    stats['shifted_local_capture'] = stats.pop('capture')
    stats['details']['shifted_residual'] = stats['details'].pop('residual')
    stats['history_objective_scope'] = 'Shifted local objective; add full_objective_offset for whole objective.'
    full['local_refit'] = dict(groups=GROUPS, parents=PARENTS, source=name,
                              fixed_output_span=5, native_data_fitting=False)
    torch.save(full, artifact)
    result = dict(pred_a=bool(numeric), pred_b=bool(stats['local_converged']),
                  pred_c=bool(stats['initial']-stats['final']>=1e-6), name=name,
                  whole_objective=whole, full_objective_offset=preflight['full_objective_offset'],
                  initial_cpu_gpu_error=initial_error, gpu_initial_seconds=gpu_initial_seconds,
                  executor_relative_error=replay, injection_relative_error=injection,
                  parent1_consumer_energy_fractions=roles, parent1_reader_cosine_before_after=cosine,
                  price=price(full, factors(background)), optimization=stats,
                  artifact=dict(path=str(artifact), sha256=digest(artifact)),
                  scope='Closed five-group local graph fit; other59groups frozen, output directions restricted to their original five-dimensional span. Per-arm C is objective gain>=1e-6; aggregate C tests graph move. No corpus, native body forwards, global convergence or circuit identification.')
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key!='optimization'}), flush=True)
    aggregate()
    return result
