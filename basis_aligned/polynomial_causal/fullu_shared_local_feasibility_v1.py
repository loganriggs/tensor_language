"""Exact coefficient-metric prices/bounds; no native local-subspace fit or text.

Use the existing product Gram. Center vocabulary rows, retain the mean exactly.
Compare a global-plus-single-group-private decoder with a global SVD decoder.
The union-span bound relaxes group sparsity, so is optimistic, not achievable error.
"""
import json
import time
from pathlib import Path
import torch
from audit_fullu_output_functions_v1 import CK
from joint_quadratic_fit_v1 import product_cross

P = Path(__file__).resolve().parent


def price(v, d, g, groups, local):
    # Explicit mean, all reader banks, global/private token coefficients, int32 IDs.
    floats = d + (g + groups * local) * d + v * (g + local)
    nbytes = 4 * floats + 4 * v
    flat_rank = min(d, (nbytes - 4 * d) // (4 * (d + v)))
    return dict(float_parameters=floats, assignment_int32=v, total_bytes=nbytes,
                matched_global_rank=int(flat_rank), total_bank_width=g + groups * local,
                active_coefficients_per_token=g + local)


def control():
    torch.manual_seed(9317)
    dt = torch.float64
    u, l, r, down = [torch.randn(*s, dtype=dt) for s in [(11,4),(7,5),(7,5),(4,7)]]
    metric = down @ product_cross(l,r,l,r) @ down.T
    root = torch.linalg.cholesky((metric + metric.T) / 2)
    x = u @ root
    h = (l[:,:,None] * r[:,None,:] + r[:,:,None] * l[:,None,:]) / 2
    dense = torch.einsum('vk,kij->vij', u @ down, h).flatten(1)
    metric_error = float((x @ x.T - dense @ dense.T).norm() / (dense @ dense.T).norm())
    bank = torch.randn(3,4,dtype=dt)
    # bank @ root^{-1}, without saving a dense inverse in the compiled program.
    readers = torch.linalg.solve_triangular(root.T,bank.T,upper=True).T
    reconstructed = torch.einsum('ak,kij->aij', readers @ down, h).flatten(1)
    compiled_error = float((reconstructed @ reconstructed.T - bank @ bank.T).norm() / (bank @ bank.T).norm())

    # A true shared/private fixture: the union has width 18, each token uses only 4.
    v, d, g, groups, local = 2048, 32, 2, 8, 2
    q = torch.linalg.qr(torch.randn(d,d,dtype=dt)).Q.T
    labels = torch.arange(v) % groups
    gc = torch.randn(v,g,dtype=dt)
    lc = torch.randn(v,local,dtype=dt)
    fixture = gc @ q[:g]
    predicted = fixture.clone()
    for k in range(groups):
        ix = labels == k
        contribution = lc[ix] @ q[g+k*local:g+(k+1)*local]
        fixture[ix] += contribution
        predicted[ix] += contribution
    costs = price(v,d,g,groups,local)
    ev = torch.linalg.svdvals(fixture - fixture.mean(0)).square()
    global_error = float((ev[costs['matched_global_rank']:].sum() / fixture.square().sum()).sqrt())
    assert max(metric_error,compiled_error) < 1e-11
    assert global_error > .25
    return dict(dense_metric_relative_error=metric_error,compiled_reader_metric_error=compiled_error,
                planted_exact_representation_error=float((fixture-predicted).norm()),
                planted_matched_global_relative_error=global_error,planted_price=costs,
                caveat='Known planted banks/labels demonstrate representational advantage, not solver recovery.')


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    checks = control()
    sd = torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u = sd['lm_head.weight'].double()
    l,r,down = [sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ['Left','Right','Down']]
    v,d = u.shape
    gram = product_cross(l,r,l,r)
    metric = down @ gram @ down.T
    metric = (metric + metric.T) / 2
    root = torch.linalg.cholesky(metric)
    mean = u.mean(0)
    centered_gram = u.T @ u - v * mean[:,None] * mean[None,:]
    covariance = root.T @ centered_gram @ root
    ev = torch.linalg.eigvalsh((covariance+covariance.T)/2).flip(0)
    centered_energy = float(ev.sum())
    mean_energy = float(v * (mean @ metric @ mean))
    total = centered_energy + mean_energy
    prior = json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())
    trace_error = abs(total / prior['native_total'] - 1)
    assert trace_error < 1e-10
    assert float(ev.min()) >= -1e-12 * float(ev.max())
    def error(rank):
        return float((ev[min(rank,d):].clamp_min(0).sum() / total).sqrt())
    rows = []
    for g,groups,local in [(32,16,4),(64,16,8),(64,32,8),(128,32,16),(128,64,16)]:
        costs = price(v,d,g,groups,local)
        rows.append(dict(global_width=g,groups=groups,private_width=local,**costs,
                         bytes_fraction_of_native_U=costs['total_bytes']/(4*v*d),
                         optimal_matched_global_relative_error=error(costs['matched_global_rank']),
                         optimistic_union_span_error_lower_bound=error(costs['total_bank_width'])))
    result = dict(schema='fullu.shared.local.feasibility.v1',controls=checks,
                  native_shape=[v,d],native_U_float_parameters=v*d,
                  coefficient_energy=total,mean_energy_fraction=mean_energy/total,
                  centered_trace_replay_error=trace_error,configurations=rows,
                  wall_seconds=time.perf_counter()-started,body_forwards=0,
                  scope='Uncentered full-token quadratic coefficient Frobenius error; mean retained exactly. '
                  'Only output-map storage is priced here; native L/R/Down and nonlinear execution remain. '
                  'No native grouped fit, behavioral result, or arithmetic lower bound.')
    destination = P/'FULLU_SHARED_LOCAL_FEASIBILITY_V1_RESULT.json'
    with destination.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
