"""Global 2D squared-loading overlap minimum by sorted sinusoid sign events.

For F=[a,b], minimize sum min((FR)_0^2,(FR)_1^2). Its variable
part is -sum |(a^2-b^2) cos(phi) + 2ab sin(phi)|, phi=2 theta.
Within every sign interval this is one sinusoid. Enumerating boundaries
and interior maxima gives a global solution in O(V log V) time.
"""
import hashlib
import json
import math
from pathlib import Path

import torch


def optimize(f):
    a, b = f.unbind(1)
    coefficients = torch.stack((a.square()-b.square(), 2*a*b), 1)
    coefficients = coefficients[coefficients.square().sum(1)>0]
    roots = torch.remainder(torch.atan2(coefficients[:, 1], coefficients[:, 0])+math.pi/2, math.pi)
    selected = (roots>0) & (roots<math.pi)
    events, order = roots[selected].sort()
    reference = float(events[0]/2) if len(events) else math.pi/2
    signs = (coefficients[:, 0]*math.cos(reference)+coefficients[:, 1]*math.sin(reference)).sign()
    initial = (coefficients*signs[:, None]).sum(0)
    jumps = -2*coefficients[selected][order]*signs[selected][order, None]
    sums = torch.cat((initial[None], initial[None]+jumps.cumsum(0)), 0)
    boundaries = torch.cat((torch.zeros(1), events, torch.tensor([math.pi])))
    left, right = boundaries[:-1], boundaries[1:]
    maximum_angle = torch.remainder(torch.atan2(sums[:, 1], sums[:, 0]), 2*math.pi)
    valid = (maximum_angle>=left) & (maximum_angle<=right)
    candidates = torch.stack((left, right, torch.where(valid, maximum_angle, left)), 1)
    values = sums[:, :1]*candidates.cos()+sums[:, 1:]*candidates.sin()
    index = int(values.argmax())
    phi = candidates.reshape(-1)[index]
    theta = phi/2
    co, si = theta.cos(), theta.sin()
    rotation = torch.stack((torch.stack((co, -si)), torch.stack((si, co))))
    direct = (coefficients[:, 0]*phi.cos()+coefficients[:, 1]*phi.sin()).abs().sum()
    relative = float((direct-values.reshape(-1)[index]).abs()/direct.clamp_min(1e-30))
    return rotation, dict(angle_degrees=float(theta*180/math.pi),
                          intervals=len(sums), analytic_relative_error=relative)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    root = Path(__file__).parent
    target = root / 'BRANCH_OVERLAP_ROTATION_V1.json'
    assert not target.exists()
    source = root / 'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt'
    saved = torch.load(source, weights_only=True, map_location='cpu')
    node = saved['nodes'][1]
    c, p = node['writers'].double(), node['partners'].double()
    scale = c.norm(dim=0)
    o, k = c/scale, p*scale
    binding = json.loads((root/'SHARED_NODE_PARENT1_FINEWEB_V1_BINDING.json').read_text())['files']
    checkpoint = next(name for name in binding if name.endswith('/pytorch_model.bin'))
    weights = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
    physical = torch.linalg.solve_triangular(saved['output_whitener'].double(), o, upper=True)
    f = weights['lm_head.weight'].double() @ physical
    rotation, stats = optimize(f)
    rotated = f @ rotation
    old_overlap = f.square().min(1).values.sum()
    new_overlap = rotated.square().min(1).values.sum()
    # Independent direct finite-angle evaluations cannot beat the global optimum.
    grid = torch.linspace(0, math.pi/2, 129)
    grid_overlap = []
    for theta in grid:
        co, si = theta.cos(), theta.sin()
        z = torch.stack((f[:, 0]*co+f[:, 1]*si, -f[:, 0]*si+f[:, 1]*co), 1)
        grid_overlap.append(z.square().min(1).values.sum())
    base = torch.zeros(12, 2)
    base[:6, 0] = torch.arange(1, 7)
    base[6:, 1] = torch.arange(2, 8)
    angle = .371
    mixing = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    planted = base @ mixing
    recovery, planted_stats = optimize(planted)
    recovered_overlap = float((planted @ recovery).square().min(1).values.sum()/base.square().sum())
    new_c, new_p = o @ rotation, k @ rotation
    old_m, new_m = c @ p.T, new_c @ new_p.T
    errors = dict(analytic=stats['analytic_relative_error'],
                  full_node_matrix=float((new_m-old_m).norm()/old_m.norm()),
                  orthogonality=float((rotation.T@rotation-torch.eye(2)).abs().max()),
                  grid_violation=float((new_overlap-torch.stack(grid_overlap).min()).clamp_min(0)/old_overlap),
                  planted_overlap=recovered_overlap,
                  planted_analytic=planted_stats['analytic_relative_error'])
    valid = all(math.isfinite(v) and v<=1e-9 for v in errors.values())
    reduction = float(1-new_overlap/old_overlap)
    moment_ratio = float(rotated.pow(4).sum()/f.pow(4).sum())
    artifact = root/'BRANCH_OVERLAP_ROTATION_V1.pt'
    torch.save(dict(reader=node['reader'], writers=new_c, partners=new_p,
                    rotation=rotation, output_whitener=saved['output_whitener']), artifact)
    result = dict(pred_a=valid, pred_b=valid and reduction>=.25,
                  pred_c=valid and moment_ratio>=.95, errors=errors, optimizer=stats,
                  overlap_before=float(old_overlap), overlap_after=float(new_overlap),
                  overlap_relative_reduction=reduction, fourth_moment_ratio=moment_ratio,
                  varimax=json.loads((root/'BRANCH_OUTPUT_ROTATION_V1.json').read_text())['overlap_after'],
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                  scope='Global minimum only within orthogonal rotations of this frozen, unit full-U two-writer subspace. No optimization over the shared reader, subspace, nonorthogonal rescalings, or model data. Summed node unchanged; individual deletions change.')
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
