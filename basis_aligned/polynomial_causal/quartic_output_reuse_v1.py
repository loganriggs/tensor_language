"""Fixed-program output-gauge diagnostic; no native data fitting.

Preregistered A: orthogonal gauge invariance <=1e-8.
B: V3 best-axis nodewise minority energy >=.05.
C: V3 exclusive refit coefficient error >=.1.
These do not certify semantic reuse or globally optimal exclusive assignment.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from coupled_quartic_writer_v1 import gram, solve

P = Path(__file__).resolve().parent

def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)

def diagnose(k, a):
    d = k.diagonal()
    total = float((d[:, None] * a.square()).sum())
    def loss(angle):
        return float((d * (a @ rotation(angle)).square().min(-1).values).sum()) / total
    grid = np.linspace(0, math.pi / 2, 4096, endpoint=False)
    losses = [loss(t) for t in grid]
    idx = int(np.argmin(losses)); step = math.pi / 8192
    fit = minimize_scalar(loss, bounds=(grid[idx]-step, grid[idx]+step), method='bounded', options={'xatol': 1e-14})
    rotated = a @ rotation(fit.x)
    assignment = rotated.abs().argmax(-1)
    exclusive = torch.zeros_like(a)
    cross = k @ rotated
    for branch in range(2):
        ids = torch.where(assignment == branch)[0]
        if len(ids):
            coef, _ = solve(k[ids][:, ids], cross[ids, branch:branch+1])
            exclusive[ids, branch] = coef[:, 0]
    delta = exclusive - rotated
    energy = float((rotated * (k @ rotated)).sum())
    err = math.sqrt(max(0., float((delta * (k @ delta)).sum())) / energy)
    return dict(angle=float(fit.x), minority_node_energy=loss(fit.x), exclusive_refit_relative_error=err,
                assigned_counts=[int((assignment == b).sum()) for b in range(2)],
                node_energy_sum_over_program_energy=total / energy)

def main():
    torch.set_num_threads(2)
    reports = []
    for version in (1, 2, 3):
        path = P / f'COUPLED_QUARTIC_NONLINEAR_V{version}_PROGRAM.pt'
        program = torch.load(path, weights_only=True)
        k = gram(program['input_readers'], program['inner_weights'])
        a = program['mixing']
        report = diagnose(k, a)
        rotated = diagnose(k, a @ rotation(.371))
        report['rotation_error'] = abs(report['minority_node_energy'] - rotated['minority_node_energy'])
        report['version'] = version
        report['program_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        reports.append(report)
    result = dict(pred_a=max(r['rotation_error'] for r in reports) <= 1e-8,
                  pred_b=reports[-1]['minority_node_energy'] >= .05,
                  pred_c=reports[-1]['exclusive_refit_relative_error'] >= .1,
                  reports=reports,
                  scope='Fixed 32-node program, common orthogonal two-output rotation. Nodewise penalty ignores cancellation; exclusive refit uses full Gram. Assignment minimizes nodewise penalty, not global refitted error. No semantic or native causal claim.')
    out = P / 'QUARTIC_OUTPUT_REUSE_V1_RESULT.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert result['pred_a']

if __name__ == '__main__':
    main()
