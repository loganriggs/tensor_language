"""CPU control of existing conditional writer solve; predictions on AGENT_BOARD.
Explicit coefficient tensors independently check implicit Gram calculations.
"""
import json
from pathlib import Path
import torch
from joint_quadratic_fit_v1 import optimal_writers, product_cross, implicit_squared_error

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
torch.manual_seed(749)
l, r = torch.randn(11, 7), torch.randn(11, 7)
d = torch.randn(5, 11)
rows = []
for duplicate in (False, True):
    a, b = torch.randn(8, 7), torch.randn(8, 7)
    if duplicate:
        a[-1], b[-1] = a[0], b[0]
    w, cross, gram = optimal_writers(d, l, r, a, b)
    native = (l[:, :, None]*r[:, None, :] + r[:, :, None]*l[:, None, :])/2
    candidate = (a[:, :, None]*b[:, None, :] + b[:, :, None]*a[:, None, :])/2
    target = (d @ native.flatten(1))
    residual = target-w @ candidate.flatten(1)
    normal = float(torch.linalg.norm(residual @ candidate.flatten(1).T)
                   / torch.linalg.norm(target @ candidate.flatten(1).T))
    for rank in (2, 5):
        u = torch.randn(rank, 5)
        metric = u.T @ u
        explicit = (u @ residual).square().sum()
        implicit = implicit_squared_error(metric, d, product_cross(l,r,l,r), w, cross, gram)
        error = float(abs(explicit-implicit)/explicit)
        gains = []
        identity_errors = []
        for _ in range(20):
            delta = .1*torch.randn_like(w)
            actual = (u @ (residual-delta@candidate.flatten(1))).square().sum()-explicit
            expected = (u @ delta @ candidate.flatten(1)).square().sum()
            gains.append(float(actual/explicit))
            identity_errors.append(float(abs(actual-expected)/explicit))
        rows.append(dict(duplicate=duplicate, output_rank=rank, gram_rank=int(torch.linalg.matrix_rank(gram)),
                         normal_residual=normal, implicit_explicit_error=error,
                         minimum_relative_loss_increase=min(gains), quadratic_identity_error=max(identity_errors)))
predictions = dict(pred_a_normal=all(x['normal_residual']<=1e-10 for x in rows),
                   pred_b_explicit=all(x['implicit_explicit_error']<=1e-10 for x in rows),
                   pred_c_perturbations=all(x['minimum_relative_loss_increase']>=-1e-10 and x['quadratic_identity_error']<=1e-10 for x in rows))
result = dict(predictions=predictions, arms=rows, scope='Synthetic FP64 conditional output solve; no native fit or data')
Path(__file__).with_name('CONDITIONAL_WRITER_METRIC_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
assert all(predictions.values())
