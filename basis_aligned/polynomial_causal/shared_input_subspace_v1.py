"""Mixed-plus-inside quadratic projection onto a shared input subspace."""
import json
from pathlib import Path
import torch
from congruence_block_operator_v1 import factor_forms, sandwich


def value_gradient(e, left, right, gram, second):
    le, re = left @ e, right @ e
    cores = (le[:, :, None]*re[:, None, :] + re[:, :, None]*le[:, None, :])/2
    weighted = (gram @ cores.flatten(1)).reshape_as(cores)
    inside = (cores*weighted).sum()
    incidence = (e*(second@e)).sum()
    grad = 4*second@e - 2*(
        left.T @ torch.einsum('mab,mb->ma', weighted, re)
        + right.T @ torch.einsum('mab,mb->ma', weighted, le))
    return 2*incidence-inside, grad, inside


def control():
    torch.set_num_threads(2); torch.set_default_dtype(torch.float64); torch.manual_seed(1331)
    forms = torch.randn(6, 9, 9); forms = (forms+forms.transpose(1, 2))/2
    l, r, w = factor_forms(forms); k = w.T@w
    second = sandwich(l, r, k, torch.eye(9)); total = forms.square().sum()
    e = torch.linalg.qr(torch.randn(9, 3)).Q; p = e@e.T; c = torch.eye(9)-p
    extracted = forms-c@forms@c
    value, grad, inside = value_gradient(e, l, r, k, second)
    variable = e.clone().requires_grad_(); score, _, _ = value_gradient(variable, l, r, k, second)
    automatic, = torch.autograd.grad(score, variable)
    x = torch.randn(21, 9)
    def evaluate(q, xx): return torch.einsum('ni,vij,nj->nv', xx, q, xx)
    removal = evaluate(forms, x)-evaluate(forms, x@c)
    errors = dict(
        coefficient_projection=abs(float(value-extracted.square().sum()))/float(total),
        gradient=float((grad-automatic).norm()/automatic.norm()),
        native_removal=float((removal-evaluate(extracted, x)).norm()/removal.norm()),
        remainder_orthogonality=abs(float((extracted*(c@forms@c)).sum()/total)))
    values, vectors = torch.linalg.eigh(second)
    spectral = vectors[:, -3:]
    found, _, _ = value_gradient(spectral, l, r, k, second)
    upper = torch.minimum(total, 2*values[-3:].sum())
    orth = torch.linalg.qr(torch.randn(3, 3)).Q
    rotated, _, _ = value_gradient(e@orth, l, r, k, second)
    errors['within_subspace_gauge'] = abs(float(rotated-value))/float(total)
    # Construct a PURE mixed family: inside-only Tucker captures zero, but this
    # broader shared-reader family recovers all of it.
    mixed = p@forms@c+c@forms@p
    ml, mr, mw = factor_forms(mixed); ms = sandwich(ml, mr, mw.T@mw, torch.eye(9))
    mv, _, mi = value_gradient(e, ml, mr, mw.T@mw, ms)
    mixed_capture = float(mv/mixed.square().sum())
    inside_capture = float(mi/mixed.square().sum())
    full, _, _ = value_gradient(torch.eye(9), l, r, k, second)
    errors['full_space'] = abs(float(full/total)-1)
    passed = max(errors.values())<1e-10 and float(found)<=float(upper)+1e-10 and float(found)>=float(upper)/2-1e-10 and abs(mixed_capture-1)<1e-10 and abs(inside_capture)<1e-10
    result = dict(instrument_passed=passed, errors=errors,
                  spectral_capture=float(found/total), upper_bound=float(upper/total),
                  planted_mixed_capture=mixed_capture, planted_inside_only_capture=inside_capture,
                  scope='CPU controls only. Arbitrary interactions touching a shared input space; no low output-rank restriction, no text or semantic identification.')
    with Path(__file__).with_name('SHARED_INPUT_SUBSPACE_V1_CONTROL.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2)); assert passed


if __name__ == '__main__': control()
