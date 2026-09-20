"""CPU red-team control: matrix compression can fail on a simple polynomial.

This is a planted counterexample, not evidence of native model compressibility.
"""
import json
from pathlib import Path
import torch


def quadratic(left, right, writer):
    raw = left.T @ (writer[:, None] * right)
    return (raw + raw.T) / 2


def shared_rank_one(left, right, writer):
    _, _, vh = torch.linalg.svd(torch.cat([left, right]), full_matrices=False)
    p = vh[:1].T
    return p @ quadratic(left @ p, right @ p, writer) @ p.T


def main():
    torch.set_num_threads(2)
    left = torch.diag(torch.tensor([100., 1.], dtype=torch.float64))
    right = torch.diag(torch.tensor([.0001, 1.], dtype=torch.float64))
    writer = torch.ones(2, dtype=torch.float64)
    gauge = torch.tensor([.001, 1.], dtype=torch.float64)
    balanced_left, balanced_right = gauge[:, None]*left, right/gauge[:, None]
    target = quadratic(left, right, writer)
    balanced_target = quadratic(balanced_left, balanced_right, writer)
    torch.testing.assert_close(target, balanced_target, rtol=1e-14, atol=1e-14)
    estimates = {'raw_matrix_svd': shared_rank_one(left, right, writer),
                 'balanced_matrix_svd': shared_rank_one(balanced_left, balanced_right, writer)}
    e, v = torch.linalg.eigh(target)
    i = e.abs().argmax()
    estimates['joint_quadratic_spectral'] = e[i] * torch.outer(v[:, i], v[:, i])
    errors = {k: float((q-target).norm()/target.norm()) for k,q in estimates.items()}
    assert errors['raw_matrix_svd'] > .99
    assert errors['balanced_matrix_svd'] < .011
    assert errors['joint_quadratic_spectral'] < .011
    # Independent evaluation and sensitivity tripwire, including signed output.
    x = torch.tensor([[1.,0.],[0.,1.],[1.,2.],[-3.,.5]], dtype=torch.float64)
    native = ((x@left.T)*(x@right.T))@writer
    dense = torch.einsum('bi,ij,bj->b', x, target, x)
    torch.testing.assert_close(native, dense, rtol=1e-14, atol=1e-14)
    assert not torch.allclose(native, -dense)
    result = dict(target='0.01*x0^2 + x1^2', norm='symmetric coefficient Frobenius',
                  errors=errors, gauge_invariance_error=float((target-balanced_target).norm()),
                  independent_evaluation_max_error=float((native-dense).abs().max()),
                  sign_mutation_detected=True,
                  conclusion='Raw factor-SVD failure does not rule out a simple joint decomposition.',
                  scope='planted CPU control, no native compression claim')
    path = Path(__file__).with_name('DECOMPOSITION_GAUGE_CONTROL_RESULT.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
