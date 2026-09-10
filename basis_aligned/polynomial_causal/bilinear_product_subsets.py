"""Exact source-product edits and signed local-write scores; no suffix approximation.

Inputs to the MLP are already normalized. These functions do not remove the
native prefix, normalizer, bias, or downstream computations from the price.
"""
import json
import torch


def products(x, left, right):
    return (x @ left.T) * (x @ right.T)


def subset_write(phi, down, indices):
    return phi[..., indices] @ down[:, indices].T


def interchange(base_phi, donor_phi, down, indices):
    return subset_write(donor_phi - base_phi, down, indices)


def signed_scores(delta_phi, down):
    """Mean alignment of each product write with the complete paired MLP write.

    Rows must be valid fit tokens only. Sum(score) = mean(||delta_m||**2).
    Negative scores are cancellation, not negative causal importance. This is
    a local proposal score, not a prediction of downstream behavioral effects.
    """
    delta_m = delta_phi @ down.T
    return (delta_phi * (delta_m @ down)).mean(0)


def tensor_gram(left, right, down):
    """Gram of tied-input symmetric tensor terms, without building D cubed.

    For term j, T_j = down_j outer sym(left_j outer right_j).
    This is coefficient geometry, not a distribution or behavioral metric.
    """
    ll, rr, lr = left @ left.T, right @ right.T, left @ right.T
    return (down.T @ down) * (ll*rr + lr*lr.T) / 2


def controls():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(910318)
    rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    left, right, down = rand(7, 4), rand(7, 4), rand(4, 7)
    x, y, bias = rand(9, 4), rand(9, 4), rand(4)
    p, q = products(x, left, right), products(y, left, right)
    all_indices, first, second = list(range(7)), [0, 2, 4], [1, 3, 5, 6]
    native = p @ down.T + bias
    full = q @ down.T + bias
    errors = {}
    error = lambda a, b: float((a - b).abs().max())
    errors['complete_interchange'] = error(native + interchange(p, q, down, all_indices), full)
    errors['disjoint_union'] = error(interchange(p, q, down, first) + interchange(p, q, down, second), full-native)
    removed_down = down.clone(); removed_down[:, first] = 0
    errors['static_weight_removal'] = error(native-subset_write(p, down, first), p @ removed_down.T + bias)
    score = signed_scores(q-p, down)
    errors['score_conservation'] = error(score.sum(), ((q-p) @ down.T).square().sum(-1).mean())
    # Independent nonzero factor rescalings, compensated in Down.
    a = torch.tensor([2, -.5, 4, -2, .25, 8, -4], dtype=torch.float64)
    b = torch.tensor([.5, 4, -2, .25, -8, -1, 2], dtype=torch.float64)
    changed_down = down / (a*b)
    pp, qq = products(x, a[:, None]*left, b[:, None]*right), products(y, a[:, None]*left, b[:, None]*right)
    errors['subset_rescaling_invariance'] = error(interchange(pp, qq, changed_down, first), interchange(p, q, down, first))
    errors['score_rescaling_invariance'] = error(signed_scores(qq-pp, changed_down), score)
    perm = torch.tensor([6, 2, 0, 5, 1, 4, 3])
    errors['score_permutation_equivariance'] = error(signed_scores((q-p)[:, perm], down[:, perm]), score[perm])
    symmetric = (left[:, :, None]*right[:, None, :] + right[:, :, None]*left[:, None, :])/2
    explicit = torch.einsum('oj,jab->joab', down, symmetric).flatten(1)
    gram = tensor_gram(left, right, down)
    errors['tensor_gram_without_dense_tensor'] = error(gram, explicit @ explicit.T)
    errors['tensor_gram_rescaling_invariance'] = error(gram, tensor_gram(a[:, None]*left, b[:, None]*right, changed_down))
    # Two huge opposite writes cancel. An energy-only ranking selects both
    # above the only surviving write. This is a planted local example.
    cancel_delta = torch.ones(1, 3, dtype=torch.float64)
    cancel_down = torch.tensor([[100., -100., 1.]], dtype=torch.float64)
    diagonal_energy = cancel_delta.square().mean(0)*cancel_down.square().sum(0)
    cancellation_score = signed_scores(cancel_delta, cancel_down)
    assert diagonal_energy.tolist() == [10000., 10000., 1.]
    assert cancellation_score.tolist() == [100., -100., 1.]
    # Signed scores detect cancellation but still do not identify a minimal
    # support: the positive 100 score also exceeds the sufficient unit's 1.
    assert float(subset_write(cancel_delta, cancel_down, [2])) == 1.
    # Additive source edits need joint nonlinear suffix evaluation.
    suffix = lambda z: z.square()
    base, u, v = [torch.tensor(t, dtype=torch.float64) for t in (1., 2., 3.)]
    joint = suffix(base+u+v)-suffix(base)
    naive = suffix(base+u)-suffix(base)+suffix(base+v)-suffix(base)
    errors['nonlinear_joint_identity'] = error(joint-naive, 2*u*v)
    assert all(v < 1e-10 for v in errors.values())
    return {'passed': True, 'errors': errors,
            'cancellation': {'diagonal_energy': diagonal_energy.tolist(), 'signed_scores': cancellation_score.tolist(), 'complete_write': 1., 'unit_2_alone_write': 1.},
            'nonlinear_suffix_joint_minus_sum': float(joint-naive),
            'native_model_loaded': False, 'gpu_accessed': False,
            'scope': 'Algebra controls only; neither score establishes causal sharing or a semantic feature.'}


if __name__ == '__main__':
    print(json.dumps(controls(), indent=2))
