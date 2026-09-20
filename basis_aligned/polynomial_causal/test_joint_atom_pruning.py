import torch
from joint_atom_pruning import atom_gram, greedy_removal_order, removal_energy, subset_error_bound


def test_gram_removal_and_each_greedy_step_against_dense():
    torch.manual_seed(617)
    C, A, B = [torch.randn(*shape, dtype=torch.float64) for shape in [(4, 7), (7, 3), (7, 3)]]
    atoms = torch.einsum('vk,ki,kj->kvij', C, A, B)
    atoms = (atoms+atoms.transpose(-1, -2))/2
    flat = atoms.flatten(1)
    gram = atom_gram(C, A, B)
    torch.testing.assert_close(gram, flat@flat.T)
    order = greedy_removal_order(gram)
    residual = torch.zeros_like(atoms[0])
    remaining = set(range(7))
    for step, i in enumerate(order.tolist()):
        best = min(remaining, key=lambda j: float((residual+atoms[j]).square().sum()))
        assert i == best
        residual += atoms[i]
        remaining.remove(i)
        torch.testing.assert_close(removal_energy(gram, order[step+1:]), residual.square().sum())


def test_cancellation_and_factor_rescaling():
    C = torch.tensor([[1., -1., .01]], dtype=torch.float64)
    A = B = torch.ones(3, 1, dtype=torch.float64)
    gram = atom_gram(C, A, B)
    # Removing the first cancelling pair together costs zero despite large norms.
    torch.testing.assert_close(removal_energy(gram, torch.tensor([2])), torch.zeros((), dtype=C.dtype))
    scale = torch.tensor([.01, -20., 3.], dtype=C.dtype)
    torch.testing.assert_close(atom_gram(C/scale, A*scale[:, None], B), gram)


def test_bound_below_every_refitted_subset():
    import itertools
    torch.manual_seed(618)
    atoms = torch.randn(5, 11, dtype=torch.float64)
    gram = atoms@atoms.T
    eig, bounds = subset_error_bound(gram, range(6))
    assert eig[0] > 0
    for size in range(6):
        for support in itertools.combinations(range(5), size):
            if size:
                selected = atoms[list(support)]
                coefficients = torch.linalg.lstsq(selected.T, atoms.sum(0)).solution
                residual = atoms.sum(0)-coefficients@selected
            else:
                residual = atoms.sum(0)
            error = float(residual.norm()/atoms.sum(0).norm())
            assert error+1e-12 >= bounds[size]
