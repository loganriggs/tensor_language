"""Exhaustive symmetric-coordinate coefficient norm; no sampling of entries."""
import itertools
import math
from collections import Counter
import torch
from implicit_quartic import entries


def symmetric_indices(d, device='cpu'):
    tuples = list(itertools.combinations_with_replacement(range(d), 4))
    weights = [24 // math.prod(math.factorial(n) for n in Counter(t).values()) for t in tuples]
    return torch.tensor(tuples, device=device), torch.tensor(weights, device=device)


def projected_energy(teacher, basis, batch=256):
    """basis has orthonormal columns; return ||H restricted to basis^4||_F^2."""
    C, L, R, D, A, B = teacher
    restricted = (C, L, R, D, A @ basis, B @ basis)
    indices, weights = symmetric_indices(basis.shape[1], basis.device)
    total = torch.zeros((), device=basis.device, dtype=torch.float64)
    for start in range(0, len(indices), batch):
        values = entries(*restricted, indices[start:start + batch]).double()
        total += (values.square().sum(1) * weights[start:start + batch]).sum()
    return total


def check():
    torch.set_num_threads(1)
    torch.manual_seed(1820)
    teacher = [torch.randn(*shape, dtype=torch.float64) for shape in [(2, 4), (4, 3), (4, 3), (3, 5), (5, 6), (5, 6)]]
    basis = torch.linalg.qr(torch.randn(6, 3, dtype=torch.float64))[0]
    projected = projected_energy(teacher, basis)
    full = torch.cartesian_prod(*[torch.arange(3)] * 4)
    C,L,R,D,A,B = teacher
    direct = entries(C,L,R,D,A@basis,B@basis,full).square().sum()
    relative = float(abs(projected-direct)/direct)
    assert relative < 1e-12
    _, weights = symmetric_indices(32)
    assert len(weights) == 52360 and int(weights.sum()) == 32**4
    return dict(relative_error=relative, dimension32_unique_entries=len(weights), ordered_entries=int(weights.sum()))

if __name__ == '__main__':
    print(check())
