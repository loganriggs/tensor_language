"""Reuse exact symmetric-product ALS with each reader in a fixed sparse span."""
import torch
from symmetric_product_als_v1 import normal_operator, pcg


def decode(values, atoms):
    return torch.einsum('nk,nkd->nd', values, atoms)


def project(readers, atoms):
    return torch.einsum('nd,nkd->nk', readers, atoms)


@torch.no_grad()
def solve(atoms, partner, output_gram, rhs, initial):
    def operator(values):
        return project(normal_operator(decode(values, atoms), partner, output_gram), atoms)
    gram = atoms @ atoms.transpose(-1, -2)
    pb = project(partner, atoms)
    blocks = .5 * output_gram.diag()[:, None, None] * (
        partner.square().sum(1)[:, None, None] * gram + pb[:, :, None] * pb[:, None, :])
    chol = torch.linalg.cholesky((blocks + blocks.transpose(-1, -2)) / 2)
    precondition = lambda v: torch.cholesky_solve(v[..., None], chol).squeeze(-1)
    return pcg(operator, project(rhs, atoms), precondition, initial=initial,
               tolerance=1e-9, max_iterations=300)
