"""Polynomial-quotient geometry for shared quadratic-feature programs.

Represent quartics as sums of Q(x)R(x), keeping unsymmetrized coefficient
representatives. Compare their fully symmetric parts without a d**4 tensor.
Dense quadratic matrices here are a reduced-space reference, not a claim
that materializing all native-width intermediate quadratics is affordable.
"""
import torch


def symmetric(matrix):
    return (matrix + matrix.transpose(-1, -2)) / 2


def product_inner(Q, R, S, T):
    """<Sym(Q⊗R), Sym(S⊗T)> for four symmetric quadratic forms.

    Broadcasting over leading dimensions is supported. The six distinct
    input pairings collapse to two Frobenius products and a matrix trace.
    Gradients propagate through all factors; intermediate bases need not be
    orthogonal. Antisymmetric parts are removed because x^T Q x ignores them.
    """
    Q, R, S, T = map(symmetric, (Q, R, S, T))
    frob = lambda a, b: (a * b).sum(dim=(-2, -1))
    mixed = (Q @ S @ R @ T).diagonal(dim1=-2, dim2=-1).sum(-1)
    return (frob(Q, S)*frob(R, T) + frob(Q, T)*frob(R, S) + 4*mixed) / 6


def program_inner(bank_a, edges_a, writers_a, bank_b, edges_b, writers_b):
    """Full-output polynomial inner product, streamed one edge pair at a time.

    Quadratic bank shape [features,d,d], edges [edges,2], writers [outputs,edges].
    Cost is quadratic in edges and cubic in reduced input width. This exact
    reference is meant to validate scalable approximations, not conceal cost.
    """
    result = writers_a.new_zeros(())
    for i, (a, b) in enumerate(edges_a.tolist()):
        for j, (c, d) in enumerate(edges_b.tolist()):
            result = result + (writers_a[:, i]*writers_b[:, j]).sum() * product_inner(
                bank_a[a], bank_a[b], bank_b[c], bank_b[d])
    return result


def execute(bank, edges, writers, x, zero_features=()):
    """Execute each quadratic once and reuse it at every incident root edge.

    This is the polynomial numerator only. Normalization and source generation
    belong in the enclosing model graph and cannot be silently expanded away.
    zero_features enables local node-removal controls, not a causal claim.
    """
    values = torch.einsum('...i,aij,...j->...a', x, bank, x)
    if zero_features:
        values = values.clone()
        values[..., list(zero_features)] = 0
    edges = edges.to(device=x.device)
    products = values[..., edges[:, 0]] * values[..., edges[:, 1]]
    return products @ writers.T


def sparse_program_price(bank, edges, writers):
    """Support price for a sparse executor, counting reused pairs once.

    Count only live features. Coalesce unordered pairs inside each quadratic;
    exact zeros alone are free. Root duplicate pairs can share multiplication,
    but writer coefficients are charged as supplied. Inputs are coordinates,
    so this price excludes any external leaf projection or downstream adapter.
    """
    active = writers.ne(0).any(dim=0)
    edges = edges[active.to(edges.device)]
    used = sorted(set(edges.flatten().tolist()))
    d = bank.shape[-1]
    p, q = torch.triu_indices(d, d, device=bank.device)
    coefficients = symmetric(bank)[used][:, p, q]
    quadratic_support = coefficients.ne(0)
    root_pairs = {tuple(sorted(e)) for e in edges.tolist()}
    return dict(features=len(used), quadratic_coefficients=int(quadratic_support.sum()),
                shared_input_products=int(quadratic_support.any(dim=0).sum()),
                root_products=len(root_pairs), root_coefficients=int(writers[:, active].ne(0).sum()),
                support_records=int(quadratic_support.sum()) + int(writers[:, active].ne(0).sum()),
                scope='local sparse polynomial program; external projections and norms excluded')
