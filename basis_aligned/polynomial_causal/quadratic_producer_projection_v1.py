"""Exact Frobenius projection onto native bilinear product matrices."""
import torch

def atom_gram(left, right):
    cross = left @ right.T
    return ((left @ left.T) * (right @ right.T) + cross * cross.T) / 2

def correlations(left, right, matrices):
    return torch.stack([((left @ q) * right).sum(-1) for q in matrices], 1)

def project(gram, cross):
    scale = gram.diagonal().sqrt()
    normalized = gram / scale[:, None] / scale[None, :]
    rhs = cross / scale[:, None]
    factor = torch.linalg.cholesky(normalized)
    solution = torch.cholesky_solve(rhs, factor) / scale[:, None]
    residual = (gram @ solution - cross).norm() / cross.norm()
    return solution, float(residual)

def matrix(left, right, coefficients):
    raw = left.T @ (coefficients[:, None] * right)
    return (raw + raw.T) / 2
