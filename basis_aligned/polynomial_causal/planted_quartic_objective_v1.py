"""Exact coefficient objective for synthetic quadratic-square programs."""
import torch
from coupled_quartic_writer_v1 import gram
from quartic_manifold_lbfgs_v1 import tangent

def objective(true_readers, true_weights, true_mixing):
    energy = float((true_mixing * (gram(true_readers, true_weights) @ true_mixing)).sum())
    def evaluate(b, n, divisor, gradient=False):
        if gradient:
            b = b.detach().requires_grad_()
            n = n.detach().requires_grad_()
        with torch.set_grad_enabled(gradient):
            count = len(b)
            joint = gram(torch.cat([b, true_readers]), torch.cat([n, true_weights]))
            k = joint[:count, :count]
            c = joint[:count, count:] @ true_mixing
            with torch.no_grad():
                mixing = torch.linalg.solve(k, c)
            loss = (energy + (mixing * (k @ mixing)).sum() - 2 * (mixing * c).sum()) / divisor
            if gradient:
                gb, gn = torch.autograd.grad(loss, (b, n))
                gb, gn = tangent(b, n, gb, gn)
                return float(loss.detach()), mixing, gb.detach(), gn.detach()
            return float(loss), mixing
    return evaluate, energy
