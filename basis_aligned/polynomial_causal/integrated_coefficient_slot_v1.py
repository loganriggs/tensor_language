"""Exact covariance integration of one slot in a multilinear coefficient map.

evaluate(z) must be row-separable, linear and homogeneous in z, with shape
[batch, outputs]. The returned norm assumes E[z z.T]=I. Other slots remain
fixed. create_graph=True supports gradients through all coefficient parameters.
"""
import torch

def integrate(evaluate,prototype,writer_gram):
    z=torch.zeros_like(prototype,requires_grad=True)
    outputs=evaluate(z)
    jac=torch.stack([torch.autograd.grad(outputs[:,j].sum(),z,create_graph=True,retain_graph=True)[0]
                     for j in range(outputs.shape[1])],dim=1)
    return torch.einsum('nid,ij,njd->n',jac,writer_gram,jac)
