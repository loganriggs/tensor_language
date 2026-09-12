"""Exact quartic coefficient norm under a common producer-input projection.

Formal source slots are [background, whitened producer]; this is not a native
input-distribution metric. No large source tensor or vocabulary tensor needed.
"""
import torch

def branch_gram(forms, kernel):
    # C_i = A_i K. Trace(C_i C_j) is the inner product of pulled quadratics.
    c=forms@kernel
    tr=torch.einsum('iab,jba->ij',c,c)
    cycle=[]
    for i in range(1,len(forms)):
        row=[]
        for j in range(1,len(forms)):
            row.append(torch.trace((c[0]@c[0])@(c[i]@c[j])))
        cycle.append(torch.stack(row))
    cycle=torch.stack(cycle)
    return (tr[0,0]*tr[1:,1:]+tr[0,1:,None]*tr[0,None,1:]+4*cycle)/6

def norm(forms, kernel, writer_gram):
    return (branch_gram(forms,kernel)*writer_gram).sum()

def retained(forms, root, directions, writer_gram):
    z=root@directions
    k=torch.eye(len(root),device=root.device,dtype=root.dtype)+z@z.T
    return norm(forms,k,writer_gram)

def tangent(directions, gradient):
    return gradient-directions@(directions.T@gradient)

def retract(directions, step):
    return torch.linalg.qr(directions+step,mode='reduced').Q
