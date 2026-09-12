"""Exact coefficient Gram for squares of full symmetric quadratic forms."""
import torch

def full_gram(q):
    k=q.new_empty((len(q),len(q)))
    for i in range(len(q)):
        for j in range(i+1):
            product=q[i]@q[j]
            value=(product.trace().square()+2*(product*product.T).sum())/3
            k[i,j]=k[j,i]=value
    return k

def low_full_cross(b,n,q):
    columns=[]
    for matrix in q:
        inner=b.transpose(-1,-2)@matrix@b
        trace=(n*inner.diagonal(dim1=-2,dim2=-1)).sum(-1)
        cycle=(n[:,:,None]*n[:,None,:]*inner.square()).sum((-1,-2))
        columns.append((trace.square()+2*cycle)/3)
    return torch.stack(columns,1)
