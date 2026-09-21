"""Selected-pair quartic dictionary with shared low-rank quadratic features.

Unlike the complete upper-triangular dictionary, the Gram grows with selected
pairs, not all possible pairs of quadratic features. Pair support is explicit.
"""
import torch
from quartic_cp import directional


def validate_pairs(pairs, width):
    assert pairs.ndim == 2 and pairs.shape[0] == 2
    assert pairs.dtype == torch.long
    assert bool(((pairs >= 0) & (pairs < width)).all())
    assert bool((pairs[0] <= pairs[1]).all())
    assert torch.unique(pairs.T, dim=0).shape[0] == pairs.shape[1]


def gram(U, V, pairs):
    E = torch.cat([U, V], 1)
    F = torch.cat([V, U], 1) / 2
    edge = torch.einsum('iad,jbd->ijab', F, E)
    quad = torch.einsum('ijab,jiba->ij', edge, edge)
    i, j = pairs
    first = quad[i[:,None],i[None,:]]*quad[j[:,None],j[None,:]]
    first = first + quad[i[:,None],j[None,:]]*quad[j[:,None],i[None,:]]
    a=edge[i[:,None],i[None,:]]; b=edge[i[None,:],j[:,None]]
    c=edge[j[:,None],j[None,:]]; d=edge[j[None,:],i[:,None]]
    trace=((a@b@c)*d.transpose(-1,-2)).sum((-1,-2))
    return (first+4*trace)/6


def native_cross(teacher, U, V, pairs):
    i,j=pairs; k=U.shape[1]; count=len(i)
    a=U[i,:,None,:].expand(count,k,k,-1).reshape(count*k*k,-1)
    b=V[i,:,None,:].expand(count,k,k,-1).reshape(count*k*k,-1)
    c=U[j,None,:,:].expand(count,k,k,-1).reshape(count*k*k,-1)
    d=V[j,None,:,:].expand(count,k,k,-1).reshape(count*k*k,-1)
    return directional(*teacher,[a,b,c,d]).reshape(count,k*k,-1).sum(1).T


def entries(U, V, pairs, indices):
    a,b=pairs
    def pair(i,j):return (.5*(U[:,:,i]*V[:,:,j]+U[:,:,j]*V[:,:,i])).sum(1).T
    def outer(s,t):return .5*(s[:,a]*t[:,b]+t[:,a]*s[:,b])
    i,j,k,l=indices.T
    return (outer(pair(i,j),pair(k,l))+outer(pair(i,k),pair(j,l))+outer(pair(i,l),pair(j,k)))/3


def features(x, U, V, pairs):
    q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),U.shape[0],U.shape[1]).sum(-1)
    return q[:,pairs[0]]*q[:,pairs[1]]


def support(width=144, count=512, seed=1000):
    assert width <= count <= width*(width+1)//2
    diagonal=torch.arange(width).repeat(2,1)
    other=torch.triu_indices(width,width,offset=1)
    take=torch.randperm(other.shape[1],generator=torch.Generator().manual_seed(seed))[:count-width]
    result=torch.cat([diagonal,other[:,take]],1)
    validate_pairs(result,width)
    return result
