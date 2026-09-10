"""Exact fixed-basis sparse core; spectral input initialization is not global optimization."""
import torch

def input_marginal(l,r,output_gram):
    g=output_gram
    result=l.T@((g*(r@r.T))@l)
    result=result+r.T@((g*(l@l.T))@r)
    cross=l.T@((g*(r@l.T))@r)
    result=(result+cross+cross.T)/4
    return (result+result.T)/2

def orthogonal_core(l,r,d,basis,chunk=256):
    a=l@basis.T;b=r@basis.T
    i,j=torch.triu_indices(len(basis),len(basis),device=basis.device)
    columns=[]
    for start in range(0,len(i),chunk):
        ii=i[start:start+chunk];jj=j[start:start+chunk]
        norm=torch.where(ii==jj,torch.full_like(ii,2,dtype=l.dtype),torch.full_like(ii,2**.5,dtype=l.dtype))
        feature=(a[:,ii]*b[:,jj]+a[:,jj]*b[:,ii])/norm
        columns.append(d@feature)
    return torch.cat(columns,dim=1),torch.stack([i,j])
