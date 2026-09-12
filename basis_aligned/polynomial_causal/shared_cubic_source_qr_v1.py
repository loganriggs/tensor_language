"""Equivalent direct coefficient QR, avoiding squared source condition numbers.

Detached local linear spans contain the current coefficient tensors exactly.
Their omitted normal derivatives are orthogonal to the current tensors, so
first derivatives of captured squared norm are preserved. Higher derivatives
are not promised by this detached-span implementation.
"""
import itertools,math
import torch
from shared_cubic_source_projection_v1 import cross_factors

def cubic_coefficients(atoms):
    basis=torch.linalg.qr(atoms.detach().flatten(0,1).T,mode='reduced')[0]
    x=atoms@basis;dimension=basis.shape[1]
    ids=list(itertools.combinations_with_replacement(range(dimension),3))
    index=torch.tensor(ids,device=atoms.device)
    mult=torch.tensor([6/math.prod(math.factorial(t.count(i)) for i in set(t)) for t in ids],device=atoms.device,dtype=atoms.dtype)
    coefficients=sum(x[:,0,index[:,p[0]]]*x[:,1,index[:,p[1]]]*x[:,2,index[:,p[2]]] for p in itertools.permutations(range(3)))/6
    return coefficients*mult.sqrt()

def capture(atoms,q1,k1,q2,k2,values,output):
    coefficients=cubic_coefficients(atoms)
    _,r=torch.linalg.qr(coefficients.T,mode='reduced')
    a,b,c=cross_factors(atoms,q1,k1,q2,k2,values,output);total=atoms.new_zeros(())
    for h in range(len(a)):
        readers=torch.cat([a[h,:,[0,2,4]].flatten(0,1),b[h,:,[2,0,1]].flatten(0,1)])
        qb=torch.linalg.qr(readers.detach().T,mode='reduced')[0]
        ob=torch.linalg.qr(c[h,:,[3,1,0]].detach().flatten(0,1).T,mode='reduced')[0]
        aa=a[h]@qb;bb=b[h]@qb;cc=c[h]@ob
        raw=torch.einsum('rta,rtb,rto->roab',aa,bb,cc);cross=(raw+raw.transpose(-1,-2))/2
        projected=torch.linalg.solve_triangular(r.T,cross.flatten(1),upper=False)
        total=total+projected.square().sum()
    return total
