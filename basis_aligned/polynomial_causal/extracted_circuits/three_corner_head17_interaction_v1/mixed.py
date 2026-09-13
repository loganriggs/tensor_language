"""Exact finite mixed response of the joint score1*score2*value contraction."""
import itertools
import torch

def contract(ports):
    a,b,v=ports
    return torch.einsum('...s,...s,...sd->...d',a,b,v)

def decompose(native,child,remainder,additive):
    """All ports are actual corner values; no assumption that normalized ports add."""
    c=tuple(x-y for x,y in zip(child,native))
    r=tuple(x-y for x,y in zip(remainder,native))
    bar=tuple(n+x+y for n,x,y in zip(native,c,r))
    defect=tuple(x-y for x,y in zip(additive,bar))
    bank=[(native[j],c[j],r[j]) for j in range(3)]
    cross={}
    for key in itertools.product(range(3),repeat=3):
        if 1 in key and 2 in key:
            cross[''.join('0cr'[k] for k in key)]=contract(tuple(bank[j][k] for j,k in enumerate(key)))
    inherited={}
    for mask in range(1,8):
        inherited[str(mask)]=contract(tuple(defect[j] if mask&(1<<j) else bar[j] for j in range(3)))
    local=sum(cross.values());propagated=sum(inherited.values())
    return dict(cross_terms=cross,defect_terms=inherited,cross=local,defect=propagated,total=local+propagated)
