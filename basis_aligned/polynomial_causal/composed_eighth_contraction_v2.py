"""Symmetric eight-linear contraction of a bilinear producer and two quadratic readers.

Bias-free homogeneous numerator only. RMS, background and mixed paths remain
external. Reuses all 28 producer pairs across 70 four-slot subsets.
"""
import itertools
import torch

def contract(x,left,right,down,forms,scale=1.):
    assert x.ndim==3 and x.shape[1]==8 and forms.shape[0]>=2
    forms=(forms+forms.transpose(-1,-2))/2
    pair_list=list(itertools.combinations(range(8),2));index={pair:i for i,pair in enumerate(pair_list)}
    pairs=torch.tensor(pair_list,device=x.device)
    l=x@left.T;r=x@right.T
    h=(l[:,pairs[:,0]]*r[:,pairs[:,1]]+r[:,pairs[:,0]]*l[:,pairs[:,1]])/2
    producer=(h@down.T)*scale
    transformed=torch.einsum('npi,kij->nkpj',producer,forms)
    scores=transformed@producer.transpose(-1,-2)[:,None,:,:]
    subsets=list(itertools.combinations(range(8),4));lookup={s:i for i,s in enumerate(subsets)}
    first=[];second=[];complements=[]
    for a,b,c,d in subsets:
        first.append([index[(a,b)],index[(a,c)],index[(a,d)]])
        second.append([index[(c,d)],index[(b,d)],index[(b,c)]])
        complements.append(lookup[tuple(i for i in range(8) if i not in (a,b,c,d))])
    first=torch.tensor(first,device=x.device);second=torch.tensor(second,device=x.device)
    quartic=scores[:,:,first,second].mean(-1)
    complement=torch.tensor(complements,device=x.device)
    return (quartic[:,0,None,:]*quartic[:,1:,complement]).mean(-1)
