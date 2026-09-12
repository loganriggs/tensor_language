"""Pairing-resolved exact contractions for an analytic diagonal control variate."""
import itertools,math
import torch
from mixed_repeated_contraction_v1 import matchings

def pairing_values(background,x,left,right,down,forms,grade):
    forms=(forms+forms.transpose(-1,-2))/2;nb=4-grade
    pairs=list(itertools.combinations(range(2*grade),2));index={pair:i for i,pair in enumerate(pairs)}
    if pairs:
        ids=torch.tensor(pairs,device=x.device);l=x@left.T;r=x@right.T
        values=((l[:,ids[:,0]]*r[:,ids[:,1]]+r[:,ids[:,0]]*l[:,ids[:,1]])/2)@down.T
        bank=torch.cat([background,values],1)
    else:bank=background
    scores=torch.einsum('npi,kij->nkpj',bank,forms)@bank.transpose(-1,-2)[:,None]
    ids=torch.tensor([list(range(nb))+[nb+index[p] for p in m] for m in matchings(tuple(range(2*grade)))],device=x.device)
    splits=list(itertools.combinations(range(4),2));i=torch.tensor(splits,device=x.device)
    j=torch.tensor([[v for v in range(4) if v not in pair] for pair in splits],device=x.device)
    parent=scores[:,0,ids[:,i[:,0]],ids[:,i[:,1]]]
    partner=scores[:,1:,ids[:,j[:,0]],ids[:,j[:,1]]]
    return math.comb(4,grade)*(parent[:,None]*partner).mean(-1)

def sampled_remainder(error,writer_gram):
    mean=error.mean(-1)
    full=torch.einsum('ni,ij,nj->n',mean,writer_gram,mean)
    diagonal=torch.einsum('nip,ij,njp->n',error,writer_gram,error)/error.shape[-1]**2
    return full-diagonal
