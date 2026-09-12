"""Exact separately symmetric background/input coefficient contractions.

Grade k has 4-k background slots and 2*k producer-input slots. Includes the
binomial polynomial coefficient. RMS and the background's own producer remain
external; the quadratic producer's bias can be absorbed into the background.
"""
import itertools,math
import torch

def matchings(items):
    if not items:return [()]
    first=items[0];out=[]
    for second in items[1:]:
        rest=tuple(i for i in items[1:] if i!=second)
        out.extend((((first,second),)+m) for m in matchings(rest))
    return out

def contract(background,x,left,right,down,forms,grade,scale=1.):
    assert 0<=grade<=4 and background.shape[1]==4-grade and x.shape[1]==2*grade
    forms=(forms+forms.transpose(-1,-2))/2;nbackground=4-grade
    pair_list=list(itertools.combinations(range(2*grade),2));index={pair:i for i,pair in enumerate(pair_list)}
    if pair_list:
        pairs=torch.tensor(pair_list,device=x.device);l=x@left.T;r=x@right.T
        products=(l[:,pairs[:,0]]*r[:,pairs[:,1]]+r[:,pairs[:,0]]*l[:,pairs[:,1]])/2
        producer=(products@down.T)*scale;bank=torch.cat([background,producer],dim=1)
    else:bank=background
    transformed=torch.einsum('npi,kij->nkpj',bank,forms);scores=transformed@bank.transpose(-1,-2)[:,None,:,:]
    slot_ids=[list(range(nbackground))+[nbackground+index[p] for p in matching] for matching in matchings(tuple(range(2*grade)))]
    ids=torch.tensor(slot_ids,device=x.device);splits=list(itertools.combinations(range(4),2))
    i=torch.tensor(splits,device=x.device);j=torch.tensor([[v for v in range(4) if v not in pair] for pair in splits],device=x.device)
    parent=scores[:,0,ids[:,i[:,0]],ids[:,i[:,1]]]
    partner=scores[:,1:,ids[:,j[:,0]],ids[:,j[:,1]]]
    return math.comb(4,grade)*(parent[:,None,:,:]*partner).mean((-1,-2))
