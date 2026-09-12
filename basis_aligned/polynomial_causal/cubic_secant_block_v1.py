"""Stable polynomial coordinates for the span of two nearby cubic products."""
import itertools
import torch
from shared_cubic_source_projection_v1 import atom_gram,cross_factors,query_output_gram

def align_pair(first,second):
    choices=[]
    for perm in itertools.permutations(range(3)):
        candidate=second[list(perm)]
        sign=(first*candidate).sum(-1).sign();sign=torch.where(sign==0,torch.ones_like(sign),sign)
        candidate=candidate*sign[:,None]
        choices.append((float((first-candidate).square().sum()),candidate))
    return min(choices,key=lambda x:x[0])[1]

def encode(atoms,pair=(5,6),tangent=False):
    i,j=pair;other=[r for r in range(len(atoms)) if r not in pair]
    first=atoms[i];second=align_pair(first,atoms[j]);base=(first+second)/2
    delta=(first-second)/2;t=delta.norm();direction=delta/t
    products=[]
    for mask in range(8):products.append(torch.stack([direction[k] if mask&(1<<k) else base[k] for k in range(3)]))
    components=torch.cat([atoms[other],torch.stack(products)])
    mix=atoms.new_zeros(len(atoms),len(components));mix[:len(other),:len(other)]=torch.eye(len(other),device=atoms.device,dtype=atoms.dtype)
    for mask in range(8):
        degree=mask.bit_count();coefficient=1 if degree<2 else (0 if tangent else t*t)
        mix[-2+degree%2,len(other)+mask]=coefficient
    return components,mix,dict(separation=float(t),base=base,direction=direction)

def gram_kernel(components,mix,weights):
    g=mix@atom_gram(components)@mix.T
    k=mix[None]@query_output_gram(cross_factors(components,*weights))@mix.T[None]
    return g,k

def capture(components,mix,weights):
    g,k=gram_kernel(components,mix,weights)
    return torch.linalg.solve(g,k.sum(0)).trace()
