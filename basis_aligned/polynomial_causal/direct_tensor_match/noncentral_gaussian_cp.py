"""Exact affine-CP moments and native quartic cross under a shifted Gaussian."""
import itertools
import torch
from quartic_cp import directional
from native_quartic_gaussian_projection import project


def partial_matchings(indices):
    if not indices:
        yield (), ()
        return
    a=indices[0]
    for singles,pairs in partial_matchings(indices[1:]):
        yield (a,)+singles,pairs
    for j,b in enumerate(indices[1:],1):
        for singles,pairs in partial_matchings(indices[1:j]+indices[j+1:]):
            yield singles,((a,b),)+pairs

PARTITIONS={n:tuple(partial_matchings(tuple(range(n)))) for n in range(9)}


def affine_moment(factors,biases):
    n=len(factors)
    if n==0:return 1.
    result=torch.zeros_like(biases[0])
    dots={(i,j):(factors[i]*factors[j]).sum(1) for i in range(n) for j in range(i+1,n)}
    for singles,pairs in PARTITIONS[n]:
        term=torch.ones_like(result)
        for i in singles:term=term*biases[i]
        for pair in pairs:term=term*dots[pair]
        result=result+term
    return result


def gram(factors,biases,other,other_biases):
    vectors=list(factors)+list(other)
    means=[v[:,None] for v in biases]+[v[None,:] for v in other_biases]
    dots={}
    for i in range(8):
        for j in range(i+1,8):
            if i<4<=j:value=vectors[i]@vectors[j].T
            else:
                value=(vectors[i]*vectors[j]).sum(1)
                value=value[:,None] if j<4 else value[None,:]
            dots[i,j]=value
    result=factors[0].new_zeros((len(factors[0]),len(other[0])))
    for singles,pairs in PARTITIONS[8]:
        term=result.new_ones((1,1))
        for i in singles:term=term*means[i]
        for pair in pairs:term=term*dots[pair]
        result=result+term
    return result


def project_shifted(teacher,location,zero_projection=None):
    """Return E F, E gradient F, E Hessian F /2 in whitened coordinates."""
    mean0,Q0=project(teacher) if zero_projection is None else zero_projection
    C,l,r,D,L,R=teacher
    a,b=L@location,R@location
    h=D@(a*b)
    J=D@(a[:,None]*R+b[:,None]*L)
    lh,rh=l@h,r@h
    lj,rj=l@J,r@J
    value=C@(lh*rh)
    gradient=(C*rh)@lj+(C*lh)@rj
    weights=((C*rh)@l+(C*lh)@r)@D
    local=[]
    for c,weight in zip(C,weights):
        raw=L.T@(weight[:,None]*R)+lj.T@(c[:,None]*rj)
        local.append(.5*(raw+raw.T))
    quadratic=torch.stack(local)+Q0
    mean=value+torch.einsum('i,vij,j->v',location,Q0,location)+mean0
    linear=gradient+2*torch.einsum('vij,j->vi',Q0,location)
    return mean,linear,quadratic


def cross(teacher,location,projection,factors,biases):
    mean,linear,quadratic=projection
    result=mean[:,None]*affine_moment(factors,biases)
    for size in range(1,5):
        for selected in itertools.combinations(range(4),size):
            rest=[i for i in range(4) if i not in selected]
            moment=affine_moment([factors[i] for i in rest],[biases[i] for i in rest])
            chosen=[factors[i] for i in selected]
            if size==1:derivative=linear@chosen[0].T
            elif size==2:derivative=2*torch.einsum('ki,vij,kj->vk',chosen[0],quadratic,chosen[1])
            elif size==3:derivative=24*directional(*teacher,[location.expand_as(chosen[0])]+chosen).T
            else:derivative=24*directional(*teacher,chosen).T
            result=result+derivative*moment
    return result
