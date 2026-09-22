"""Rectangular exact Grams and streamed envelope gradients for shared quartics.
Keeps the fitted output readout detached: derivative of the profiled ridge
objective equals the fixed-readout derivative at its exact minimizer.
"""
import itertools
import torch
from shared_gaussian_moments import PARTITIONS, native_cross as gaussian_cross
from sparse_quartic_bank import native_cross as coefficient_cross


def coefficient_rect(U,V,left_pairs,right_pairs):
    E=torch.cat([U,V],1);F=torch.cat([V,U],1)/2
    edge=torch.einsum('iad,jbd->ijab',F,E)
    quad=torch.einsum('ijab,jiba->ij',edge,edge)
    i,j=left_pairs;k,l=right_pairs
    first=quad[i[:,None],k[None,:]]*quad[j[:,None],l[None,:]]
    first=first+quad[i[:,None],l[None,:]]*quad[j[:,None],k[None,:]]
    a=edge[i[:,None],k[None,:]];b=edge[k[None,:],j[:,None]]
    c=edge[j[:,None],l[None,:]];d=edge[l[None,:],i[:,None]]
    return (first+4*((a@b@c)*d.transpose(-1,-2)).sum((-1,-2)))/6

def gaussian_rect(U,V,left_pairs,right_pairs,bias_u,bias_v,chunk=16):
    E=torch.cat([U,V],1);F=torch.cat([V,U],1)/2
    ell=(bias_u[...,None]*V+bias_v[...,None]*U).sum(1)
    mean=(U*V).sum((1,2))+(bias_u*bias_v).sum(1)
    edge=torch.einsum('iad,jbd->ijab',F,E)
    left=torch.einsum('id,jad->ija',ell,E)
    right=torch.einsum('iad,jd->ija',F,ell)
    linear_gram=ell@ell.T
    rows=[]
    for start in range(0,left_pairs.shape[1],chunk):
        a,b=left_pairs[:,start:start+chunk]
        slots=[a[:,None],b[:,None],right_pairs[0][None,:],right_pairs[1][None,:]]
        cumulants={}
        def cycle(order):
            matrices=[edge[slots[i],slots[j]] for i,j in zip(order,order[1:]+order[:1])]
            result=matrices[0]
            for matrix in matrices[1:-1]:result=result@matrix
            return (result*matrices[-1].transpose(-1,-2)).sum((-1,-2))
        for size in range(1,5):
            for chosen in itertools.combinations(range(4),size):
                if size==1:value=mean[slots[chosen[0]]]
                elif size==2:
                    i,j=chosen;value=2*cycle(chosen)+linear_gram[slots[i],slots[j]]
                else:
                    # For symmetric A, trace cycles equal their reversals.
                    orders=[(chosen[0],)+tail for tail in itertools.permutations(chosen[1:]) if tail<=tail[::-1]]
                    value=sum((2**size)*cycle(order) for order in orders)
                    linear=0.
                    for i,j in itertools.combinations(chosen,2):
                        middle=[k for k in chosen if k not in (i,j)]
                        for order in itertools.permutations(middle):
                            l=left[slots[i],slots[order[0]]]
                            r=right[slots[order[-1]],slots[j]]
                            term=(l*r).sum(-1) if size==3 else torch.einsum('...a,...ab,...b->...',l,edge[slots[order[0]],slots[order[1]]],r)
                            linear=linear+term
                    value=value+(2**(size-2))*linear
                cumulants[chosen]=value
        result=mean.new_zeros((len(a),right_pairs.shape[1]))
        for partition in PARTITIONS:
            term=1.
            for block in partition:term=term*cumulants[tuple(sorted(block))]
            result=result+term
        rows.append(result)
        # Avoid retaining tensor graphs through the local cycle-function closure.
        cumulants.clear()
    return torch.cat(rows)



def streamed_gradient(U,V,pairs,C,teacher,transformed,location,projection,S,mu,lam,chunk=8):
    """Accumulate exact gradients into leaf U/V; C must be detached optimal fit.
    Rebuild transformed features for every block, then release its autograd graph.
    Returns factor-dependent objective (ridge term is independent at fixed C).
    """
    assert U.is_leaf and V.is_leaf and U.requires_grad and V.requires_grad
    C=C.detach();total=0.;mix=1+lam;K=C.T@C
    for start in range(0,pairs.shape[1],chunk):
        block=pairs[:,start:start+chunk];stop=start+block.shape[1]
        scalar=lam/mix*(coefficient_rect(U,V,block,pairs)*K[start:stop]).sum()
        scalar.backward();total+=float(scalar.detach())
        scalar=(gaussian_rect(U@S,V@S,block,pairs,U@mu,V@mu)*K[start:stop]).sum()/mix
        scalar.backward();total+=float(scalar.detach())
        scalar=-2*lam/mix*(coefficient_cross(teacher,U,V,block)*C[:,start:stop]).sum()
        scalar.backward();total+=float(scalar.detach())
        scalar=-2/mix*(gaussian_cross(transformed,location,projection,U@S,V@S,block,U@mu,V@mu,chunk=chunk)*C[:,start:stop]).sum()
        scalar.backward();total+=float(scalar.detach())
    return total
