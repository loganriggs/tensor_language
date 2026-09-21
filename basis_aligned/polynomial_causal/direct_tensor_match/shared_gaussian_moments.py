"""Exact noncentral Gaussian Gram of shared products of low-rank quadratics.
Input factors describe q_i(z)=sum_k (U_ik z+b_ik)(V_ik z+c_ik), z~N(0,I).
"""
import itertools
import torch


def partitions(items):
    if not items:
        yield ()
        return
    first,*rest=items
    for partition in partitions(rest):
        yield ((first,),)+partition
        for j,block in enumerate(partition):
            yield partition[:j]+((first,)+block,)+partition[j+1:]

PARTITIONS=tuple(partitions([0,1,2,3]))


def gram(U,V,pairs,bias_u,bias_v,chunk=64):
    E=torch.cat([U,V],1);F=torch.cat([V,U],1)/2
    ell=(bias_u[...,None]*V+bias_v[...,None]*U).sum(1)
    mean=(U*V).sum((1,2))+(bias_u*bias_v).sum(1)
    edge=torch.einsum('iad,jbd->ijab',F,E)
    left=torch.einsum('id,jad->ija',ell,E)
    right=torch.einsum('iad,jd->ija',F,ell)
    linear_gram=ell@ell.T
    rows=[]
    for start in range(0,pairs.shape[1],chunk):
        a,b=pairs[:,start:start+chunk]
        slots=[a[:,None],b[:,None],pairs[0][None,:],pairs[1][None,:]]
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
        result=mean.new_zeros((len(a),pairs.shape[1]))
        for partition in PARTITIONS:
            term=1.
            for block in partition:term=term*cumulants[tuple(sorted(block))]
            result=result+term
        rows.append(result)
        # Avoid retaining tensor graphs through the local cycle-function closure.
        cumulants.clear()
    return torch.cat(rows)


def native_cross(teacher,location,projection,U,V,pairs,bias_u,bias_v,chunk=16):
    """Sum exact affine-CP cross terms; expand only one bounded root-pair batch."""
    from noncentral_gaussian_cp import cross
    results=[];k=U.shape[1]
    for block in pairs.split(chunk,dim=1):
        i,j=block;n=len(i)
        vectors=[U[i,:,None,:].expand(n,k,k,-1).reshape(n*k*k,-1),
                 V[i,:,None,:].expand(n,k,k,-1).reshape(n*k*k,-1),
                 U[j,None,:,:].expand(n,k,k,-1).reshape(n*k*k,-1),
                 V[j,None,:,:].expand(n,k,k,-1).reshape(n*k*k,-1)]
        biases=[bias_u[i,:,None].expand(n,k,k).reshape(-1),bias_v[i,:,None].expand(n,k,k).reshape(-1),
                bias_u[j,None,:].expand(n,k,k).reshape(-1),bias_v[j,None,:].expand(n,k,k).reshape(-1)]
        value=cross(teacher,location,projection,vectors,biases)
        results.append(value.reshape(value.shape[0],n,k*k).sum(-1))
    return torch.cat(results,dim=1)
