"""Symmetric conditional factor algebra at an actual bilinear-product interface."""
import torch
import mixed_state_projector_v1 as M


def projectors(corners, i=2, j=4):
    corners=[tuple(c) for c in corners];index={c:k for k,c in enumerate(corners)}
    q=torch.tensor(M.projector(corners,i,j),dtype=torch.float64)
    eye=torch.eye(len(corners),dtype=torch.float64)
    odd=[]
    for factor in (i,j):
        flip=[]
        for c in corners:
            x=list(c);x[factor]*=-1;flip.append(index[tuple(x)])
        odd.append((eye-eye[flip])/2)
    return torch.stack((eye-odd[0]-odd[1]+q,odd[0]-q,odd[1]-q,q))


def partition(left,right,projections):
    """[world,corner,channel]; returned products keep original corner signs."""
    l=torch.einsum('kij,wjd->kwid',projections,left)
    r=torch.einsum('kij,wjd->kwid',projections,right)
    new=l[1]*r[2]+l[2]*r[1]
    inherited=l[0]*r[3]+l[3]*r[0]
    full=torch.einsum('ij,wjd->wid',projections[3],left*right)
    return new,inherited,full


def controls():
    from itertools import product
    c=list(product((-1,1),repeat=5));q=projectors(c)
    gen=torch.Generator().manual_seed(910536)
    left=torch.randn(2,32,7,generator=gen,dtype=torch.float64)
    right=torch.randn(2,32,7,generator=gen,dtype=torch.float64)
    n,h,f=partition(left,right,q)
    error=float((n+h-f).abs().max());assert error<1e-12
    o=torch.tensor([x[2] for x in c],dtype=torch.float64)[None,:,None]
    human=torch.tensor([x[4] for x in c],dtype=torch.float64)[None,:,None]
    pn,ph,pf=partition(o,human,q)
    assert torch.equal(pn,pf) and ph.count_nonzero()==0
    pn,ph,pf=partition(torch.ones_like(o),o*human,q)
    assert torch.equal(ph,pf) and pn.count_nonzero()==0
    nn,hh,_=partition(2*left,-3*right,q)
    gauge=max(float((nn/-6-n).abs().max()),float((hh/-6-h).abs().max()))
    perm=torch.randperm(32,generator=gen);qp=projectors([c[i] for i in perm.tolist()])
    np,hp,_=partition(left[:,perm],right[:,perm],qp)
    permutation=max(float((np-n[:,perm]).abs().max()),float((hp-h[:,perm]).abs().max()))
    assert max(gauge,permutation)<1e-12
    return {'passed':True,'closure_max_abs':error,'factor_rescaling_error':gauge,
            'row_permutation_error':permutation,'new_only_and_inherited_only':True,
            'scope':'Interface algebra only, not identified semantics or standalone execution.'}
