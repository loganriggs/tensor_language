"""Exact variable projection for shared cubic source / private quadratic query factors."""
import itertools
import torch
PERMS=list(itertools.permutations(range(3)))

def atom_gram(atoms):
    dots=torch.einsum('rid,tjd->ritj',atoms,atoms)
    return sum(dots[:,0,:,p[0]]*dots[:,1,:,p[1]]*dots[:,2,:,p[2]] for p in PERMS)/6

def cross_factors(atoms,q1,k1,q2,k2,values,output):
    # Query quadratic times physical writer, six terms per source atom/head.
    a=torch.einsum('hkd,hkz,riz->hrid',q1,k1,atoms)
    b=torch.einsum('hkd,hkz,riz->hrid',q2,k2,atoms)
    c=torch.einsum('ohk,hkz,riz->hrio',output,values,atoms)
    perm=torch.tensor(PERMS,device=atoms.device)
    return a[:,:,perm[:,0]],b[:,:,perm[:,1]],c[:,:,perm[:,2]]/6

def query_output_gram(factors):
    a,b,c=factors;heads,rank,terms,_=a.shape
    a=a.flatten(1,2);b=b.flatten(1,2);c=c.flatten(1,2)
    aa=a@a.transpose(-1,-2);bb=b@b.transpose(-1,-2)
    ab=a@b.transpose(-1,-2);ba=b@a.transpose(-1,-2);cc=c@c.transpose(-1,-2)
    return (((aa*bb+ab*ba)/2)*cc).reshape(heads,rank,terms,rank,terms).sum((2,4))

def capture(atoms,q1,k1,q2,k2,values,output):
    gram=atom_gram(atoms);kernel=query_output_gram(cross_factors(atoms,q1,k1,q2,k2,values,output))
    scale=gram.diagonal().sqrt();normalized=gram/scale[:,None]/scale[None,:]
    chol=torch.linalg.cholesky(normalized)
    rhs=kernel.sum(0)/scale[:,None]/scale[None,:]
    return torch.cholesky_solve(rhs,chol).trace()

def execute(query,source,atoms,factors):
    """Diagonal source/query evaluation, returns [batch,head,physical_output]."""
    a,b,c=factors
    qa=torch.einsum('nd,hrtd->nhrt',query,a);qb=torch.einsum('nd,hrtd->nhrt',query,b)
    cross=torch.einsum('nhrt,hrto->nhro',qa*qb,c)
    phi=torch.einsum('nd,rid->nri',source,atoms).prod(-1)
    dual=torch.linalg.solve(atom_gram(atoms),phi.T).T
    return torch.einsum('nr,nhro->nho',dual,cross)
