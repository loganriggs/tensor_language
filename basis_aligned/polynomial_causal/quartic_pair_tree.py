"""Pair-tree HT with identity leaves and an uncompressed output mode."""
import itertools
import numpy as np

def symmetrize(H):
    return sum(H.transpose((0,)+tuple(i+1 for i in perm)) for perm in itertools.permutations(range(4)))/24

def factor(H,rank):
    o,p=H.shape[:2];T=H.reshape(o,p*p,p*p)
    left=np.linalg.svd(T.transpose(1,0,2).reshape(p*p,-1),full_matrices=False)[0][:,:rank]
    right=np.linalg.svd(T.transpose(2,0,1).reshape(p*p,-1),full_matrices=False)[0][:,:rank]
    core=np.einsum('ip,oij,jq->opq',left,T,right)
    return left.reshape(p,p,-1),right.reshape(p,p,-1),core

def reconstruct(factors):
    left,right,core=factors
    return np.einsum('ija,oab,klb->oijkl',left,core,right)

def evaluate(factors,x):
    left,right,core=factors
    q=np.einsum('i,ija,j->a',x,left,x);r=np.einsum('i,ija,j->a',x,right,x)
    return np.einsum('oab,a,b->o',core,q,r)

if __name__=='__main__':
    p=5;H=np.einsum('ij,kl->ijkl',np.eye(p),np.eye(p))[None];S=symmetrize(H)
    raw=factor(H,1);full=factor(S,25);x=np.random.default_rng(83).normal(size=p)
    assert np.linalg.norm(reconstruct(raw)-H)<1e-12
    assert np.linalg.norm(reconstruct(full)-S)<1e-12
    assert abs(evaluate(raw,x)[0]-(x@x)**2)<1e-10
    assert np.linalg.matrix_rank(S.reshape(25,25))==15
    print({'raw_pair_rank':1,'symmetric_pair_rank':15,'same_polynomial_replay':float(abs(evaluate(raw,x)-evaluate(full,x)).max())})
