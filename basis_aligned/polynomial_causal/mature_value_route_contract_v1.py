"""Canonical mature-source/common-query contraction; factors remain native-defined."""
import numpy as np


def contract(pattern,values,output_weight,local_scale):
    # [corner,head,query,source], [corner,source,head,channel], [output,head,channel].
    return local_scale*np.einsum('nhqs,nshd,ohd->nqo',pattern,values,output_weight)


def controls(corners):
    from mixed_state_projector_v1 import projector
    q=projector(corners,2,4)
    index={tuple(c):i for i,c in enumerate(corners)};mean=np.zeros_like(q)
    for i,c in enumerate(corners):
        for fo,fh in ((1,1),(-1,1),(1,-1),(-1,-1)):
            other=list(c);other[2]*=fo;other[4]*=fh;mean[i,index[tuple(other)]]=.25
    rng=np.random.default_rng(910711)
    pattern=rng.normal(size=(32,2,2,3));pattern[:,:,0,2]=0 # to cannot see future action.
    value=np.einsum('ij,jshd->ishd',q,rng.normal(size=(32,3,2,4)))
    weight=rng.normal(size=(5,2,4));scale=1.65625
    original=contract(pattern,value,weight,scale)
    routing_mean=np.einsum('ij,jhqs->ihqs',mean,pattern)
    selected=contract(routing_mean,value,weight,scale)
    error=float(np.max(abs(np.einsum('ij,jqo->iqo',q,original)-selected)))
    assert error<1e-12
    assert np.max(abs(np.einsum('ij,jqo->iqo',q,selected)-selected))<1e-12
    return {'passed':True,'conditional_mean_routing_identity_max_abs':error,
            'sources':['first_joint_information','to','action'],'queries':['to','action'],
            'causal_pattern_mask':[[1,1,0],[1,1,1]],
            'scope':'Synthetic exact contraction and fixed information-stage geometry. First joint-information token has different noun roles across layouts; no native factor swap or independent producer claimed.'}
