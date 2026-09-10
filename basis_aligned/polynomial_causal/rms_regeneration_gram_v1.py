"""Exact conditional RMS interaction from a 3x3 Gram matrix after raw removal.

Four corners ordered (-1,-1),(-1,+1),(+1,-1),(+1,+1).
Inputs and any downstream linear maps remain explicitly charged.
"""
import itertools,json
from pathlib import Path
import numpy as np
SIGNS=np.array(list(itertools.product((-1.,1.),repeat=2)))
BASIS=np.column_stack((np.ones(4),SIGNS,SIGNS.prod(1)))


def reconstruct(raw,epsilon,reader):
    x=np.asarray(raw,dtype=np.float64)
    assert x.ndim==2 and x.shape[0]==4 and epsilon>0
    coefficients=BASIS.T@x/4
    a=coefficients[:3] # x0,xo,xh; remove the raw mixed coefficient
    gram=a@a.T/x.shape[1]
    weights=BASIS[:,:3]
    squared_norm=np.einsum('ni,ij,nj->n',weights,gram,weights)
    scale=1/np.sqrt(squared_norm+epsilon)
    scale_coefficients=BASIS.T@scale/4
    # g_oh*x0 + g_h*xo + g_o*xh
    mixture=scale_coefficients[[3,2,1]]
    mixed=mixture@a
    # Fold each downstream linear map into the three value vectors, not the RMS.
    folded=mixture@(a@reader.T)
    direct_raw=weights@a
    direct=direct_raw/np.sqrt((direct_raw**2).mean(-1,keepdims=True)+epsilon)
    direct_mixed=BASIS[:,3]@direct/4
    return {'mixed':mixed,'direct_mixed':direct_mixed,'folded':folded,
            'direct_read':direct_mixed@reader.T,'gram':gram,
            'squared_norm':squared_norm,'scale_coefficients':scale_coefficients}


def controls():
    rng=np.random.default_rng(910746);records=[]
    for width in (3,16,128,1152):
        for _ in range(8):
            raw=rng.normal(size=(4,width));reader=rng.normal(size=(5,width))
            r=reconstruct(raw,np.finfo(np.float32).eps,reader)
            records.append({'width':width,'mixed_error':float(np.max(np.abs(r['mixed']-r['direct_mixed']))),
                            'folded_error':float(np.max(np.abs(r['folded']-r['direct_read'])))})
    orthogonal=BASIS[:,:3]@np.eye(3)
    orth=reconstruct(orthogonal,1e-7,np.eye(3))
    nonorth=np.column_stack((2+SIGNS[:,0],.5+SIGNS[:,1]))
    regen=reconstruct(nonorth,1e-7,np.eye(2))
    assert np.linalg.norm(orth['mixed'])==0
    assert np.linalg.norm(regen['mixed'])>.01
    assert max(r['mixed_error'] for r in records)<1e-12
    assert max(r['folded_error'] for r in records)<1e-12
    return {'passed':True,'records':records,'orthogonal_mixed_norm':float(np.linalg.norm(orth['mixed'])),
        'nonorthogonal_mixed_norm':float(np.linalg.norm(regen['mixed'])),
        'scope':'Exact synthetic conditional Gram interface and folded linear reader; not native identification or OOD evidence.'}

if __name__=='__main__':
    result=controls();dest=Path(__file__).with_name('RMS_REGENERATION_GRAM_V1_CONTROLS.json')
    assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'max_mixed_error':max(r['mixed_error'] for r in result['records']),
                      'max_folded_error':max(r['folded_error'] for r in result['records']),
                      'nonorthogonal_mixed_norm':result['nonorthogonal_mixed_norm']}))
