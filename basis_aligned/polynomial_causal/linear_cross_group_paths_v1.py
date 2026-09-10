"""A cross-group intervention interaction can be a purely linear serial path."""
import json
from pathlib import Path
import numpy as np


def group_effects(a,m,x,delta):
    a0=a@x;m0=m@(x+a0);edited=x-delta
    cube=[]
    for a_live in (False,True):
        row=[]
        for m_live in (False,True):
            av=a@edited if a_live else a0
            mv=m@(edited+av) if m_live else m0
            row.append(edited+av+mv)
        cube.append(row)
    z=np.array(cube);total=z[0,0]-z[1,1]
    attention=z[0,0]-z[1,0];mlp=z[0,0]-z[0,1]
    interaction=total-attention-mlp
    return total,attention,mlp,interaction


def controls():
    rng=np.random.default_rng(910837);errors=[]
    for width in (3,8,32):
        a=rng.normal(size=(width,width))/width**.5;m=rng.normal(size=(width,width))/width**.5
        x=rng.normal(size=width);delta=rng.normal(size=width)
        total,ea,em,interaction=group_effects(a,m,x,delta)
        errors.append(float(np.max(abs(interaction-m@a@delta))))
        assert np.allclose(total,(a+m+m@a)@delta,rtol=1e-12,atol=1e-12)
    assert max(errors)<1e-12
    a=np.zeros((3,3));m=np.zeros((3,3));a[1,0]=1;m[2,1]=1
    delta=np.array([1.,0.,0.]);reader=np.array([0.,0.,1.])
    total,ea,em,interaction=group_effects(a,m,np.ones(3),delta)
    readouts=[float(reader@v) for v in (total,ea,em,interaction)]
    assert readouts==[1.,0.,0.,1.]
    assert reader@(m@a)@delta==1 and reader@(a@m)@delta==0
    # A sign-only oddness test is insufficient to distinguish linear from cubic response.
    linear=lambda t:t;cubic=lambda t:t**3
    assert linear(1)+linear(-1)==cubic(1)+cubic(-1)==0
    assert linear(.5)==.5*linear(1) and cubic(.5)!=.5*cubic(1)
    return {'passed':True,'random_matrix_interaction_max_error':max(errors),
        'pure_serial_path_reader_effects':dict(zip(('total','attention_alone','mlp_alone','interaction'),readouts)),
        'folded_MA_reader':1.,'reversed_AM_reader':0.,
        'sign_only_linearity_test_rejects_cubic':False,
        'scope':'Exact linear-control path identity, not a claim that the native transformer is linear or that an approximate Jacobian replaces it.'}

if __name__=='__main__':
    result=controls();f=Path(__file__).with_name('LINEAR_CROSS_GROUP_PATHS_V1_CONTROLS.json')
    with f.open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result))
