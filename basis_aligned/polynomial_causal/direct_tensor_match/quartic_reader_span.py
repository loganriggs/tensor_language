"""Exact local sensitivity of a pure native-format quartic to unread inputs."""
import torch


def forward(teacher,x):
    C,l,r,D,A,B=teacher
    h=((x@A.T)*(x@B.T))@D.T
    return ((h@l.T)*(h@r.T))@C.T


def jacobian(teacher,x):
    C,l,r,D,A,B=teacher
    a,b=A@x,B@x;h=D@(a*b)
    first=D@(a[:,None]*B+b[:,None]*A)
    second=C@((l@h)[:,None]*r+(r@h)[:,None]*l)
    return second@first


def reader_basis(U,V):
    directions=torch.cat([U.flatten(0,1),V.flatten(0,1)])
    # SVD ensures a rank-deficient planted dictionary is handled correctly.
    _,s,vh=torch.linalg.svd(directions,full_matrices=False)
    return vh[s>s[0]*1e-10].T


def controls():
    torch.manual_seed(4800);torch.set_default_dtype(torch.float64)
    t=[torch.randn(*s) for s in [(3,4),(4,5),(4,5),(5,6),(6,7),(6,7)]];x=torch.randn(7)
    j=jacobian(t,x);ref=torch.func.jacrev(lambda z:forward(t,z))(x);relative=float((j-ref).norm()/ref.norm())
    eye=torch.eye(7);u=eye[0].reshape(1,1,7);v=eye[1].reshape(1,1,7);q=reader_basis(u,v)
    one=torch.ones(1,1)
    visible=[one,one,one,one,eye[0:1],eye[1:2]]
    invisible=[one,one,one,one,eye[2:3],eye[2:3]]
    results=[]
    for target in [visible,invisible]:
        j=jacobian(target,x);lost=j-(j@q)@q.T;results.append(float(lost.norm()/j.norm()))
    assert relative<1e-12 and results[0]<1e-12 and abs(results[1]-1)<1e-12
    return dict(jacobian_relative_error=relative,visible_lost_fraction=results[0],unread_lost_fraction=results[1])
if __name__=='__main__':
    import json
    from pathlib import Path
    result=controls();Path(__file__).with_name('QUARTIC_READER_SPAN_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
