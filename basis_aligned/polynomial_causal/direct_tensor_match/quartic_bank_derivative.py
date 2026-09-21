"""Exact derivatives of shared quadratic-bank pair products."""
import torch
from quartic_reader_span import reader_basis


def feature_jacobian(U,V,x):
    a,b=U@x,V@x;q=(a*b).sum(1)
    jq=(a[:,:,None]*V+b[:,:,None]*U).sum(1)
    i,j=torch.triu_indices(len(U),len(U),device=x.device)
    return q[i,None]*jq[j]+q[j,None]*jq[i]


def split_error(teacher_j,student_j,basis):
    e=teacher_j-student_j
    parallel=(e@basis)@basis.T
    perpendicular=e-parallel
    return dict(total=e.square().sum(),within=parallel.square().sum(),outside=perpendicular.square().sum(),teacher=teacher_j.square().sum(),student_outside=(student_j-(student_j@basis)@basis.T).square().sum())


def controls():
    from empirical_quartic_dictionary import features
    torch.manual_seed(5200);torch.set_default_dtype(torch.float64);rows=[]
    for m,k,d in [(2,1,7),(3,2,5),(4,2,9),(1,1,4),(32,4,7)]:
        u,v=torch.randn(m,k,d),torch.randn(m,k,d);x=torch.randn(d);L=torch.randn(d,d)+torch.eye(d)
        got=feature_jacobian(u,v,x);ref=torch.func.jacrev(lambda z:features(z[None],u,v)[0])(x);err=float((got-ref).norm()/ref.norm());basis=reader_basis(u@L,v@L);student=torch.randn(3,len(got))@got@L;teacher=torch.randn_like(student);terms=split_error(teacher,student,basis);closure=float(abs(terms['total']-terms['within']-terms['outside'])/terms['total']);leak=float((terms['student_outside']/student.square().sum()).sqrt());assert max(err,closure,leak)<1e-12;rows.append(dict(width=m,products=k,input_width=d,derivative_error=err,closure_error=closure,student_unread_fraction=leak))
    return rows
if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2);rows=controls();Path(__file__).with_name('QUARTIC_BANK_DERIVATIVE_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
