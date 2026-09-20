"""Implicit third-order tensor for the complete previous-MLP-dependent numerator.

Teacher=(C,L,R,Dprev). Inputs are midpoint n=h-m/2 and previous channel products p,
with m=Dprev p. Normalization denominator remains external and explicit.
"""
import torch

def evaluate(teacher,n,p):
 C,L,R,D=teacher;m=p@D.T
 return ((n@L.T)*(m@R.T)+(m@L.T)*(n@R.T))@C.T

def student_value(student,n,p):
 C,A,B=student
 return ((n@A.T)*(p@B.T))@C.T

def student_inner(first,second):
 C,A,B=first;W,P,Q=second
 return ((C.T@W)*(A@P.T)*(B@Q.T)).sum()

def teacher_cross(teacher,student):
 C,L,R,D=teacher;W,A,B=student;projected=D@B.T
 # Avoid materializing L@D and R@D, or the output-by-input-by-channel tensor.
 return ((C.T@W)*((L@A.T)*(R@projected)+(R@A.T)*(L@projected))).sum()

def toy_check():
 gen=torch.Generator().manual_seed(261030);rand=lambda *shape:torch.randn(*shape,generator=gen,dtype=torch.float64)
 teacher=(rand(5,9),rand(9,7),rand(9,7),rand(7,11));C,L,R,D=teacher;n=rand(19,7);p=rand(19,11);m=p@D.T;h=n+m/2
 bilinear=lambda x:((x@L.T)*(x@R.T))@C.T
 reference=bilinear(h)-bilinear(h-m);value=evaluate(teacher,n,p);replay=float((value-reference).norm()/reference.norm());assert replay<1e-12
 dense=torch.einsum('vk,ki,kj->vij',C,L,R@D)+torch.einsum('vk,ki,kj->vij',C,R,L@D)
 expanded=torch.einsum('vij,bi,bj->bv',dense,n,p);dense_error=float((value-expanded).norm()/value.norm());assert dense_error<1e-12
 with torch.enable_grad():
  student=tuple(t.requires_grad_() for t in [rand(5,3),rand(3,7),rand(3,11)])
  T=torch.einsum('vk,ki,kj->vij',*student);implicit=student_inner(student,student)-2*teacher_cross(teacher,student);direct=T.square().sum()-2*(dense*T).sum();error=float((implicit-direct).abs().detach()/direct.detach().abs());g1=torch.autograd.grad(implicit,student,retain_graph=True);g2=torch.autograd.grad(direct,student);gradient=max(float((a-b).norm()/b.norm()) for a,b in zip(g1,g2));assert error<1e-12 and gradient<1e-12
 return dict(midpoint_identity=replay,dense_tensor_replay=dense_error,implicit_objective_replay=error,gradient_replay=gradient,scope='Independent midpoint/channel-product coordinates. Exact mixed tensor coefficients; no Gaussian functional metric or native compression claim.')
if __name__=='__main__':
 import json
 from pathlib import Path
 r=toy_check();Path(__file__).with_name('MIDPOINT_FOLDED_TENSOR_ORACLE_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
