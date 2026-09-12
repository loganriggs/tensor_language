"""Formal joint-QK function overlap, independent of downstream common writer."""
import json,torch,time
from pathlib import Path
from folded_normalized_router_v1 import rotary
from joint_router_polynomial_gram_v1 import dense_biquadratic
P=Path(__file__).resolve().parent

def inner(a,b,c,d):
 # Each matrix A is represented as qa.T @ ka.
 def dot(x,y):return ((x[0]@y[0].T)*(x[1]@y[1].T)).sum()
 x=torch.trace((a[0]@d[0].T)@(d[1]@b[1].T)@(b[0]@c[0].T)@(c[1]@a[1].T))
 y=torch.trace((a[0]@c[0].T)@(c[1]@b[1].T)@(b[0]@d[0].T)@(d[1]@a[1].T))
 return (dot(a,c)*dot(b,d)+dot(a,d)*dot(b,c)+x+y)/4

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(191512);tic=time.perf_counter();out=P/'SCALAR_JOINT_KEY_CROSSHEAD_V1_RESULT.json';assert not out.exists()
 mats=[(torch.randn(3,5),torch.randn(3,7)) for _ in range(4)];A=[q.T@k for q,k in mats];T=dense_biquadratic(A[0],A[1]);S=dense_biquadratic(A[2],A[3]);direct=(T*S).sum();err=float(abs(inner(*mats)-direct)/(T.norm()*S.norm()))
 p=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');banks=torch.load(P/'SCALAR_JOINT_KEY_BANK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['bands'];cross=banks[0,0].T@banks[1,0];ev=torch.linalg.svdvals(cross).square();linear=float(ev.mean());sym=float((ev.sum().square()+ev.square().sum())/(64*65))
 # Independent small explicit symmetric-square support check.
 U=torch.linalg.qr(torch.randn(11,3)).Q;V=torch.linalg.qr(torch.randn(11,3)).Q
 def symbasis(W):
  return torch.stack([(torch.outer(W[:,i],W[:,j])+torch.outer(W[:,j],W[:,i]))/(2 if i==j else 2**.5) for i in range(3) for j in range(i,3)]).flatten(1)
 X,Y=symbasis(U),symbasis(V);actual=float((X@Y.T).square().sum()/6);e=torch.linalg.svdvals(U.T@V).square();formula=float((e.sum().square()+e.square().sum())/12);support_err=abs(formula-actual)
 rows=[]
 for pos in (0,15,31):
  rot=rotary(32,128).T@rotary(pos,128)
  for band in ('full','leading'):
   factors=[]
   for h in range(2):
    for qn,kn in [('q1','k1'),('q2','k2')]:
     q=p[qn][h].double();k=p[kn][h].double()
     if band=='leading':k=(k@banks[h,0])@banks[h,0].T
     factors.append((q,rot@k))
   a,b,c,d=factors;cos=float(inner(a,b,c,d)/(inner(a,b,a,b)*inner(c,d,c,d)).sqrt());rows.append(dict(source_position=pos,band=band,numerator_cosine=cos))
 valid=max(err,support_err)<=1e-10
 result=dict(pred_a=valid,pred_b=valid and all(abs(r['numerator_cosine'])<=.2 for r in rows),pred_c=valid and sym<=.1,dense_cross_inner_error=err,symmetric_support_control_error=support_err,leading_linear_support_overlap=linear,leading_symmetric_product_support_overlap=sym,rows=rows,seconds=time.perf_counter()-tic,scope='Raw-coordinate formal separately symmetric biquadratic numerator similarity and symmetric-product support envelopes. Actual contextual states differ across layers; native normalizers omitted from geometry, retained in execution. No disjoint taskspaces, behavioral interchange, or OOD claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert valid
if __name__=='__main__':main()
