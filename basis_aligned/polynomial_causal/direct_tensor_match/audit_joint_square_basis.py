"""Numerical commutator obstruction for one orthogonal square-feature basis."""
import json,time
import torch
from audit_conditional_residual_accounting import P,SCALE
from audit_root_matched_reader import CK

def pair(A,B):
 A=A/A.norm();B=B/B.norm();ea,qa=torch.linalg.eigh(A);eb,qb=torch.linalg.eigh(B);c=(A@B-B@A).norm();s=(ea.abs().max().square()+eb.abs().max().square()).sqrt();lower=c/((s.square()+c).sqrt()+s)/2**.5
 upper=[]
 for Q in [qa,qb]:
  aa=Q.T@A@Q;bb=Q.T@B@Q;off=lambda x:x-torch.diag(x.diag());upper.append(float((off(aa).square().sum()+off(bb).square().sum()).sqrt()/2**.5))
 assert float(lower)<=min(upper)+1e-12
 return dict(commutator_norm=float(c),relative_error_lower_bound=float(lower),eigenbasis_upper_bounds=upper)

def controls():
 torch.manual_seed(27000);dt=torch.float64;Q,_=torch.linalg.qr(torch.randn(5,5,dtype=dt));a=torch.randn(5,dtype=dt);b=torch.randn(5,dtype=dt);A=(Q*a)@Q.T;B=(Q*b)@Q.T;N=torch.randn(5,5,dtype=dt);N=(N+N.T)/2;cases=[('commuting',A,B),('near',A,B+1e-4*N),('generic',A,N)];V=torch.eye(5,dtype=dt)+.3*torch.randn(5,5,dtype=dt);cases.append(('nonorthogonal_shared_squares',(V*a)@V.T,(V*b)@V.T));M=N.clone();M[:2,2:]=0;M[2:,:2]=0;cases.append(('blocks',torch.diag(a),M));rows=[]
 for name,A,B in cases:rows.append(dict(kind=name,**pair(A,B)))
 assert rows[0]['commutator_norm']<1e-12 and rows[3]['commutator_norm']>1e-3
 return rows

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checks=controls();state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');W=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].double();U=state['lm_head.weight'].double();UW=U@W;readers=U.T@UW/UW.square().sum(0);del U,UW
 get=lambda n:state[f'transformer.h.17.mlp.{n}.weight'].double();L,R,D=get('Left'),get('Right'),get('Down');C=readers.T@D/SCALE;mats=[]
 for c in C:
  M=L.T@(c[:,None]*R);mats.append((M+M.T)/2)
 T=torch.stack(mats);torch.manual_seed(27001);x=torch.randn(13,1152,dtype=torch.float64);native=((x@L.T)*(x@R.T))@C.T;fold=torch.einsum('ni,vij,nj->nv',x,T,x);replay=float((native-fold).norm()/native.norm());assert replay<1e-10
 rows=[]
 for i in range(0,16,2):
  r=dict(outputs=[i,i+1],**pair(T[i],T[i+1]));rows.append(r);print(r,flush=True)
 out=dict(controls=checks,rows=rows,quadratic_replay=replay,seconds=time.monotonic()-start,scope='True singleMLP17quadratic coefficients, selected16linearoutputreaders, all1152inputs. Eightfixedpairs eachsliceunitFrobenius. Obstruction onlysharedORTHOGONALsquarebasis; notnonorthogonal/overcomplete/Tucker/DAGgeneral. Boundfloatingpoint not intervalcertificate; no functionaltext/nativeintervention inference.')
 (P/'JOINT_SQUARE_BASIS_V1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
