"""Price distinct quadratic interactions in a fixed shared orthogonal basis."""
import json,time
import torch
from audit_conditional_residual_accounting import P,SCALE
from audit_root_matched_reader import CK

def masks(T):
 d=T.shape[-1];norm=T.flatten(1).norm(dim=1);A=T/norm[:,None,None];H=sum(a@a for a in A);_,Q=torch.linalg.eigh(H);core=Q.T@A@Q;energy=core.square().sum(0);i,j=torch.triu_indices(d,d);gain=energy[i,j]*torch.where(i==j,1.,2.);order=gain.argsort(descending=True);result=[]
 diag=torch.eye(d,dtype=torch.bool);result.append(('squares',diag))
 used=set();block=diag.clone();edges=torch.triu_indices(d,d,1);priority=energy[edges[0],edges[1]].argsort(descending=True)
 for k in priority.tolist():
  a,b=map(int,edges[:,k]);
  if a not in used and b not in used:used.update([a,b]);block[a,b]=block[b,a]=True
  if len(used)>=d:break
 result.append(('paired_blocks',block))
 for k in [d,2*d,4*d]:
  take=order[:min(k,len(order))];mask=torch.zeros(d,d,dtype=torch.bool);mask[i[take],j[take]]=True;mask[j[take],i[take]]=True;result.append((f'top_{k}',mask))
 return norm,Q,core,result

def measure(T):
 n,Q,G,choices=masks(T);d=T.shape[-1];rows=[]
 for name,mask in choices:
  E=G*(~mask);natural=E*n[:,None,None];gf=lambda A:2*A.square().sum((1,2))+A.diagonal(dim1=-2,dim2=-1).sum(1).square();count=int(torch.triu(mask).sum())
  rows.append(dict(centered_gaussian_error=float(natural.norm()/T.norm()),target_mean_energy_fraction=float((G*n[:,None,None]).diagonal(dim1=-2,dim2=-1).sum(1).square().sum()/gf(G*n[:,None,None]).sum()),residual_mean_energy_fraction=float(natural.diagonal(dim1=-2,dim2=-1).sum(1).square().sum()/gf(natural).sum()),name=name,products=count,stored_floats=d*d+len(T)*count,equal_output_coefficient_error=float(E.norm()/G.norm()),natural_coefficient_error=float(natural.norm()/T.norm()),feature_coefficient_errors=E.flatten(1).norm(dim=1).tolist(),natural_gaussian_error=float((gf(natural).sum()/gf(G*n[:,None,None]).sum()).sqrt()),equal_output_gaussian_error=float((gf(E).sum()/gf(G).sum()).sqrt())))
 return Q,G,n,choices,rows

def controls():
 torch.manual_seed(28000);d=12;out=3;dt=torch.float64;Q,_=torch.linalg.qr(torch.randn(d,d,dtype=dt));rows=[]
 for kind in ['diagonal','paired_blocks','sparse','dense','signed_shared']:
  A=torch.randn(out,d,d,dtype=dt);A=(A+A.transpose(-1,-2))/2
  if kind=='diagonal':A=torch.diag_embed(A.diagonal(dim1=-2,dim2=-1))
  if kind=='paired_blocks':A*=torch.arange(d)[:,None]//2==torch.arange(d)[None,:]//2
  if kind=='sparse':A*=torch.eye(d,dtype=dt)+torch.eye(d,dtype=dt).roll(1,0)+torch.eye(d,dtype=dt).roll(-1,0)
  if kind=='signed_shared':A[1]=-A[0];A[2]=2*A[0]
  T=Q@A@Q.T;q,g,n,choices,metrics=measure(T);x=torch.randn(9,d,dtype=dt);native=torch.einsum('ni,vij,nj->nv',x,T,x);replay=torch.einsum('ni,vij,nj->nv',x@q,g*n[:,None,None],x@q);err=float((native-replay).norm()/native.norm());assert err<1e-12
  if kind=='diagonal':assert metrics[0]['equal_output_coefficient_error']<1e-10
  if kind=='paired_blocks':assert metrics[1]['equal_output_coefficient_error']<1e-10
  rows.append(dict(kind=kind,replay=err,metrics=metrics))
 return rows

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checks=controls();state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');W=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].double();U=state['lm_head.weight'].double();UW=U@W;readers=U.T@UW/UW.square().sum(0);del U,UW;get=lambda n:state[f'transformer.h.17.mlp.{n}.weight'].double();L,R,D=get('Left'),get('Right'),get('Down');C=readers.T@D/SCALE;T=[]
 for c in C:
  A=L.T@(c[:,None]*R);T.append((A+A.T)/2)
 T=torch.stack(T);Q,G,n,choices,rows=measure(T);x=torch.randn(11,1152,dtype=torch.float64);ref=((x@L.T)*(x@R.T))@C.T;fold=torch.einsum('ni,vij,nj->nv',x@Q,G*n[:,None,None],x@Q);replay=float((ref-fold).norm()/ref.norm());assert replay<1e-10
 result=dict(controls=checks,rows=rows,native_replay=replay,native_price=dict(products=4608,stored_floats=2*4608*1152+16*4608),seconds=time.monotonic()-start,scope='True nativeMLP17quadratic selected16outputprojections/all1152inputs. Singleenergy-eigenbasis, nojointoptimization. TopKoptimalonlyinthisfixedbasisandnormalizedcoefficientmetric. Gaussianfunctionerrors includetraces, nottextmetrics. Counts share eachpairproductonce; noexport/causalclaim.')
 (P/'ORTHOGONAL_INTERACTION_V1.json').write_text(json.dumps(result,indent=2)+'\n')
 for r in rows:print(r['name'],r['products'],r['equal_output_coefficient_error'],r['natural_gaussian_error'])
if __name__=='__main__':main()
