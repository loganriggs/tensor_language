"""Construct block terms from an output SVD and signed quadratic spectra."""
import json,time
import torch
from audit_conditional_residual_accounting import P,SCALE
from audit_root_matched_reader import CK
from audit_output_local_bilinear import select

def rotate(T):
 flat=T.flatten(1);e,W=torch.linalg.eigh(flat@flat.T);W=W.flip(1);A=(W.T@flat).reshape_as(T)
 return W,A

def construct(T,budget):
 W,A=rotate(T);e,V=torch.linalg.eigh(A);gains=torch.zeros_like(e);d=e.shape[1]
 for v in range(len(e)):
  p,n=select(e[v],d);gains[v,:len(p)]+=e[v,p].square();gains[v,:len(n)]+=e[v,n].square()
 ids=gains.flatten().argsort(descending=True)[:budget];ks=torch.bincount(ids//d,minlength=len(e));res=[];readers=0
 for v,k in enumerate(ks.tolist()):
  p,n=select(e[v],k);take=torch.cat([p,n]);hat=(V[v][:,take]*e[v,take])@V[v][:,take].T;res.append(A[v]-hat);readers+=len(take)
 E=(W@torch.stack(res).flatten(1)).reshape_as(T)
 return E,ks,readers,W,A

def controls():
 torch.manual_seed(32000);rows=[]
 for seed in range(5):
  d=9;V,_=torch.linalg.qr(torch.randn(d,d,dtype=torch.float64));W,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));A=torch.stack([(v+1)*torch.outer(V[:,v],V[:,v]) for v in range(3)]);T=(W@A.flatten(1)).reshape_as(A);E,_,_,q,a=construct(T,3)
  err=float(E.norm()/T.norm());x=torch.randn(11,d,dtype=torch.float64);y=torch.einsum('ni,vij,nj->nv',x,T,x);z=torch.einsum('ni,vij,nj->nv',x,a,x)@q.T;replay=float((y-z).norm()/y.norm());assert max(err,replay)<1e-12;rows.append(dict(seed=seed,exact_reconstruction=err,rotation_replay=replay))
 return rows

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checks=controls()
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');W=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].double();U=state['lm_head.weight'].double();UW=U@W;readers=U.T@UW/UW.square().sum(0);del U,UW
 get=lambda n:state[f'transformer.h.17.mlp.{n}.weight'].double();L,R,D=get('Left'),get('Right'),get('Down');C=readers.T@D/SCALE;mat=[]
 for c in C:
  a=L.T@(c[:,None]*R);mat.append((a+a.T)/2)
 T=torch.stack(mat);energy=T.square().sum((1,2));gf=lambda X:2*X.square().sum((1,2))+X.diagonal(dim1=-2,dim2=-1).sum(1).square();baselines=json.loads((P/'OUTPUT_LOCAL_BILINEAR_V1.json').read_text());rows=[]
 for metric in ['natural','equal_output']:
  scale=torch.ones_like(energy) if metric=='natural' else energy.sqrt()
  for budget in [512,1024,2048,4608]:
   E,ks,count,W,A=construct(T/scale[:,None,None],budget);E*=scale[:,None,None];W=scale[:,None]*W;sq=E.square().sum((1,2));ref=next(r for r in baselines['adaptive'] if r['metric']==metric and r['products']==budget);err=float((sq.sum()/energy.sum()).sqrt());balanced=float((sq/energy).mean().sqrt());field='natural_centered_error' if metric=='natural' else 'equal_output_centered_error';ratio=(err if metric=='natural' else balanced)/ref[field];rows.append(dict(metric=metric,products=budget,allocation=ks.tolist(),stored_floats=count*1152+256,natural_centered_error=err,equal_output_centered_error=balanced,per_original_output_centered_errors=(sq/energy).sqrt().tolist(),uncentered_gaussian_error=float((gf(E).sum()/gf(T).sum()).sqrt()),original_basis_error=ref[field],relative_error_ratio=ratio,pred_improvement=ratio<=.8 if metric=='natural' and budget in [512,4608] else None));print(metric,budget,err,balanced,ratio,flush=True)
 x=torch.randn(11,1152,dtype=torch.float64);native=((x@L.T)*(x@R.T))@C.T;rot=torch.einsum('ni,vij,nj->nv',x,A,x)@W.T;replay=float((rot-native).norm()/native.norm());assert replay<1e-10
 out=dict(rows=rows,controls=checks,native_rotation_replay=replay,seconds=time.monotonic()-start,scope='Selected16-output true MLP17 quadratic. Exact output rotation plus restricted independent-slice spectral optimization. Not global block-term optimum, not native text or causal evidence. Sign/destination metadata excluded from float counts.')
 (P/'OUTPUT_ROTATED_BLOCK_V1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
