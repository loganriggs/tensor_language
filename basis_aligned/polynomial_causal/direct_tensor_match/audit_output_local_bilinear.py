"""Optimal Frobenius approximation with bounded real bilinear terms per output."""
import json,time
import torch
from audit_conditional_residual_accounting import P,SCALE
from audit_root_matched_reader import CK

def select(e,k):
 pos=torch.where(e>0)[0];neg=torch.where(e<0)[0]
 return pos[e[pos].argsort(descending=True)[:k]],neg[e[neg].abs().argsort(descending=True)[:k]]

def factors(e,V,k):
 pos,neg=select(e,k);paired=min(len(pos),len(neg));left=[];right=[]
 for a,b in zip(pos[:paired],neg[:paired]):
  u=V[:,a]*e[a].sqrt();v=V[:,b]*(-e[b]).sqrt();left.append(u+v);right.append(u-v)
 for a in pos[paired:]:
  u=V[:,a]*e[a].sqrt();left.append(u);right.append(u)
 for b in neg[paired:]:
  v=V[:,b]*(-e[b]).sqrt();left.append(v);right.append(-v)
 return torch.stack(left),torch.stack(right)

def controls():
 torch.manual_seed(28002);d=9;Q,_=torch.linalg.qr(torch.randn(d,d,dtype=torch.float64));rows=[]
 for kind in ['positive','negative','indefinite','rank_deficient','signed_shared']:
  lam=torch.arange(1,d+1,dtype=torch.float64)
  if kind=='negative':lam=-lam
  if kind in ['indefinite','signed_shared']:lam[::2]*=-1
  if kind=='rank_deficient':lam[3:]=0
  T=Q@torch.diag(lam)@Q.T
  if kind=='signed_shared':T=-2*T
  e,V=torch.linalg.eigh(T);x=torch.randn(13,d,dtype=torch.float64)
  for k in [2,d]:
   pos,neg=select(e,k);ids=torch.cat([pos,neg]);hat=(V[:,ids]*e[ids])@V[:,ids].T;left,right=factors(e,V,k)
   y=((x@left.T)*(x@right.T)).sum(1);ref=torch.einsum('ni,ij,nj->n',x,hat,x)
   err=float((y-ref).norm()/ref.norm());tail=e.clone();tail[ids]=0
   terr=float(abs((T-hat).norm()-tail.norm())/T.norm());assert max(err,terr)<1e-12
   rows.append(dict(kind=kind,k=k,replay=err,tail_identity=terr))
 return rows

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checks=controls()
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');W=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].double();U=state['lm_head.weight'].double();UW=U@W;readers=U.T@UW/UW.square().sum(0);del U,UW
 get=lambda n:state[f'transformer.h.17.mlp.{n}.weight'].double()
 L,R,D=get('Left'),get('Right'),get('Down');C=readers.T@D/SCALE;mat=[]
 for c in C:
  a=L.T@(c[:,None]*R);mat.append((a+a.T)/2)
 T=torch.stack(mat);e,V=torch.linalg.eigh(T);energy=e.square().sum(1);tr=e.sum(1);rows=[]
 for k in [8,16,32,64,128,288,512,1152]:
  tail=e.clone();products=0;unique_readers=0
  for v in range(16):
   pos,neg=select(e[v],k);tail[v,pos]=0;tail[v,neg]=0;products+=max(len(pos),len(neg));unique_readers+=len(pos)+len(neg)
  sq=tail.square().sum(1);per=(sq/energy).sqrt()
  rows.append(dict(k=k,products=products,stored_reader_floats=unique_readers*1152,natural_centered_error=float((sq.sum()/energy.sum()).sqrt()),equal_output_centered_error=float((sq/energy).mean().sqrt()),per_output_centered_errors=per.tolist(),uncentered_gaussian_error=float(((2*sq+tail.sum(1).square()).sum()/(2*energy+tr.square()).sum()).sqrt())))
 adaptive=[]
 gains=torch.zeros_like(e)
 for v in range(16):
  pos,neg=select(e[v],1152)
  gains[v,:len(pos)]+=e[v,pos].square();gains[v,:len(neg)]+=e[v,neg].square()
 for metric in ['natural','equal_output']:
  scores=gains if metric=='natural' else gains/energy[:,None]
  order=scores.flatten().argsort(descending=True)
  for budget in [512,1024,2048,4608]:
   selected=order[:budget];ks=torch.bincount(selected//1152,minlength=16);tail=e.clone();readers_count=0
   for v,k in enumerate(ks.tolist()):
    pos,neg=select(e[v],k);tail[v,pos]=0;tail[v,neg]=0;readers_count+=len(pos)+len(neg)
   sq=tail.square().sum(1);per=(sq/energy).sqrt()
   adaptive.append(dict(metric=metric,products=budget,allocation=ks.tolist(),stored_reader_floats=readers_count*1152,natural_centered_error=float((sq.sum()/energy.sum()).sqrt()),equal_output_centered_error=float((sq/energy).mean().sqrt()),per_output_centered_errors=per.tolist(),uncentered_gaussian_error=float(((2*sq+tail.sum(1).square()).sum()/(2*energy+tr.square()).sum()).sqrt())))
 x=torch.randn(13,1152,dtype=torch.float64);native=((x@L.T)*(x@R.T))@C.T;fold=torch.einsum('ni,vij,nj->nv',x,T,x);replay=float((native-fold).norm()/native.norm());assert replay<1e-10
 factor_errors=[]
 for v in range(16):
  a,b=factors(e[v],V[v],64);p,n=select(e[v],64);ids=torch.cat([p,n]);ref=((x@V[v,:,ids]).square()*e[v,ids]).sum(1);y=((x@a.T)*(x@b.T)).sum(1);err=float((y-ref).norm()/ref.norm());assert err<1e-10;factor_errors.append(err)
 primary=next(r for r in rows if r['k']==288)
 out=dict(adaptive=adaptive,rows=rows,controls=checks,native_replay=replay,k64_factor_replays=factor_errors,pred_centered=primary['natural_centered_error']<.2,pred_all_outputs=max(primary['per_output_centered_errors'])<.3,seconds=time.monotonic()-start,scope='Independent output-local bilinear dictionaries; optimal coefficient approximation per slice at k terms. Not global shared-circuit optimum or text fidelity. Reader storage excludes signs/destination metadata; square readers shared.')
 (P/'OUTPUT_LOCAL_BILINEAR_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 for r in rows:print(r['k'],r['products'],r['natural_centered_error'],max(r['per_output_centered_errors']))
if __name__=='__main__':main()
