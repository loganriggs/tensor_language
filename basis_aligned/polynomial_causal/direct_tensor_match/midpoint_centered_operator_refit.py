"""Exact original-weight normal equations for a fixed centered product dictionary."""
from pathlib import Path
import json,time,torch

def cross_gram(A,B,L,R,Mn,Mm):
 return (A.T@Mn@L.T)*(B.T@Mm@R.T)+(A.T@Mn@R.T)*(B.T@Mm@L.T)

def toy():
 g=torch.Generator().manual_seed(26202)
 rand=lambda *shape:torch.randn(*shape,generator=g,dtype=torch.float64)
 n,m,A,B,L,R,C=rand(7,4),rand(9,4),rand(4,3),rand(4,3),rand(5,4),rand(5,4),rand(2,5)
 n-=n.mean(0);m-=m.mean(0);Mn=n.T@n/len(n);Mm=m.T@m/len(m)
 X=((n@A)[:,None,:]*(m@B)[None,:,:]).reshape(-1,3)
 Y=(((n@L.T)[:,None,:]*(m@R.T)[None,:,:])+((n@R.T)[:,None,:]*(m@L.T)[None,:,:])).reshape(-1,5)@C.T
 K=(A.T@Mn@A)*(B.T@Mm@B);cross=cross_gram(A,B,L,R,Mn,Mm)@C.T
 err=float((cross-X.T@Y/len(X)).norm()/cross.norm());gram=float((K-X.T@X/len(X)).norm()/K.norm())
 assert max(err,gram)<1e-12
 return dict(cross_replay=err,gram_replay=gram)

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter();p=Path(__file__).resolve().parent
 out=p/'MIDPOINT_CENTERED_OPERATOR_REFIT_V1.json';assert not out.exists();checks=toy()
 ck='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
 state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');ru=torch.linalg.qr(state['lm_head.weight'].double(),mode='r').R
 L=state['transformer.h.17.mlp.Left.weight'].double();R=state['transformer.h.17.mlp.Right.weight'].double();C=ru@state['transformer.h.17.mlp.Down.weight'].double()
 rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n,m,y=[rows[k].flatten(0,1).double() for k in ['n','m','y']]
 S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double()
 e={k:v.double() for k,v in torch.load(p/'MIDPOINT_GRAPH_PRODUCT_PRUNE_V1.pt',weights_only=True)['products512'].items()}
 A=e['Pn']@e['Tn'];B=e['Pm']@e['Tm'];W=e['output_basis']@e['output_core'];nc=n-n.mean(0);mc=m-m.mean(0)
 Mn=nc.T@nc/len(n);Mm=mc.T@mc/len(m)
 K=(A.T@Mn@A)*(B.T@Mm@B);cross=cross_gram(A,B,L,R,Mn,Mm)@C.T
 scale=K.diag().sqrt().clamp_min(1e-12);Kn=K/scale[:,None]/scale[None,:];prior=scale[:,None]*W.T
 eig,V=torch.linalg.eigh(Kn);rhs=V.T@(cross/scale[:,None]-Kn@prior)
 aa,bb,ac,bc=n@A,m@B,nc@A,mc@B
 nl,nr,ml,mr=n@L.T,n@R.T,m@L.T,m@R.T
 base=(nl*mr+nr*ml)@C.T;checks['native_replay']=float((base-y).norm()/y.norm());assert checks['native_replay']<1e-5
 basepred=(aa*bb-e['product_mean'])@W.T+e['full_mean'];den=((y-y.mean(0))@S).norm()
 records=[]
 for ridge in [None,0.,.001,.01,.1,1.,10.]:
  correction=torch.zeros_like(W.T) if ridge is None else (V@(rhs/(eig.clamp_min(1e-10)+ridge)[:,None]))/scale[:,None]
  adjusted=basepred+(ac*bc)@correction
  stats=dict(ridge=ridge,paired_error=float(((adjusted-base)@S).norm()/den),swaps=[])
  for shift in [1,7,16]:
   ids=torch.arange(len(n)).reshape(rows['n'].shape[:2]).roll(shift,0).flatten();dml,dmr=ml[ids]-ml,mr[ids]-mr
   delta=(nl*dmr+nr*dml)@C.T;interaction=((nl-nl.mean(0))*dmr+(nr-nr.mean(0))*dml)@C.T
   pred=(aa*(bb[ids]-bb))@W.T+(ac*(bc[ids]-bc))@correction
   pint=(ac*(bc[ids]-bc))@(W.T+correction)
   stats['swaps'].append(dict(shift=shift,effect_error=float(((pred-delta)@S).norm()/(delta@S).norm()),interaction_error=float(((pint-interaction)@S).norm()/(interaction@S).norm())))
  records.append(stats)
 result=dict(checks=checks,records=records,seconds=time.perf_counter()-start,scope='Original-weight teacher; exact independent centered calibration-marginal metric, no approximate teacher. Fixed512 dictionary; update centered interaction only, preserve old mean/single-input terms. Recombination diagnostics reuse prior screened shifts; no independent heldout claim. Ridge acts on column-normalized coefficients. No exported native candidate or simplicity claim: centered correction implementation costs require accounting.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
