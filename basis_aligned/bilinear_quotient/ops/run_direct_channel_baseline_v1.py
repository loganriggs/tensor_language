#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_refit pred_b_metric pred_c_optimization_gap
"""Width-matched teacher channel baseline, exact root refit.
pred_a_refit: no relative squared objective worsening >1e-8.
pred_b_metric: Gram and implicit relative squared error agree<1e-8.
pred_c_optimization_gap: width1024 Frobenius refit below0.9548 error.
Null: channel subset cannot beat random-start fit. 18 solves, no native forwards.
Parameter price3*1152*width plus QR frames; no causal adoption.
"""
import os,sys,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(widths=[128,512,1024],metrics=['gaussian','frobenius'],policies=['energy','random0','random1'],native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 sys.path.insert(0,str(P));from implicit_quadratic import inner
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();out=P/'NATIVE_CHANNEL_BASELINE_V1.json';assert not out.exists()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().double()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));E=torch.cat([torch.eye(1152,device='cuda',dtype=torch.float64),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.17.attn.c_proj.weight')],1);_,re=torch.linalg.qr(E.T)
  c=ru@w('transformer.h.17.mlp.Down.weight');a=w('transformer.h.17.mlp.Left.weight')@re.T;b=w('transformer.h.17.mlp.Right.weight')@re.T
  an=a.norm(dim=1);bn=b.norm(dim=1);a/=an[:,None];b/=bn[:,None];c=c*an*bn
  aa=a@a.T;bb=b@b.T;ab=a@b.T;Kf=.5*(aa*bb+ab*ab.T);trace=(a*b).sum(-1);Kg=2*Kf+trace[:,None]*trace[None,:];del aa,bb,ab
  kernels={'gaussian':Kg,'frobenius':Kf};energies={m:float(((c@k)*c).sum()) for m,k in kernels.items()};records=[]
  for metric,K in kernels.items():
   score=c.square().sum(0)*K.diag()
   for policy in PLAN['policies']:
    if policy=='energy':order=score.argsort(descending=True)
    else:torch.manual_seed(int(policy[-1]));order=torch.randperm(len(a),device='cuda')
    for width in PLAN['widths']:
     ids=order[:width];ks=K[ids][:,ids];cross=c@K[:,ids];values,V=torch.linalg.eigh(ks);keep=values>values[-1]*1e-10;d=(cross@V[:,keep]/values[keep])@V[:,keep].T;baseline=c[:,ids]
     gram_error=(energies[metric]+float(((d@ks)*d).sum())-2*float((cross*d).sum()))/energies[metric]
     before=(energies[metric]+float(((baseline@ks)*baseline).sum())-2*float((cross*baseline).sum()))/energies[metric]
     errors={}
     for name,kk in kernels.items():errors[name]=math.sqrt(max(0,(energies[name]+float(((d@kk[ids][:,ids])*d).sum())-2*float(((c@kk[:,ids])*d).sum()))/energies[name]))
     implicit=(energies[metric]+float(inner(d,a[ids],b[ids],d,a[ids],b[ids],metric=='gaussian'))-2*float(inner(c,a,b,d,a[ids],b[ids],metric=='gaussian')))/energies[metric]
     normal=float((d@ks-cross).norm()/cross.norm())
     row=dict(metric=metric,policy=policy,width=width,relative_errors=errors,unchanged_relative_error=math.sqrt(max(0,before)),gram_relative_squared_error=gram_error,implicit_relative_squared_error=implicit,normal_equation_residual=normal,minimum_eigenvalue=float(values[0]),maximum_eigenvalue=float(values[-1]),retained_rank=int(keep.sum()),parameter_values=3*1152*width,selected_channels=ids.cpu().tolist());records.append(row);print(metric,policy,width,errors,flush=True)
     out.write_text(json.dumps(dict(plan=PLAN,records=records,seconds=time.perf_counter()-start),indent=2)+'\n')
  result=dict(plan=PLAN,records=records,seconds=time.perf_counter()-start,predictions=dict(pred_a_refit=all(r['gram_relative_squared_error']<=r['unchanged_relative_error']**2+1e-8 for r in records),pred_b_metric=all(abs(r['gram_relative_squared_error']-r['implicit_relative_squared_error'])<1e-8 for r in records),pred_c_optimization_gap=any(r['width']==1024 and r['metric']=='frobenius' and r['relative_errors']['frobenius']<.9548 for r in records)),scope='Teacher-channel selection and globally optimal output refit; width-matched baseline, not random initialization, no circuit identification')
  out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
