#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_solve pred_b_response pred_c_coefficient
"""Frozen3686product writer refit with normalized coefficient+response loss.
Fourlambda0/.1/1/10; primary1 fixed. Normalresidual<1e-8; primary response
error<=.8baseline; covariance coefficient error<=1.1baseline.0nativeforwards.
Historical2048states only; directions are centered nativeMLP16 source writes
propagated through recipient RMS derivative. No opened48panel fitting.
Same14,067,072coefficients with exact mean/tangent affine repair.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import torch
 sys.path.insert(0,str(P));from centered_product_response import features,controls
 torch.set_num_threads(2);check=controls()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(controls=check,lambdas=[0,.1,1,10],primary=1,forwards=0)));return
 from normalized_bilinear_response import rms_pair
 torch.set_grad_enabled(False);start=time.monotonic();out=P/'FULL_CHANNEL_RESPONSE_REFIT_V1.json';assert not out.exists()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().double()
 U=w('lm_head.weight');RU=torch.linalg.cholesky(U.T@U).T;del U
 L,R,D=w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.17.mlp.Down.weight')
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);h,z=data['h'].cuda().double(),data['z'].cuda().double();source=w('transformer.h.17.lambdas')[0]*((z@w('transformer.h.16.mlp.Left.weight').T)*(z@w('transformer.h.16.mlp.Right.weight').T))@w('transformer.h.16.mlp.Down.weight').T
 x,v=rms_pair(h,source-source.mean(0),torch.finfo(torch.float32).eps);mu=x.mean(0);dx=x-mu;ev,V=torch.linalg.eigh(dx.T@dx/len(dx));root=(V*ev.clamp_min(1e-8*ev.mean()).sqrt())@V.T
 an=(L@root).norm(dim=1);bn=(R@root).norm(dim=1);a=L/an[:,None];b=R/bn[:,None];A=a@root;B=b@root;C=(RU@D)*an*bn;cscale=C.norm();C/=cscale
 ab=A@B.T;K0=.5*((A@A.T)*(B@B.T)+ab*ab.T);gf=features(dx,v,a,b);K1=gf.T@gf/len(gf);e0=float(((C@K0)*C).sum());e1=float(((C@K1)*C).sum());ids=torch.tensor(next(r['selected_channels'] for r in json.loads((P/'FULL_CHANNEL_DELETION_V1.json').read_text())['records'] if r['geometry']=='activation_covariance' and r['policy']=='conditional' and r['width']==3686),device='cuda');records=[]
 for lam in (0.,.1,1.,10.):
  K=K0/e0+lam*K1/e1;rhs=(C@K)[:,ids];ks=K[ids][:,ids];writer=torch.cholesky_solve(rhs.T,torch.linalg.cholesky(ks)).T;normal=float((writer@ks-rhs).norm()/rhs.norm());errors=[]
  for gram,energy in [(K0,e0),(K1,e1)]:
   cross=(C@gram)[:,ids];errors.append(max(0,(energy-2*float((writer*cross).sum())+float(((writer@gram[ids][:,ids])*writer).sum()))/energy)**.5)
  records.append(dict(response_weight=lam,coefficient_error=errors[0],response_error=errors[1],normal_residual=normal))
  if lam==1.:
   ar,br=a[ids],b[ids];wr=torch.linalg.solve(RU,writer*cscale);linear=D@((mu@R.T)[:,None]*L+(mu@L.T)[:,None]*R)-wr@((mu@br.T)[:,None]*ar+(mu@ar.T)[:,None]*br);bias=D@((L@mu)*(R@mu))-wr@((ar@mu)*(br@mu))-linear@mu
   program=dict(a=ar.cpu().float(),b=br.cpu().float(),writer=wr.cpu().float(),linear=linear.cpu().float(),bias=bias.cpu().float());torch.save(program,P/'FULL_CHANNEL_RESPONSE_PROGRAM_V1.pt')
 baseline,primary=records[0],records[2];pred=dict(pred_a_solve=all(r['normal_residual']<1e-8 for r in records),pred_b_response=primary['response_error']<=.8*baseline['response_error'],pred_c_coefficient=primary['coefficient_error']<=1.1*baseline['coefficient_error'])
 result=dict(predictions=pred,records=records,controls=check,primary_weight=1,stored_floats=sum(t.numel() for t in program.values()),seconds=time.monotonic()-start,scope='Exact conditional writer solves; same retained products and mean/tangent affine repair. Historical normalized input/source directions; measures pre-final-normalization unembedding metric, not actual endpoint JVP. This is an explicit data-informed response constraint. Primary fixed before outcomes; no adoption until function, intervention and continuation behavior tests.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
