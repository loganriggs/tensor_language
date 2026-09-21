#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_finite pred_b_coefficient pred_c_response
"""Full1152output quadratic learned-direction pilot, same3686products.
Two starts existing response program/random; Adam.05,100steps cosine.
Inputs whitened by historical covariance; exact conditional writer solves.
Pred finite/FP32export drift<1e-4; selected coefficient error<=.9warm initial;
selected response error<=1.1warm initial. Selection coefficient+response loss
only, no evaluation panel. Same14,067,072coefficients; no semantic adoption.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,steps=100,starts=['inherited','random'],products=3686)));return
 import torch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from profiled_learned_quadratic_factors import fit,readout
 from learned_quadratic_factors import Objective
 from normalized_bilinear_response import rms_pair
 start=time.monotonic();out=P/'LEARNED_FULL_QUADRATIC_V1.json';assert not out.exists()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().double()
 U=w('lm_head.weight');RU=torch.linalg.cholesky(U.T@U).T;del U
 L,R,D=w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.17.mlp.Down.weight')
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);h,z=data['h'].cuda().double(),data['z'].cuda().double();source=w('transformer.h.17.lambdas')[0]*((z@w('transformer.h.16.mlp.Left.weight').T)*(z@w('transformer.h.16.mlp.Right.weight').T))@w('transformer.h.16.mlp.Down.weight').T
 x,v=rms_pair(h,source-source.mean(0),torch.finfo(torch.float32).eps);mu=x.mean(0);dx=x-mu;ev,V=torch.linalg.eigh(dx.T@dx/len(dx));ev=ev.clamp_min(1e-8*ev.mean());root=(V*ev.sqrt())@V.T;inverse=(V*ev.rsqrt())@V.T
 A,B=L@root,R@root;an,bn=A.norm(dim=1),B.norm(dim=1);A/=an[:,None];B/=bn[:,None];C=(RU@D)*an*bn;cscale=C.norm();C/=cscale;xx,vv=dx@inverse,v@inverse;teacher=(C,A,B);objective=Objective(*teacher,xx,vv)
 old={k:v.cuda().double() for k,v in torch.load(P/'FULL_CHANNEL_RESPONSE_PROGRAM_V1.pt',weights_only=True).items()};aw,bw=old['a']@root,old['b']@root;aw/=aw.norm(dim=1,keepdim=True);bw/=bw.norm(dim=1,keepdim=True)
 with torch.no_grad():
  cw=readout(objective,aw,bw);before=objective.losses(cw,aw,bw);initial=dict(coefficient_error=float(before[0].clamp_min(0).sqrt()),response_error=float(before[1].clamp_min(0).sqrt()))
 records=[];best=None
 for index,name in enumerate(('inherited','random')):
  init=(cw,aw*math.sqrt(1152),bw*math.sqrt(1152)) if name=='inherited' else None
  result,factors=fit(teacher,3686,'adam',.05,980+index,100,xx,vv,init);result['start']=name;records.append(result);score=result['coefficient_error']**2+result['response_error']**2
  if best is None or score<best[0]:best=(score,name,factors)
  print(json.dumps(result),flush=True)
 _,selected,(writer,aa,bb)=best;ar,br=aa@inverse,bb@inverse;wr=torch.linalg.solve(RU,writer*cscale);linear=D@((mu@R.T)[:,None]*L+(mu@L.T)[:,None]*R)-wr@((mu@br.T)[:,None]*ar+(mu@ar.T)[:,None]*br);bias=D@((L@mu)*(R@mu))-wr@((ar@mu)*(br@mu))-linear@mu
 program=dict(a=ar.cpu().float(),b=br.cpu().float(),writer=wr.cpu().float(),linear=linear.cpu().float(),bias=bias.cpu().float());path=P/'LEARNED_FULL_QUADRATIC_PROGRAM_V1.pt';torch.save(program,path)
 with torch.no_grad():
  q={k:t.cuda() for k,t in program.items()};value=((x@ar.T)*(x@br.T))@wr.T+x@linear.T+bias;xf=x.float();actual=((xf@q['a'].T)*(xf@q['b'].T))@q['writer'].T+xf@q['linear'].T+q['bias'];drift=float((actual.double()-value).norm()/value.norm())
 chosen=next(r for r in records if r['start']==selected);pred=dict(pred_a_finite=all(math.isfinite(r[k]) for r in records for k in ('coefficient_error','response_error')) and drift<1e-4,pred_b_coefficient=chosen['coefficient_error']<=.9*initial['coefficient_error'],pred_c_response=chosen['response_error']<=1.1*initial['response_error'])
 result=dict(predictions=pred,records=records,initial=initial,selected=selected,fp32_export_drift=drift,stored_floats=sum(t.numel() for t in program.values()),program_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.monotonic()-start,scope='Full original lastMLP polynomial in historical covariance geometry, both input factors learned at fixed width. Conditional output weights solved every step. Historical source-response direction constraint retained;100steps is a pilot, not convergence proof. Inherited reader coordinates scaled sqrt1152 to match random initialization row scales. No heldout data used, no new semantic unit or general graph search claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
