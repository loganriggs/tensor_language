#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_finite pred_b_mixed_guard pred_c_optimization
"""Controlled metric comparison at fixed3686product budget,100stepsAdam.05.
Mixed: normalized cov+response+.1isotropic; isotropic: weight-only loss.
Same inherited readers and covariance parameter coordinates; not end-to-end
weight-only discovery. Best fitting loss checkpoint only; no native outcomes.
Pred FP32drift<1e-4; mixediso<=.4023864 and cov<=1.1*.1116528;
both objectives improve>=5%relative to their initial exact output solve.
Each export14,067,072coefficients including data-informed affine repair.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,steps=100,objectives=['mixed','isotropic'],products=3686)));return
 import torch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from multigeometry_quadratic import Objective
 from normalized_bilinear_response import rms_pair
 start=time.monotonic();out=P/'FULL_QUADRATIC_MULTIGEOMETRY_FIT_V1.json';assert not out.exists()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().double()
 U=w('lm_head.weight');RU=torch.linalg.cholesky(U.T@U).T;del U
 L,R,D=w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.17.mlp.Down.weight')
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);h,z=data['h'].cuda().double(),data['z'].cuda().double();source=w('transformer.h.17.lambdas')[0]*((z@w('transformer.h.16.mlp.Left.weight').T)*(z@w('transformer.h.16.mlp.Right.weight').T))@w('transformer.h.16.mlp.Down.weight').T
 x,v=rms_pair(h,source-source.mean(0),torch.finfo(torch.float32).eps);mu=x.mean(0);dx=x-mu;ev,V=torch.linalg.eigh(dx.T@dx/len(dx));ev=ev.clamp_min(1e-8*ev.mean());root=(V*ev.sqrt())@V.T;inverse=(V*ev.rsqrt())@V.T
 A,B=L@root,R@root;an,bn=A.norm(dim=1),B.norm(dim=1);A/=an[:,None];B/=bn[:,None];C=(RU@D)*an*bn;cscale=C.norm();C/=cscale;xx,vv=dx@inverse,v@inverse;teacher=(C,A,B)

 old={k:v.cuda().double() for k,v in torch.load(P/'FULL_CHANNEL_RESPONSE_PROGRAM_V1.pt',weights_only=True).items()};aw,bw=old['a']@root,old['b']@root;aw/=aw.norm(dim=1,keepdim=True);bw/=bw.norm(dim=1,keepdim=True);I=torch.eye(1152,device='cuda',dtype=torch.float64);audit=Objective(*teacher,[(1.,I),(1.,inverse)],xx,vv,1.);records=[]
 for name,geometries,response_weight in [('mixed',[(1.,I),(.1,inverse)],1.),('isotropic',[(1.,inverse)],0.)]:
  objective=Objective(*teacher,geometries,xx,vv,response_weight);params=[torch.nn.Parameter(t.clone()*math.sqrt(1152)) for t in (aw,bw)];opt=torch.optim.Adam(params,lr=.05);best=None;history=[];arm_start=time.monotonic()
  for step in range(101):
   aa,bb=[t/t.norm(dim=1,keepdim=True).clamp_min(1e-12) for t in params]
   with torch.no_grad():writer=objective.readout(aa,bb)
   loss,_,_=objective.losses(writer,aa,bb);score=float(loss.detach());assert math.isfinite(score)
   if step==0:initial_loss=score
   if best is None or score<best[0]:best=(score,step,writer.detach().clone(),aa.detach().clone(),bb.detach().clone())
   if step%25==0:history.append(dict(step=step,loss=score));print(json.dumps(dict(arm=name,step=step,loss=score)),flush=True)
   if step==100:break
   opt.zero_grad();loss.backward();opt.step()
   for group in opt.param_groups:group['lr']=.05*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/100)))
  loss,selected_step,writer,aa,bb=best
  with torch.no_grad():
   _,metric,resp=audit.losses(writer,aa,bb);ar,br=aa@inverse,bb@inverse;wr=torch.linalg.solve(RU,writer*cscale);linear=D@((mu@R.T)[:,None]*L+(mu@L.T)[:,None]*R)-wr@((mu@br.T)[:,None]*ar+(mu@ar.T)[:,None]*br);bias=D@((L@mu)*(R@mu))-wr@((ar@mu)*(br@mu))-linear@mu
   program=dict(a=ar.cpu().float(),b=br.cpu().float(),writer=wr.cpu().float(),linear=linear.cpu().float(),bias=bias.cpu().float());path=P/f'FULL_QUADRATIC_{name.upper()}_PROGRAM_V1.pt';torch.save(program,path);q={k:t.cuda() for k,t in program.items()};value=((x@ar.T)*(x@br.T))@wr.T+x@linear.T+bias;xf=x.float();actual=((xf@q['a'].T)*(xf@q['b'].T))@q['writer'].T+xf@q['linear'].T+q['bias'];drift=float((actual.double()-value).norm()/value.norm())
   record=dict(arm=name,initial_loss=initial_loss,selected_loss=loss,selected_step=selected_step,covariance_error=float(metric[0].clamp_min(0).sqrt()),isotropic_error=float(metric[1].clamp_min(0).sqrt()),response_error=float(resp.clamp_min(0).sqrt()),fp32_export_drift=drift,stored_floats=sum(t.numel() for t in program.values()),program_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),history=history,seconds=time.monotonic()-arm_start);records.append(record);print(json.dumps(record),flush=True)
  del objective,params,opt,best,writer,aa,bb
 mixed=records[0];pred=dict(pred_a_finite=all(math.isfinite(r['selected_loss']) and r['fp32_export_drift']<1e-4 for r in records),pred_b_mixed_guard=mixed['isotropic_error']<=.4023864 and mixed['covariance_error']<=1.1*.1116528,pred_c_optimization=all(r['selected_loss']<=.95*r['initial_loss'] for r in records))
 result=dict(predictions=pred,records=records,seconds=time.monotonic()-start,scope='Two fitting objectives, identical data-informed inherited reader start and covariance parameter coordinates. Isotropic loss itself uses only weight tensor; affine repair/calibration coordinates remain data-informed. Best objective checkpoint within100steps, no validation-based selection. Full3686product quadratic target, not arbitrary DAG/HT search; native behavior pending.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
