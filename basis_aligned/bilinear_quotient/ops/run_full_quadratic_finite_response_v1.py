#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_finite pred_b_guard pred_c_learning
"""Historical finite source-swap response; exact endpointRMS and polarization.
Fixed and100step learnedreaders, cov+.1iso+finite-response loss, no sweep.
Both same3686products14067072coefficients. FP32drift<1e-4; cov<=.12282,
iso<=.4023864; learned objective improves>=5%over exact fixedreader solve.
Donors roll257of2048historicalstates; no opened nativepanel fitting.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,steps=100,objectives=['finite_fixed','finite_learned'],products=3686)));return
 import torch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from multigeometry_quadratic import Objective
 from normalized_bilinear_response import rms_pair
 from finite_source_response import normalized_secant
 start=time.monotonic();out=P/'FULL_QUADRATIC_FINITE_RESPONSE_V1.json';assert not out.exists()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().double()
 U=w('lm_head.weight');RU=torch.linalg.cholesky(U.T@U).T;del U
 L,R,D=w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.17.mlp.Down.weight')
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);h,z=data['h'].cuda().double(),data['z'].cuda().double();source=w('transformer.h.17.lambdas')[0]*((z@w('transformer.h.16.mlp.Left.weight').T)*(z@w('transformer.h.16.mlp.Right.weight').T))@w('transformer.h.16.mlp.Down.weight').T
 x,v=rms_pair(h,source-source.mean(0),torch.finfo(torch.float32).eps);mu=x.mean(0);dx=x-mu;ev,V=torch.linalg.eigh(dx.T@dx/len(dx));ev=ev.clamp_min(1e-8*ev.mean());root=(V*ev.sqrt())@V.T;inverse=(V*ev.rsqrt())@V.T
 A,B=L@root,R@root;an,bn=A.norm(dim=1),B.norm(dim=1);A/=an[:,None];B/=bn[:,None];C=(RU@D)*an*bn;cscale=C.norm();C/=cscale;midpoint,step=normalized_secant(h,source.roll(257,0)-source,mu,torch.finfo(torch.float32).eps);xx,vv=midpoint@inverse,step@inverse;teacher=(C,A,B)

 old={k:v.cuda().double() for k,v in torch.load(P/'FULL_CHANNEL_RESPONSE_PROGRAM_V1.pt',weights_only=True).items()};aw,bw=old['a']@root,old['b']@root;aw/=aw.norm(dim=1,keepdim=True);bw/=bw.norm(dim=1,keepdim=True);I=torch.eye(1152,device='cuda',dtype=torch.float64);audit=Objective(*teacher,[(1.,I),(1.,inverse)],xx,vv,1.);records=[]
 for name,steps in [('finite_fixed',0),('finite_learned',100)]:
  geometries,response_weight=[(1.,I),(.1,inverse)],1.
  objective=Objective(*teacher,geometries,xx,vv,response_weight);params=[torch.nn.Parameter(t.clone()*math.sqrt(1152)) for t in (aw,bw)];opt=torch.optim.Adam(params,lr=.05);best=None;history=[];arm_start=time.monotonic()
  for step in range(steps+1):
   aa,bb=[t/t.norm(dim=1,keepdim=True).clamp_min(1e-12) for t in params]
   with torch.no_grad():writer=objective.readout(aa,bb)
   loss,_,_=objective.losses(writer,aa,bb);score=float(loss.detach());assert math.isfinite(score)
   if step==0:initial_loss=score
   if best is None or score<best[0]:best=(score,step,writer.detach().clone(),aa.detach().clone(),bb.detach().clone())
   if step%25==0:history.append(dict(step=step,loss=score));print(json.dumps(dict(arm=name,step=step,loss=score)),flush=True)
   if step==steps:break
   opt.zero_grad();loss.backward();opt.step()
   for group in opt.param_groups:group['lr']=.05*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/100)))
  loss,selected_step,writer,aa,bb=best
  with torch.no_grad():
   _,metric,resp=audit.losses(writer,aa,bb);ar,br=aa@inverse,bb@inverse;wr=torch.linalg.solve(RU,writer*cscale);linear=D@((mu@R.T)[:,None]*L+(mu@L.T)[:,None]*R)-wr@((mu@br.T)[:,None]*ar+(mu@ar.T)[:,None]*br);bias=D@((L@mu)*(R@mu))-wr@((ar@mu)*(br@mu))-linear@mu
   program=dict(a=ar.cpu().float(),b=br.cpu().float(),writer=wr.cpu().float(),linear=linear.cpu().float(),bias=bias.cpu().float());path=P/f'FULL_QUADRATIC_{name.upper()}_PROGRAM_V1.pt';torch.save(program,path);q={k:t.cuda() for k,t in program.items()};value=((x@ar.T)*(x@br.T))@wr.T+x@linear.T+bias;xf=x.float();actual=((xf@q['a'].T)*(xf@q['b'].T))@q['writer'].T+xf@q['linear'].T+q['bias'];drift=float((actual.double()-value).norm()/value.norm())
   record=dict(arm=name,initial_loss=initial_loss,selected_loss=loss,selected_step=selected_step,covariance_error=float(metric[0].clamp_min(0).sqrt()),isotropic_error=float(metric[1].clamp_min(0).sqrt()),response_error=float(resp.clamp_min(0).sqrt()),fp32_export_drift=drift,stored_floats=sum(t.numel() for t in program.values()),program_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),history=history,seconds=time.monotonic()-arm_start);records.append(record);print(json.dumps(record),flush=True)
  del objective,params,opt,best,writer,aa,bb
 fixed,learned=records
 pred=dict(pred_a_finite=all(math.isfinite(r['selected_loss']) and r['fp32_export_drift']<1e-4 for r in records),pred_b_guard=all(r['isotropic_error']<=.4023864 and r['covariance_error']<=.12282 for r in records),pred_c_learning=learned['selected_loss']<=.95*fixed['selected_loss'])
 result=dict(predictions=pred,records=records,seconds=time.monotonic()-start,scope='Historical source roll257 finite swaps with explicit input RMS; exact centered quadratic difference. Fixed versus learned readers, same mixed objective and affine repair. Data-informed fitting, no native outcome selection; final normalization/softcap fidelity pending. Not arbitrary graph search.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
