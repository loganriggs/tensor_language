#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_values pred_c_derivatives
"""Joint quartic readout, fixedlambda1primary,0/.01/.1/10secondary.
Primarypanel1values<=.8*.1504982867andcal<=5%; deriv<=.8*.3560211862.
Integritysolve<1e-8/export<1e-4/lambda0replaywithin1e-5absolute.
Same656products903168coefficients; twoopenedpanels, noadoptionclaim.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,panels=2,anchors=16,lambdas=[0,.01,.1,1,10],features=528)));return
 import torch
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from quartic_reader_span import jacobian
 from quartic_bank_derivative import feature_jacobian
 from empirical_quartic_dictionary import features,evaluate
 from quartic_joint_readout import solve
 start=time.monotonic();out=P/'QUARTIC_JOINT_READOUT_V1.json';assert not out.exists()
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);scale=data['teacher_scale'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
 old=torch.load(P/'WIDE_NATIVE_QUARTIC_32_INHERITED_LONG_V1.pt',weights_only=True);U,V=old['U'].cuda().double(),old['V'].cuda().double();L=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True)['metric_transforms']['centered_floor01'].cuda().double();xs=[p['rows'].cuda().double() for p in panels];ys=[y.cuda().double() for y in data['targets']];phi=[features(x,U,V) for x in xs];stats=[]
 for x in xs:
  dg=torch.zeros(528,528,device='cuda',dtype=torch.float64);dc=torch.zeros(1152,528,device='cuda',dtype=torch.float64);energy=torch.zeros((),device='cuda',dtype=torch.float64)
  for index in torch.linspace(0,len(x)-1,16).long():
   z=x[index];j=jacobian(teacher,z.float()).double()@L;p=feature_jacobian(U,V,z)@L;dg.add_(p@p.T);dc.add_(j@p.T);energy.add_(j.square().sum())
  stats.append((dg,dc,energy))
 records=[]
 for weight in [0.,.01,.1,1.,10.]:
  writer,info=solve(phi[0],ys[0],*stats[0],weight);values=[float((p@writer.T-y).norm()/y.norm()) for p,y in zip(phi,ys)];derivatives=[]
  for dg,dc,energy in stats:
   loss=(((writer@dg)*writer).sum()-2*(writer*dc).sum()+energy)/energy;assert float(loss)>-1e-10;derivatives.append(float(loss.clamp_min(0).sqrt()))
  archive=dict(U=U.cpu().float(),V=V.cpu().float(),C=torch.linalg.solve(ru.double(),writer*scale).cpu().float());physical={k:v.cuda() for k,v in archive.items()};actual=evaluate(physical,xs[0][:128].float()).double()@ru.double().T/scale;reference=phi[0][:128]@writer.T;drift=float((actual-reference).norm()/reference.norm());tag=format(weight,'g').replace('.','p');path=P/f'QUARTIC_JOINT_READOUT_LAMBDA_{tag}_V1.pt';torch.save(archive,path)
  row=dict(weight=weight,value_errors=values,derivative_errors=derivatives,normal_residual=info['normal_residual'],export_replay=drift,stored_coefficients=sum(z.numel() for z in archive.values()),program_sha256=hashlib.sha256(path.read_bytes()).hexdigest());records.append(row);print(json.dumps(row),flush=True)
 primary=next(r for r in records if r['weight']==1);baseline=records[0];replay=max(abs(a-b) for a,b in zip(baseline['value_errors'],[.02158625648,.1504982867]));pred=dict(pred_a_integrity=replay<1e-5 and all(r['normal_residual']<1e-8 and r['export_replay']<1e-4 for r in records),pred_b_values=primary['value_errors'][0]<=.05 and primary['value_errors'][1]<=.8*.1504982867,pred_c_derivatives=primary['derivative_errors'][1]<=.8*.3560211862)
 result=dict(predictions=pred,records=records,baseline_replay_absolute_error=replay,seconds=time.monotonic()-start,scope='Fixed learned dictionary, calibrationvalues and16calibrationcenteredderivativeanchors. Secondaryweights notselection; bothpanelsalreadyopened. No featurechange, freshOOD, semantic ornativeinterventionadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
