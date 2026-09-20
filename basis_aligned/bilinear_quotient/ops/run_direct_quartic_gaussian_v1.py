#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_mean pred_b_radial pred_c_residual_mean
"""Exact native quartic Gaussian mean;4096 independent function queries.
pred_a_mean MC mean relative discrepancy<.05; pred_b_radial radial error<.9;
pred_c_residual_mean some student mean mismatch explains>=.5 residual energy.
Null: trace/radial structure does not explain apparent Gaussian fit behavior.
No native forwards, no fitting. Output frame and baseline vector counted separately.
"""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(function_queries=4096,batch_size=256,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 sys.path.insert(0,str(P));from quartic_gaussian import mean
 torch.set_num_threads(4);start=time.perf_counter();out=P/'NATIVE_QUARTIC_GAUSSIAN_V1.json';assert not out.exists();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().double()
 def execute(p,x):
  C,L,R,D,A,B=p;q=((x@A.T)*(x@B.T))@D.T;return ((q@L.T)*(q@R.T))@C.T
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight'),w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];d=1152;mu=mean(*teacher);torch.manual_seed(1633);x=torch.randn(4096,d,device='cuda',dtype=torch.float64);y=torch.cat([execute(teacher,z) for z in x.split(256)]);energy=y.square().sum(-1);total=float(energy.mean());se=float(energy.std()/math.sqrt(len(energy)));mean_error=float((y.mean(0)-mu).norm()/mu.norm());radial=x.square().sum(-1).square()[:,None]*mu/(d*(d+2));raderr=float((y-radial).norm()/y.norm());constant=float((y-mu).norm()/y.norm());frob=json.load(open(P/'NATIVE_QUARTIC_QUERY_V1.json'))['stratified_energy'];records=[];missing=[]
  for version in ['V1','V2']:
   path=P/f'NATIVE_STOCHASTIC_QUARTIC_{version}.pt'
   if not path.exists():missing.append(version);continue
   artifact=torch.load(path,weights_only=False);scale=artifact['teacher_scale'].cuda().double()
   for key,factors in artifact['students'].items():
    C,L,R,A,B=[v.cuda().double() for v in factors];p=[C*scale,L,R,torch.eye(A.shape[0],device='cuda',dtype=torch.float64),A,B];student_mean=mean(*p);pred=torch.cat([execute(p,z) for z in x.split(256)]);residual_energy=float((pred-y).square().sum(-1).mean());mean_delta=float((student_mean-mu).square().sum());records.append(dict(version=version,student=key,gaussian_function_error=math.sqrt(residual_energy/total),exact_mean_error_relative_to_teacher_mean=float((student_mean-mu).norm()/mu.norm()),mean_residual_energy_fraction_estimate=mean_delta/residual_energy))
  result=dict(plan=PLAN,teacher_exact_mean_energy=float(mu.square().sum()),teacher_gaussian_energy_estimate=total,teacher_gaussian_energy_se=se,monte_carlo_mean_relative_error=mean_error,constant_baseline_error=constant,radial_quartic_baseline_error=raderr,estimated_teacher_chaos_fractions=dict(degree0=float(mu.square().sum())/total,degree4=24*frob/total,degree2=1-(float(mu.square().sum())+24*frob)/total),students=records,missing_artifacts=missing,seconds=time.perf_counter()-start,predictions=dict(pred_a_mean=mean_error<.05,pred_b_radial=raderr<.9,pred_c_residual_mean=any(r['mean_residual_energy_fraction_estimate']>=.5 for r in records)),scope='Exact mean plus sampled Gaussian energy on new inputs; Hermite fractions involving energy estimates have sampling uncertainty. No coefficient fitting or native behavioral metric.')
  out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
