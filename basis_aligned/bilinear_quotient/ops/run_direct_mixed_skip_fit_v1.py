#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact pred_b_gaussian pred_c_export
"""Exact Gaussian DAG skip fit; MIXED_SKIP_PLAN_V1.md.
pred_a_exact: full solve<1e-8, positiveG, rank2 analytic gain>=quadratic rank2.
pred_b_gaussian: rank2 independent Gaussian error<=quadratic-only rank2.
pred_c_export: FP32 export relative replay<1e-5 and literal counts correct.
Native transfer prediction is evaluated by subsequent registered screen, not here.
Null: reusing quadratic nodes adds little or fails domain transfer.
Price:19632+1170r coefficients; unchanged10products.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(ranks=[1,2,4,6],primary=2,probe_pairs=4096,probe_seed=260930,products=10)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P))
 from gaussian_mixed_skip import moments
 from gaussian_linear_projection import directions
 from gaussian_quartic_mean import quadratic_moments
 from frozen_program_evaluation import load_teacher,native,quartic
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'MIXED_SKIP_FIT_V1.json';assert not out.exists();start=time.perf_counter()
 base=torch.load(P/'FUSED_ROOT_PROGRAM_V1.pt',weights_only=True)['programs'][4];s={k:v.cuda().double() for k,v in base.items()}
 source=torch.load(P/'NATIVE_QUARTIC_MEAN_V1.pt',weights_only=True);scale=source['teacher_scale'];teacher=load_teacher('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',scale);teacher=[t.double() for t in teacher]
 panel=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0];mu=panel['mean'].cuda().double();M=panel['covariance'].cuda().double();M=(M+M.T)/2;A,B=s['A'],s['B'];meanp,G=moments(A,B,mu,M)
 print('native linear cross moments started',flush=True);H=torch.cat([A,B]);Klinear=directions(teacher,mu,M,H@M);print('native linear cross moments finished',flush=True)
 student_teacher=[s['output_writer'],s['root_left'],s['root_right'],torch.eye(6,device='cuda',dtype=torch.float64),A,B]
 Klinear-=directions(student_teacher,mu,M,H@M);K=torch.cat([Klinear,torch.load(P/'QUADRATIC_SKIP_PROGRAM_V1.pt',weights_only=True)['cross_covariance'].cuda().double()],1);e,v=torch.linalg.eigh((G+G.T)/2);assert float(e.min())>0;invhalf=(v/e.sqrt())@v.T;u,sv,vh=torch.linalg.svd(K@invhalf,full_matrices=False);W=K@torch.linalg.inv(G);solve=float((W@G-K).norm()/K.norm());exports={};records=[];models={}
 for rank in PLAN['ranks']:
  reader=vh[:rank]@invhalf;writer=u[:,:rank]*sv[:rank];full=writer@reader;model={**s,'constant':s['constant']-full@meanp}
  model.update(skip_writer=writer,skip_reader=reader)
  gain=float(2*(full*K).sum()-((full@G)*full).sum());archive={k:t.float().cpu() for k,t in model.items()};exports[rank]=archive;models[rank]=model;records.append(dict(rank=rank,coefficients=sum(t.numel() for t in archive.values()),products=10,analytic_mse_reduction=gain))
 ev,vec=torch.linalg.eigh(M);L=vec*ev.clamp_min(0).sqrt();rng=torch.Generator(device='cuda').manual_seed(PLAN['probe_seed']);base_mse=0.;mse={r:0. for r in PLAN['ranks']};energy=0.;replays=[]
 for offset in range(0,2*PLAN['probe_pairs'],128):
  x=mu+torch.randn(128,len(mu),device='cuda',dtype=torch.float64,generator=rng)@L.T;y=native(teacher,x);old=quartic(s,x);base_mse+=float((old-y).square().sum());energy+=float(y.square().sum())
  for rank,model in models.items():
   prediction=quartic(model,x);mse[rank]+=float((prediction-y).square().sum())
   if offset==0:
    loaded={k:v.cuda().double() for k,v in exports[rank].items()};replays.append(float((quartic(loaded,x)-prediction).norm()/prediction.norm()))
 panels=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels']
 for row in records:
  rank=row['rank'];row.update(gaussian_relative_error=(mse[rank]/energy)**.5,gaussian_mse_reduction_fraction=1-mse[rank]/base_mse,reused_fineweb=[])
  for panel in panels:
   x=panel['rows'].cuda().double();y=panel['targets'].cuda().double();pred=quartic(models[rank],x);row['reused_fineweb'].append(dict(context=panel['context'],relative_error=float((pred-y).norm()/y.norm())))
 oldfit=json.loads((P/'QUADRATIC_SKIP_FIT_V1.json').read_text());oldrank2=next(r for r in oldfit['records'] if r['rank']==2);newrank2=next(r for r in records if r['rank']==2)
 pred=dict(pred_a_exact=solve<1e-8 and newrank2['analytic_mse_reduction']>=oldrank2['analytic_mse_reduction']*(1-1e-8),pred_b_gaussian=newrank2['gaussian_relative_error']<=oldrank2['gaussian_relative_error'],pred_c_export=max(replays)<1e-5 and all(r['coefficients']==19632+1170*r['rank'] for r in records))
 guard_torch_save(dict(programs=exports,teacher_scale=scale,primitive_mean=meanp.cpu(),cross_covariance=K.cpu(),primitive_covariance=G.cpu()),str(P/'MIXED_SKIP_PROGRAM_V1.pt'))
 result=dict(plan=PLAN,records=records,predictions=pred,covariance_condition=float(e.max()/e.min()),solve_residual=solve,fp32_export_replay=max(replays),base_gaussian_relative_error=(base_mse/energy)**.5,seconds=time.perf_counter()-start,scope='Exact Gaussian fixed-feature readout fit; ten product nodes reused with linear and quadratic output skip. Native transfer pending; text only diagnostic.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
