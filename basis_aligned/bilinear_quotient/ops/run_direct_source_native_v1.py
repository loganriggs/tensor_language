#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_fidelity pred_c_behavior
"""Frozen upstream extraction: original MLP16 quadratic readers -> continuation feature.
48 captures:32 fresh FW,16 reused stdlib. Arms linear/isotropic16/covariance16/64.
pred_a_instrument exact source Q replay<1e-4; pred_b_fidelity covariance16 native
removal effect error<=.15 all/continuation/spaced cohorts both domains;
pred_c_behavior absolute mean CE difference from exact rank1<=.02 every cohort.
Null: calibration source compression does not preserve native feature effects.
Price: rank16 two quadratic readers39200 weights,32 squares, shared1152 mean;
plus native h-reader, explicit recipient RMS, final product and writer.
This is interface extraction, not a standalone replacement of the native model.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def source_reads(z,e):
 d=z-e['mu'];result=[]
 for k in ['a','b']:
  result.append(e[k+'_constant']+d@e[k+'_linear']+(((d@e[k+'_reader']).square()-e[k+'_quadratic_mean'])*e[k+'_eigenvalues']).sum(-1))
 import torch
 return torch.stack(result,-1)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  import torch
  torch.set_num_threads(2)
  v=torch.load(P/'MIDPOINT_SOURCE_COVARIANCE_PROGRAMS_V1.pt',weights_only=True)
  for key in ['isotropic_0','isotropic_16','covariance_16','covariance_64']:
   assert source_reads(torch.zeros(3,1152,dtype=torch.float64),v[key]).shape==(3,2)
  print(json.dumps(dict(forwards=48,context=256,fit=False,shape_smoke='PASS')));return
 import torch
 import torch.nn.functional as F
 import tiktoken
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from native_feature_capture import capture
 from token_boundary_conditions import annotate
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 out=P/'MIDPOINT_SOURCE_NATIVE_V1.json';assert not out.exists();artifact=P/'MIDPOINT_SOURCE_COVARIANCE_PROGRAMS_V1.pt';plan=json.loads((P/'MIDPOINT_SOURCE_NATIVE_PLAN_V1.json').read_text());assert hashlib.sha256(artifact.read_bytes()).hexdigest()==plan['program_sha256']
 cpu=torch.load(artifact,weights_only=True);keys={'linear':'isotropic_0','isotropic_16':'isotropic_16','covariance_16':'covariance_16','covariance_64':'covariance_64'};programs={k:{n:t.cuda().double() for n,t in cpu[v].items()} for k,v in keys.items()}
 e=programs['linear'];mode={k:t.cuda().double() for k,t in torch.load(P/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True).items()};fold=torch.load(P/'MIDPOINT_CONTINUATION_SOURCE_FOLD_V1.pt',weights_only=True);Qs=[fold[k]['matrix'].cuda() for k in ['a','b']];tokens=torch.load(P/'MIDPOINT_SOURCE_NATIVE_TOKENS_V1.pt',weights_only=True)
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];writer=torch.linalg.solve(e['R_U'],e['writer']);a=mode['A'][:,0];b=mode['B'][:,0]
 logits=lambda x:30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 enc=tiktoken.get_encoding('gpt2');records=[];checks=[]
 for domain,rows in tokens.items():
  assert hashlib.sha256(rows.numpy().tobytes()).hexdigest()==plan['tokens'][domain]
  for doc,row in enumerate(rows):
   labels=annotate(row.tolist(),enc);c=capture(model,row[None,:256].cuda());z=c['x16'].double().flatten(0,1)[16:];h=c['h17'].double().flatten(0,1)[16:];source=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1)[16:];scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();hread=h@a
   exact=torch.stack([source@a,source@b],1);q=torch.stack([((z@Q)*z).sum(1) for Q in Qs],1);checks.append(float((q-exact).norm()/exact.norm()))
   phi=lambda q:((hread-.5*q[:,0])/scale-e['alpha'])*(q[:,1]/scale-e['beta'])
   native=phi(exact);arms={'native':exact,**{k:source_reads(z,v) for k,v in programs.items()}}
   state=c['final'].flatten(0,1)[16:];base=logits(state);target=row[17:257].cuda();ce=F.cross_entropy(base,target,reduction='none');masks=dict(all=torch.ones(240,dtype=torch.bool,device='cuda'),continuation=torch.tensor([v['continuation'] for v in labels[16:256]],device='cuda'),spaced_word=torch.tensor([v['spaced_word'] for v in labels[16:256]],device='cuda'))
   for name,reads in arms.items():
    scalar=phi(reads);changed=logits(state-(scalar[:,None]*writer).float());effect=(changed-base).double();effect-=effect.mean(1,keepdim=True);damage=F.cross_entropy(changed,target,reduction='none')-ce
    if name=='native':reference=effect
    for cohort,mask in masks.items():
     records.append(dict(domain=domain,document=plan['documents'][domain][doc],candidate=name,cohort=cohort,sites=int(mask.sum()),ce_added_sum=float(damage[mask].sum()),reference_energy=float(reference[mask].square().sum()),error_energy=float((effect[mask]-reference[mask]).square().sum()),scalar_error_energy=float((scalar[mask]-native[mask]).square().sum()),scalar_reference_energy=float(native[mask].square().sum()),source_error_energy=float((reads[mask]-exact[mask]).square().sum()),source_reference_energy=float(exact[mask].square().sum())))
 summary={}
 for domain in tokens:
  summary[domain]={}
  for name in arms:
   summary[domain][name]={}
   for cohort in masks:
    rr=[r for r in records if r['domain']==domain and r['candidate']==name and r['cohort']==cohort];num=sum(r['sites'] for r in rr);assert num>0
    summary[domain][name][cohort]=dict(sites=num,ce_added=sum(r['ce_added_sum'] for r in rr)/num,effect_relative_error=(sum(r['error_energy'] for r in rr)/sum(r['reference_energy'] for r in rr))**.5,scalar_relative_error=(sum(r['scalar_error_energy'] for r in rr)/sum(r['scalar_reference_energy'] for r in rr))**.5,source_relative_error=(sum(r['source_error_energy'] for r in rr)/sum(r['source_reference_energy'] for r in rr))**.5)
 pred=dict(pred_a_instrument=max(checks)<1e-4,pred_b_fidelity=all(summary[d]['covariance_16'][c]['effect_relative_error']<=.15 for d in tokens for c in masks),pred_c_behavior=all(abs(summary[d]['covariance_16'][c]['ce_added']-summary[d]['native'][c]['ce_added'])<=.02 for d in tokens for c in masks))
 result=dict(predictions=pred,summary=summary,records=records,source_replay=max(checks),plan=plan,seconds=time.perf_counter()-start,scope=plan['scope']);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
if __name__=='__main__':main()
