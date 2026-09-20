#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_gain pred_b_prediction pred_c_export
"""Registered full-quartic Gaussian quadratic fitting; native plan V1.
pred_a_gain: each metric has an arm improving penalized objective >=1% of initial magnitude.
pred_b_prediction: centered weight-selected C beats paired mean-only B panel2 error by >=.005.
pred_c_export: all exports replay <1e-5 with34560 scalars,4products.
Null: improved Gaussian objective fails to improve native prediction. A/B/C controls
separate old Taylor mean, full teacher mean only, and refitted quadratic component.
Fixed rank8 linear branch. Two metrics,two optimizers,two paired starts,100steps.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(metrics=['centered','spherical'],optimizers=['adam','muon'],starts=['adam','muon'],seed=2,steps=100,lr=.005,penalty=.001,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P))
 from gaussian_projected_quadratic_cross import ProjectedQuadratic
 from fit_projected_quadratic import fit,objective
 from gaussian_quartic_mean import native_mean
 from audit_centered_compact import evaluate
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_GAUSSIAN_QUADRATIC_FIT_V1.json';assert not out.exists();start=time.perf_counter()
 base=torch.load(P/'REGULARIZED_CENTERED_EXPORT_V1.pt',weights_only=True);core=torch.load(P/'CENTERED_QUADRATIC_CAPACITY_V1.pt',weights_only=True);fits=torch.load(P/'CENTERED_REGULARIZED_REFACTOR_V1.pt',weights_only=True)['fits'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];targets=[t.cuda().double() for t in torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets']];xs=[p['rows'].cuda().double() for p in panels];mu=panels[0]['mean'].cuda().double();cov=panels[0]['covariance'].cuda().double();template={k:v.cuda().double() for k,v in base['programs'][.001].items()}
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/base['teacher_scale'],w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
 del state,ru
 oldmean=template['constant']+template['quadratic_writer']@((template['quadratic_left']@cov)*template['quadratic_right']).sum(1)
 records=[];exports={};controls=[]
 def score(model):
  return [dict(error=float((evaluate(model,x)-y).norm()/y.norm()),variation=float(((evaluate(model,x)-evaluate(model,x).mean(0))-(y-y.mean(0))).norm()/(y-y.mean(0)).norm())) for x,y in zip(xs,targets)]
 for metric in PLAN['metrics']:
  M=cov if metric=='centered' else torch.eye(len(cov),device='cuda',dtype=torch.float64)*cov.trace()/len(cov)
  with torch.no_grad():mean=native_mean([t.double() for t in teacher],mu,M)
  op=ProjectedQuadratic(teacher,mu.float(),M.float())
  for warm in PLAN['starts']:
   f=fits[(.001,warm,2)];a=(f['a']@core['input_mapback']).cuda();b=(f['b']@core['input_mapback']).cuda();c=(core['output_frame']@f['c']).cuda();A={k:v.clone() for k,v in template.items()};A.update(quadratic_left=a,quadratic_right=b,quadratic_writer=c,constant=oldmean-c@((a@cov)*b).sum(1));B={**A,'constant':mean-c@((a@M)*b).sum(1)};control=dict(metric=metric,warmstart=warm,A=score(A),B=score(B));controls.append(control)
   for optimizer in PLAN['optimizers']:
    info,(u,v,writer)=fit(op,a.float(),b.float(),optimizer=optimizer,lr=PLAN['lr'],steps=PLAN['steps'],penalty=PLAN['penalty']);u,v=u.double(),v.double();model={**template,'quadratic_left':u,'quadratic_right':v,'quadratic_writer':writer,'constant':mean-writer@((u@M)*v).sum(1)};archive={k:t.float().cpu() for k,t in model.items()};loaded={k:t.cuda() for k,t in archive.items()};replay=max(float((evaluate(loaded,x[:32])-evaluate(model,x[:32])).norm()/evaluate(model,x[:32]).norm()) for x in xs);price=sum(t.numel() for t in archive.values());key=(metric,warm,optimizer);exports[key]=archive
    row=dict(metric=metric,warmstart=warm,optimizer=optimizer,**info,panels=score(loaded),scalar_count=price,products=4,archive_replay=replay);records.append(row);print(json.dumps(row),flush=True)
  del op
 winners=[min((r for r in records if r['metric']==m),key=lambda r:r['best_objective']) for m in PLAN['metrics']];win=winners[0];paired=next(c for c in controls if c['metric']=='centered' and c['warmstart']==win['warmstart']);pred=dict(pred_a_gain=all(any((r['initial_objective']-r['best_objective'])/abs(r['initial_objective'])>=.01 for r in records if r['metric']==m) for m in PLAN['metrics']),pred_b_prediction=win['panels'][1]['error']<=paired['B'][1]['error']-.005,pred_c_export=all(r['archive_replay']<1e-5 and r['scalar_count']==34560 for r in records))
 guard_torch_save(dict(programs=exports,teacher_scale=base['teacher_scale']),str(P/'NATIVE_GAUSSIAN_QUADRATIC_FIT_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,controls=controls,records=records,winners=winners,predictions=pred,seconds=time.perf_counter()-start,scope='Gaussian quadratic projection fitted from weights and input moments; fixed linear branch. Reused diagnostic panels, no empirical model selection or OOD claim.'),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
