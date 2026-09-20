#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_moments pred_b_primary pred_c_law
"""Frozen quartic Gaussian mean correction; QUARTIC_MEAN_CORRECTION_PLAN_V1.md.
pred_a_moments: quadrature<1e-10 and centered prediction invariant<1e-10.
pred_b_primary: rank8 Gaussian correction error<.23122887 at47312 scalars.
pred_c_law: rank8 Gaussian error within.03 of empirical calibration correction.
Null: coefficient metric plus Gaussian law fails to predict empirical mean.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(source=['second_floor01','muon',1],output_ranks=[10,8,4],primary_rank=8,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P))
 from gaussian_quartic_mean import native_mean,bank_mean,check
 from shared_quadratic_bank import bank_gram
 validation=check();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_QUARTIC_MEAN_V1.json';assert not out.exists();start=time.perf_counter();source=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True);s=source['students'][tuple(PLAN['source'])];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];targets=[t.cuda().double() for t in torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets']];state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));scale=source['teacher_scale'];teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];teacher=[t.double() for t in teacher];mu=panels[0]['mean'].cuda().double();cov=panels[0]['covariance'].cuda().double();mean_teacher=native_mean(teacher,mu,cov);del teacher,state;U,V,C=[s[n].cuda().double() for n in ['U','V','C']];L=source['metric_transforms'][PLAN['source'][0]].cuda().double();G=bank_gram(U@L,V@L);ev,B=torch.linalg.eigh((G+G.T)/2);W,_,_=torch.linalg.svd(C@(B*ev.clamp_min(0).sqrt()),full_matrices=False);i,j=torch.triu_indices(4,4,device='cuda');features=[]
  for panel in panels:
   x=panel['rows'].cuda().double();q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),4,4).sum(2);features.append(q[:,i]*q[:,j])
  records=[];exports={}
  for rank in PLAN['output_ranks']:
   writer=W[:,:rank];mix=writer.T@C;approx=C if rank==10 else writer@mix;mean_student=bank_mean(U,V,approx,mu,cov);bias=mean_teacher-mean_student;predictions=[phi@approx.T for phi in features];empirical_bias=(targets[0]-predictions[0]).mean(0);rows=[];invariant=[]
   for n,(prediction,y) in enumerate(zip(predictions,targets)):
    oracle=(y-prediction).mean(0);baseline_centered=prediction-prediction.mean(0);centered_target=y-y.mean(0);row=dict(panel=n,centered_relative_error=float((baseline_centered-centered_target).norm()/centered_target.norm()),oracle_total_error=float((prediction+oracle-y).norm()/y.norm()),teacher_gaussian_mean_error=float((mean_teacher-y.mean(0)).norm()/(y.norm()/len(y)**.5)))
    for name,b in [('uncorrected',torch.zeros_like(bias)),('gaussian',bias),('empirical_calibration',empirical_bias)]:
     changed=prediction+b;row[name+'_error']=float((changed-y).norm()/y.norm());invariant.append(float(((changed-changed.mean(0))-baseline_centered).norm()/baseline_centered.norm()))
    rows.append(row)
   price=(48384 if rank==10 else 36864+1162*rank)+1152;record=dict(rank=rank,scalar_count=price,products=26,panels=rows,centered_invariance_error=max(invariant),gaussian_vs_empirical_bias_relative=float((bias-empirical_bias).norm()/empirical_bias.norm()));records.append(record)
   # Rank10 keeps the original writer so archived price is literal for every arm.
   program=dict(U=U.float().cpu(),V=V.float().cpu(),constant=bias.float().cpu())
   if rank==10:program['C']=C.float().cpu()
   else:program.update(W=writer.float().cpu(),Z=mix.float().cpu())
   assert sum(t.numel() for t in program.values())==price;exports[rank]=program;print(json.dumps(record),flush=True)
  primary=next(r for r in records if r['rank']==8);row=primary['panels'][1];pred=dict(pred_a_moments=max(validation['native_relative_error'],validation['bank_relative_error'])<1e-10 and max(r['centered_invariance_error'] for r in records)<1e-10,pred_b_primary=row['gaussian_error']<.23122887 and primary['scalar_count']==47312,pred_c_law=abs(row['gaussian_error']-row['empirical_calibration_error'])<.03);guard_torch_save(dict(programs=exports,teacher_scale=scale,mean_teacher_gaussian=mean_teacher.cpu(),scope='Gaussianbias programs only. Empirical/oracle constants are diagnostic and not exported.'),str(P/'NATIVE_QUARTIC_MEAN_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,validation=validation,records=records,predictions=pred,seconds=time.perf_counter()-start,scope='Frozenprograms plusGaussian mean correction. All evaluation reuses previouspanels. Rank10exports original C; rank4/8use W,Z. No empirical/oraclebias candidate export.'),indent=2)+'\n')
if __name__=='__main__':main()
