#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_identity pred_b_constant pred_c_variation
"""Mean-vs-variation audit: pred_a_identity pred_b_constant pred_c_variation."""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(students=16,panels=2,rows=2048,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_VARIATION_AUDIT_V1.json';assert not out.exists();start=time.perf_counter();source=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True);capture=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];fit=json.load(open(P/'NATIVE_WEIGHTED_BANK_V1.json'))['records'];lookup={(r['metric'],r['optimizer'],r['seed']):r for r in fit};state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));C=ru@w('transformer.h.17.mlp.Down.weight')/source['teacher_scale'];L=w('transformer.h.17.mlp.Left.weight');R=w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];A=w('transformer.h.16.mlp.Left.weight');B=w('transformer.h.16.mlp.Right.weight')
  def teacher(x):
   h=((x@A.T)*(x@B.T))@D.T;return (((h@L.T)*(h@R.T))@C.T).double()
  xs=[p['rows'].cuda() for p in capture];targets=[teacher(x) for x in xs];constant=targets[0].mean(0);weightconstant=teacher(capture[0]['mean'].cuda()[None])[0];panels=[];rows=[];docstats={};i,j=torch.triu_indices(4,4,device='cuda')
  for panel,(x,y) in enumerate(zip(xs,targets)):
   mean=y.mean(0);center=y-mean;energy=y.square().sum();var=center.square().sum();panels.append(dict(panel=panel,teacher_mean_energy_fraction=float(len(y)*mean.square().sum()/energy),calibration_output_mean_constant_error=float((y-constant).norm()/energy.sqrt()),weight_mean_constant_error=float((y-weightconstant).norm()/energy.sqrt())))
   for key,s in source['students'].items():
    U=s['U'].cuda();V=s['V'].cuda();writer=s['C'].cuda();q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),4,4).sum(2);pred=((q[:,i]*q[:,j])@writer.T).double();pm=pred.mean(0);pc=pred-pm;residual=pred-y;total=residual.square().sum();meanerr=len(y)*(pm-mean).square().sum();centered=(pc-center).square().sum();identity=float(abs(total-meanerr-centered)/total);error=float((total/energy).sqrt());expected=lookup[key]['empirical_calibration_error' if panel==0 else 'empirical_evaluation_error'];row=dict(metric=key[0],optimizer=key[1],seed=key[2],panel=panel,total_relative_error=error,centered_relative_error=float((centered/var).sqrt()),centered_cosine=float((pc*center).sum()/(pc.norm()*center.norm())),mean_error_fraction_of_residual=float(meanerr/total),mean_energy_error_fraction_of_teacher=float(meanerr/energy),centered_energy_error_fraction_of_teacher=float(centered/energy),decomposition_relative_discrepancy=identity,previous_error_discrepancy=abs(error-expected));rows.append(row);r=residual.reshape(32,64,1152);rm=r.mean(1);docstats[(key,panel)]=dict(total=r.square().sum((1,2)).cpu(),mean=64*rm.square().sum(1).cpu(),centered=(r-rm[:,None]).square().sum((1,2)).cpu());print(json.dumps(row),flush=True)
  best=max((r for r in fit if r['metric']=='second_floor01'),key=lambda r:r['weighted_gain']);selected=next(r for r in rows if r['panel']==1 and (r['metric'],r['optimizer'],r['seed'])==(best['metric'],best['optimizer'],best['seed']));predictions=dict(pred_a_identity=all(r['decomposition_relative_discrepancy']<1e-10 and r['previous_error_discrepancy']<1e-4 for r in rows),pred_b_constant=selected['total_relative_error']<=panels[1]['calibration_output_mean_constant_error']-.05,pred_c_variation=selected['centered_relative_error']<.6);guard_torch_save(dict(targets=[y.cpu() for y in targets],token_sha256=[p['token_sha256'] for p in capture],teacher_scale=source['teacher_scale'],document_residuals=docstats),str(P/'NATIVE_VARIATION_AUDIT_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,panels=panels,records=rows,selected_second_moment=selected,predictions=predictions,seconds=time.perf_counter()-start,scope='Reused empiricalpanel diagnostics of purequartic. Mean/centered decomposition, fixed calibrationconstant andweightconstant baselines. No outputfitting ofstudents or OOD/circuitclaim.'),indent=2)+'\n')
if __name__=='__main__':main()
