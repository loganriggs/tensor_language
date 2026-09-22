#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_parent pred_c_retention
"""Move CP256 directions after pruning, exact fixedCP512 parent target,100Muonsteps.
Two starts;768products1202176floats. Replay/export<1e-4, parentGaussian<=1%.
Both native text/response/sensitivity<=1.1unprunedparent; absolute10%reported.
"""
import os,sys,time,math,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P))
 from cp_parent_refit import prepare,objective
 from quartic_cp_profile import normalize_factors
 from quartic_cp import cp_gram
 from mixed_gaussian_cp import gram_dynamic
 from check_cp_parent_refit import controls
 torch.set_num_threads(2)
 def values(x,f,c):
  phi=torch.ones(len(x),len(f[0]),dtype=x.dtype,device=x.device)
  for a in f:phi*=x@a.T
  return phi@c.T
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  checks=controls();torch.manual_seed(13000);dtype=torch.float64;tf=normalize_factors([torch.randn(512,12,dtype=dtype) for _ in range(4)]);tc=torch.randn(16,512,dtype=dtype);params=[torch.randn(256,12,dtype=dtype,requires_grad=True) for _ in range(4)];parent=prepare(tf,tc,torch.eye(12,dtype=dtype),torch.randn(12,dtype=dtype),1000.);loss,c,info=objective(parent,normalize_factors(params));loss.backward();assert c.shape==(16,256) and all(torch.isfinite(a.grad).all() for a in params);assert values(torch.randn(7,12,dtype=dtype),normalize_factors(params),c).shape==(7,16)
  print(json.dumps(dict(control_cases=len(checks),actual512parent256student16outputs=True,backward=True)));return
 torch.backends.cuda.matmul.allow_tf32=False;out=P/'CP_PARENT_REFIT_NATIVE_V1.json';assert not out.exists();start=time.monotonic();scale=19054614563.464127
 cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'].cuda();mu=cache['mean'].cuda();pruned=torch.load(P/'MIXED_CP_PRUNING_CANDIDATES_V1.pt',weights_only=True)
 text=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].cuda().double();labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1];target=labels['target'].cuda()/scale;weights=labels['weight'].cuda();matched=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1];rec,don=torch.tensor(matched['pairs_flat'],device='cuda').T;reference=torch.tensor([r['reference'] for r in matched['pair_rows']],device='cuda',dtype=torch.float64)/scale
 @torch.no_grad()
 def native_metrics(f,c):
  pred=values(text,f,c);sens=((weights*(pred-target).square()).sum(0)/(weights*target.square()).sum(0)).sqrt()
  return dict(text_error=float((pred-target).norm()/target.norm()),root1_sensitivity_error=float(sens[1]),root1_same_token_error=float(((pred[don,1]-pred[rec,1])-reference).norm()/reference.norm()))
 @torch.no_grad()
 def parent_metrics(parent,f,c):
  def error(gss,gpp,gps):
   energy=(parent['coefficients']*(parent['coefficients']@gpp)).sum();res=(c*(c@gss)).sum()+energy-2*(c*(parent['coefficients']@gps)).sum();assert res>=-1e-7*energy
   return float((res.clamp_min(0)/energy).sqrt())
  tf=parent['factors'];fw=[a@S for a in f];bias=[a@mu for a in f];tw=parent['transformed'];tb=parent['bias']
  return dict(coefficient_error=error(cp_gram(f,f),cp_gram(tf,tf),cp_gram(tf,f)),gaussian_error=error(gram_dynamic(fw,bias,fw,bias),gram_dynamic(tw,tb,tw,tb),gram_dynamic(tw,tb,fw,bias)))
 rows=[]
 for seed in [1001,1002]:
  source=torch.load(P/f'MIXED_CP_FEATURES_SEED{seed}_V1.pt',weights_only=True);tf=[a.cuda().double() for a in source['factors']];tc=source['coefficients'].cuda().double()/scale;parent=prepare(tf,tc,S,mu,source['coefficient_weight']);baseline=native_metrics(tf,tc);init=pruned[(seed,'mixed')][256];params=[torch.nn.Parameter(a.cuda().double()) for a in init['factors']];rate=.1*math.sqrt(4/1152);opt=torch.optim.Muon(params,lr=rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw');history=[];best=None
  for step in range(101):
   factors=normalize_factors(params);loss,c,info=objective(parent,factors);value=float(loss.detach());assert math.isfinite(value)
   if step==0:
    normalizer=float(parent['energy']);initial=native_metrics(factors,c);initial_parent=parent_metrics(parent,factors,c)
    with torch.no_grad():old=values(text[:128],[a.cuda().double() for a in init['factors']],init['coefficients'].cuda().double()/scale);replay=float((values(text[:128],factors,c)-old).norm()/old.norm());assert replay<1e-4
   if best is None or value<best[0]:best=(value,step,[a.detach().clone() for a in factors],c.detach().clone())
   if step%10==0:
    row=dict(seed=seed,step=step,objective=value,elapsed=time.monotonic()-start);history.append(row);print(json.dumps(row),flush=True)
   if step==100:break
   opt.zero_grad();(loss/normalizer).backward();assert all(torch.isfinite(a.grad).all() for a in params)
   if step==0:assert torch.cuda.max_memory_allocated()<20_000_000_000
   opt.step()
   for group in opt.param_groups:group['lr']=rate*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/100)))
  with torch.no_grad():
   factors,c=best[2:];final=native_metrics(factors,c);final_parent=parent_metrics(parent,factors,c);ref=values(text[:128],factors,c);fp=values(text[:128].float(),[a.float() for a in factors],(c*scale).float()).double()/scale;drift=float((fp-ref).norm()/ref.norm());path=P/f'CP_PARENT_REFIT_SEED{seed}_V1.pt';torch.save(dict(factors=[a.cpu().float() for a in factors],coefficients=(c*scale).cpu().float(),writer=source['writer'],seed=seed,degree=4),path)
   rows.append(dict(seed=seed,unpruned_native=baseline,initial=initial,final=final,initial_parent=initial_parent,final_parent=final_parent,selected_step=best[1],history=history,baseline_replay=replay,export_error=drift,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  del opt,params,best,factors,c,loss,parent
 pred=dict(pred_a_integrity=all(r['baseline_replay']<1e-4 and r['export_error']<1e-4 for r in rows),pred_b_parent=all(r['final_parent']['gaussian_error']<=.01 for r in rows),pred_c_retention=all(all(r['final'][k]<=1.1*r['unpruned_native'][k] for k in ['text_error','root1_sensitivity_error','root1_same_token_error']) for r in rows))
 result=dict(predictions=pred,absolute_component_gate=all(r['final']['root1_sensitivity_error']<=.1 and r['final']['root1_same_token_error']<=.1 for r in rows),rows=rows,steps=100,products=768,coefficients=1202176,seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),scope='Exact compression of frozen learnedCP512parent, not exactnativequartic target. CP256 directionsmove afterpruning; unit-featuremixedmetricridge1e-10 unchanged. Nativeopenedlabels evaluationonly, bestfittingobjectivecheckpoint. No finite-removal/OOD/semanticadoption fromscreen.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
