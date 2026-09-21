#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_tangent pred_c_large_shift
"""Direct source-path response: analytic tangent vs finite alpha .01/.1/1.
Analytic FP64 central-difference replay<1e-6 and alpha1 prior replay<1e-4.
Tangent logit discrepancy<.10 all4cohorts; alpha1 discrepancy >=1.25*tangent
all4cohorts (large-shift explanation). Keep each failed prediction. 48captures,
no fit; existing3686product/14,067,072coefficient frozen replacement.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(captures=48,fit=False,alphas=[.01,.1,1.],analytic_jvp=True)));return
 import torch
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from native_feature_capture import capture
 from normalized_bilinear_response import residual_pair,output_pair,self_test
 controls=self_test();torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();out=P/'FULL_CHANNEL_SOURCE_RESPONSE_V1.json';assert not out.exists()
 p={k:v.cuda().double() for k,v in torch.load(P/'FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt',weights_only=True).items()};tokens=torch.load(P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt',weights_only=True);donors=json.loads((P/'FULL_CHANNEL_CONTINUATION_INTERCHANGE_DONORS_V1.json').read_text());assert hashlib.sha256((P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt').read_bytes()).hexdigest()==donors['token_sha256']
 model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17];U=model.lm_head.weight.double();pars={'native':(b17.mlp.Left.weight.double(),b17.mlp.Right.weight.double(),b17.mlp.Down.weight.double(),b17.mlp.Down_bias.double(),None),'student':(p['a'],p['b'],p['writer'],p['bias']+b17.mlp.Down_bias.double(),p['linear'])};records=[];checks=[]
 for domain,documents in tokens.items():
  hh=[];mm=[]
  for row in documents:
   cache=capture(model,row[None,:256].cuda());hh.append(cache['h17'].flatten(0,1)[16:255]);mm.append((b17.lambdas[0]*(cache['m16']-b16.mlp.Down_bias)).flatten(0,1)[16:255])
  hh=torch.cat(hh).double();mm=torch.cat(mm).double();mapping=torch.tensor(donors['domains'][domain]['maps']['same_cohort'],device='cuda');cohorts=torch.tensor(donors['domains'][domain]['cohort_labels'],device='cuda')
  for doc in range(len(documents)):
   ids=torch.arange(doc*239,(doc+1)*239,device='cuda');ids=ids[mapping[ids]>=0]
   if not len(ids):continue
   h=hh[ids];v=mm[mapping[ids]]-mm[ids];states={};bases={};tangents={};residuals={}
   for name,par in pars.items():
    r,dr=residual_pair(h,v,*par);states[name]=r;residuals[name]=dr;bases[name],tangents[name]=output_pair(r,dr,U)
    if doc==0:
     tiny=1e-4;forward=output_pair(*residual_pair(h[:4]+tiny*v[:4],v[:4],*par),U)[0];backward=output_pair(*residual_pair(h[:4]-tiny*v[:4],v[:4],*par),U)[0];fd=(forward-backward)/(2*tiny);checks.append(float((fd-tangents[name][:4]).norm()/tangents[name][:4].norm()))
   effects={'tangent':tangents,'tangent_common_background':{'native':tangents['native'],'student':output_pair(states['native'],residuals['student'],U)[1]},'residual_tangent':residuals}
   for alpha in (.01,.1,1.):
    effects[str(alpha)]={name:output_pair(*residual_pair(h+alpha*v,v,*par),U)[0]-bases[name] for name,par in pars.items()}
   for arm,values in effects.items():
    teacher,student=values['native'],values['student']
    if arm!='residual_tangent':teacher=teacher-teacher.mean(1,keepdim=True);student=student-student.mean(1,keepdim=True)
    for cohort,label in [('continuation',1),('spaced_word',2)]:
     mask=cohorts[ids]==label
     if mask.any():records.append(dict(domain=domain,document_index=doc,arm=arm,cohort=cohort,sites=int(mask.sum()),reference_energy=float(teacher[mask].square().sum()),error_energy=float((student[mask]-teacher[mask]).square().sum())))
 keys=('domain','arm','cohort');summary=[]
 for values in sorted(set(tuple(r[k] for k in keys) for r in records)):
  rr=[r for r in records if tuple(r[k] for k in keys)==values];summary.append(dict(zip(keys,values),sites=sum(r['sites'] for r in rr),effect_error=(sum(r['error_energy'] for r in rr)/sum(r['reference_energy'] for r in rr))**.5))
 lookup={(r['domain'],r['arm'],r['cohort']):r['effect_error'] for r in summary};old=json.loads((P/'FULL_CHANNEL_DIRECT_SOURCE_SWAP_V1.json').read_text());replay=max(abs(lookup[r['domain'],'1.0',r['cohort']]-r['effect_error']) for r in old['summary'] if r['kind']=='full' and r['family']=='same_cohort');groups=[(domain,cohort) for domain in tokens for cohort in ('continuation','spaced_word')]
 pred=dict(pred_a_instrument=max(checks)<1e-6 and replay<1e-4,pred_b_tangent=all(lookup[d,'tangent',c]<.1 for d,c in groups),pred_c_large_shift=all(lookup[d,'1.0',c]>=1.25*lookup[d,'tangent',c] for d,c in groups))
 result=dict(predictions=pred,summary=summary,records=records,central_difference_error=max(checks),prior_alpha1_replay=replay,toy_controls=controls,seconds=time.monotonic()-start,scope='Opened same-token/same-cohort donors. Analytic derivative includes input RMS, residual identity, bilinear or affine student, final RMS and softcap. Common-background derivative uses native endpoint normalization Jacobian for both residual tangents. This separates background sensitivity from derivative mismatch; it is not a new fit or whole upstream-state interchange.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
