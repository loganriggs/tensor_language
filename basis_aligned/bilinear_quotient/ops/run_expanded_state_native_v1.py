#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_response pred_c_advantage
"""Frozen purequartic input interchange, .5/1 next-document donors.
Replay<1e-5; response-trained error error<.10, <=.9narrow26 and<=.85smallcal656; baselineCEadded<.02.
Conditional term only; fixed recipient RMS/background; no fitting or semantic adoption.
Prices26/48384,656/903168,656/903168. Null:no wider-bank advantage.
"""
import os,sys,json,time,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def summarize(rows):
 result=[]
 for domain,alpha,arm in sorted({(r['domain'],r['alpha'],r['arm']) for r in rows}):
  rr=[r for r in rows if (r['domain'],r['alpha'],r['arm'])==(domain,alpha,arm)]
  ref=sum(r['reference_energy'] for r in rr);err=sum(r['error_energy'] for r in rr);est=sum(r['estimate_energy'] for r in rr);dot=sum(r['dot'] for r in rr);n=sum(r['tokens'] for r in rr)
  result.append(dict(domain=domain,alpha=alpha,arm=arm,effect_error=math.sqrt(err/ref) if ref>0 else None,cosine=dot/math.sqrt(ref*est) if ref*est>0 else None,native_effect_rms=math.sqrt(ref/(n*50304)),native_ce_response=sum(r['native_ce_response'] for r in rr)/len(rr),candidate_ce_response=sum(r['candidate_ce_response'] for r in rr)/len(rr),baseline_ce_added=sum(r.get('baseline_ce_added',0.) for r in rr)/len(rr)))
 return result

def evaluate_program(arm,programs,x,teacher):
 sys.path.insert(0,str(P))
 from empirical_quartic_dictionary import evaluate
 return teacher(x) if arm=='exact' else evaluate(programs[arm],x)

def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  import torch
  sys.path.insert(0,str(P))
  from empirical_quartic_dictionary import evaluate as bank_check
  torch.manual_seed(81);x=torch.randn(7,3,dtype=torch.float64);program=dict(U=torch.randn(2,2,3,dtype=torch.float64),V=torch.randn(2,2,3,dtype=torch.float64),C=torch.randn(4,3,dtype=torch.float64));programs={k:program for k in ['narrow26','smallcal656','expanded656']}
  for arm in programs:assert torch.equal(evaluate_program(arm,programs,x,None),bank_check(program,x))
  assert torch.equal(evaluate_program('exact',programs,x,lambda z:z),x)
  rr=[dict(domain=d,alpha=a,arm='exact',reference_energy=4.,error_energy=0.,estimate_energy=4.,dot=4.,tokens=2,native_ce_response=.1,candidate_ce_response=.1) for d in ['fineweb','stdlib'] for a in [.5,1.]]
  assert all(r['effect_error']==0 and r['cosine']==1 for r in summarize(rr));print(json.dumps(dict(captures=32,arms=4,strengths=[.5,1],aggregation_smoke='pass')));return
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(P))
 from native_feature_capture import capture
 from native_quartic_branch import pure_branch,bilinear,replace_branch
 from empirical_quartic_dictionary import evaluate as bank
 from circuit_fast_screen_producer import Bilin18TorchBackend
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'EXPANDED_STATE_NATIVE_V1.json';assert not out.exists();start=time.monotonic()
 files={'tokens': 'FULL_CHANNEL_FRESH_TOKENS_V1.pt', 'narrow26': 'WIDE_NATIVE_QUARTIC_4_INHERITED_LONG_V1.pt', 'smallcal656': 'QUARTIC_ENSEMBLE_FIT_V1.pt', 'expanded656': 'QUARTIC_EXPANDED_STATE_V1.pt'}
 hashes={k:hashlib.sha256((P/v).read_bytes()).hexdigest() for k,v in files.items()};assert hashes==json.loads((P/'EXPANDED_STATE_NATIVE_INPUTS_V1.json').read_text())['sha256']
 def move(v):
  if torch.is_tensor(v):return v.cuda()
  if isinstance(v,dict):return {k:move(t) for k,t in v.items()}
  if isinstance(v,list):return [move(t) for t in v]
  return v
 programs={k:move(torch.load(P/files[k],weights_only=True)) for k in ['narrow26','smallcal656','expanded656']};tokens=torch.load(P/files['tokens'],weights_only=True)
 model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17];L,R,D=[getattr(b17.mlp,k).weight for k in ['Left','Right','Down']]
 def teacher(x):return bilinear(b17.lambdas[0]*bilinear(x,b16.mlp.Left.weight,b16.mlp.Right.weight,b16.mlp.Down.weight),L,R,D)
 def logits(x):return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 def evaluate(arm,x):return evaluate_program(arm,programs,x,teacher)
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 checks=[];zeros=[];rows=[]
 for domain,documents in tokens.items():
  captures=[]
  for row in documents[:16]:
   c=capture(model,row[None,:256].cuda());captures.append({k:c[k].flatten(0,1)[16:255] for k in ['x16','m16','h17','final']})
  for doc,c in enumerate(captures):
   x,h,final=c['x16'],c['h17'],c['final'];donor=captures[(doc+1)%len(captures)]['x16'];original=pure_branch(c['m16'],b16.mlp.Down_bias,b17.lambdas[0],L,R,D);native=teacher(x)
   checks.extend([rel(native,original),rel(h+b17.mlp(F.rms_norm(h,(1152,))),final)])
   labels=documents[doc,17:256].cuda();base={};basece={}
   for arm in ['exact','narrow26','smallcal656','expanded656']:
    value=evaluate(arm,x);state=replace_branch(final,h,original,value);base[arm]=logits(state);basece[arm]=F.cross_entropy(base[arm],labels)
    zeros.append(float((logits(replace_branch(final,h,original,evaluate(arm,x+0*(donor-x))))-base[arm]).abs().max()))
   for alpha in [.5,1.]:
    changed={};ce={}
    for arm in base:
     z=logits(replace_branch(final,h,original,evaluate(arm,x+alpha*(donor-x))));changed[arm]=(z-base[arm]).double();ce[arm]=float(F.cross_entropy(z,labels)-basece[arm])
    ref=changed['exact']
    for arm,effect in changed.items():rows.append(dict(domain=domain,document=doc,donor=(doc+1)%len(captures),alpha=alpha,arm=arm,tokens=len(x),reference_energy=float(ref.square().sum()),estimate_energy=float(effect.square().sum()),error_energy=float((effect-ref).square().sum()),dot=float((effect*ref).sum()),native_ce_response=ce['exact'],candidate_ce_response=ce[arm],baseline_ce_added=float(basece[arm]-basece['exact'])))
   print(json.dumps(dict(domain=domain,document=doc)),flush=True)
 summary=summarize(rows);primary=[r for r in summary if r['arm']=='expanded656'];lookup={(r['domain'],r['alpha'],r['arm']):r for r in summary}
 pred=dict(pred_a_instrument=max(checks)<1e-5 and max(zeros)==0 and all(r['effect_error']==0 for r in summary if r['arm']=='exact'),pred_b_response=all(r['effect_error'] is not None and r['effect_error']<.1 for r in primary),pred_c_advantage=all(r['effect_error'] is not None and r['effect_error']<=.9*lookup[r['domain'],r['alpha'],'narrow26']['effect_error'] and r['effect_error']<=.85*lookup[r['domain'],r['alpha'],'smallcal656']['effect_error'] and r['baseline_ce_added']<.02 for r in primary))
 result=dict(predictions=pred,summary=summary,rows=rows,maximum_replay=max(checks),maximum_zero_edit=max(zeros),sha256=hashes,seconds=time.monotonic()-start,scope='Conditional purequartic input swap only, fixed recipient denominator/background; own-baseline-subtracted nonlinear logit responses; opened panels, no semantics or OOD adoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
