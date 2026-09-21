#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_preservation pred_c_baselines
"""Frozen paired384 branch vsparent656 andnarrow26, twoopeneddomains.
Replay<1e-5,graphreplay<1e-4. PairedCEadded/KL<.02eachdomain;
branch-effecterror<=1.1parent and<=.9narroweachdomain. No semanticadoption.
"""
import os,sys,json,time,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def summarize(rows):
 domains=sorted(set(r['domain'] for r in rows));arms=sorted(set(r['arm'] for r in rows));result=[]
 for domain in domains:
  ablation=sum(r['error_energy'] for r in rows if r['domain']==domain and r['arm']=='ablation')
  for arm in arms:
   rr=[r for r in rows if r['domain']==domain and r['arm']==arm];energy=sum(r['error_energy'] for r in rr)
   result.append(dict(domain=domain,arm=arm,documents=len(rr),ce_added=sum(r['ce_added'] for r in rr)/len(rr),kl=sum(r['kl'] for r in rr)/len(rr),argmax_agreement=sum(r['argmax_agreement'] for r in rr)/len(rr),effect_relative_error=math.sqrt(energy/ablation) if ablation>0 else None,worst_document_ce_added=max(r['ce_added'] for r in rr)))
 return result

def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  rows=[dict(domain=d,arm=a,error_energy=e,ce_added=.001,kl=.002,argmax_agreement=.99) for d in ['fineweb','code'] for a,e in [('ablation',4.),('paired384',1.)]];assert all(r['effect_relative_error']==.5 for r in summarize(rows) if r['arm']=='paired384');print(json.dumps(dict(captures=48,fit=False,arms=['exact','ablation','narrow26','parent656','paired384'],summary_shape_smoke='pass')));return
 import torch
 import torch.nn.functional as F
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from native_feature_capture import capture
 from native_quartic_branch import pure_branch,bilinear,replace_branch
 from empirical_quartic_dictionary import evaluate as bank_evaluate
 from paired_root_compiler import evaluate as pair_evaluate
 from shared_root_block_compiler import evaluate as spectral_evaluate
 from circuit_fast_screen_producer import Bilin18TorchBackend
 out=P/'PAIRED_ROOT_BRANCH_V1.json';assert not out.exists();start=time.monotonic()
 files={'tokens':'FULL_CHANNEL_FRESH_TOKENS_V1.pt','narrow26':'WIDE_NATIVE_QUARTIC_4_INHERITED_LONG_V1.pt','parent656':'QUARTIC_JOINT_READOUT_LAMBDA_1_V1.pt','paired384':'PAIRED_ROOT_RANK_16_V2.pt','spectral':'QUARTIC_RANK_BLOCK_16_V1.pt'};hashes={k:hashlib.sha256((P/f).read_bytes()).hexdigest() for k,f in files.items()}
 frozen=json.loads((P/'PAIRED_ROOT_BRANCH_INPUTS_V1.json').read_text());assert hashes==frozen['sha256']
 def move(value):
  if torch.is_tensor(value):return value.cuda()
  if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
  if isinstance(value,list):return [move(v) for v in value]
  return value
 tokens=torch.load(P/files['tokens'],weights_only=True);programs={k:move(torch.load(P/f,weights_only=True)) for k,f in files.items() if k!='tokens'};model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17];L,R,D=[getattr(b17.mlp,k).weight for k in ['Left','Right','Down']];rows=[];checks=[];graph_checks=[]
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 def logits(x):return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 for domain,documents in tokens.items():
  for doc,row in enumerate(documents):
   cache=capture(model,row[None,:256].cuda());h=cache['h17'];final=cache['final'];checks.append(rel(h+b17.mlp(F.rms_norm(h,(1152,))),final));original=pure_branch(cache['m16'],b16.mlp.Down_bias,b17.lambdas[0],L,R,D);previous=bilinear(cache['x16'],b16.mlp.Left.weight,b16.mlp.Right.weight,b16.mlp.Down.weight);exact=bilinear(b17.lambdas[0]*previous,L,R,D);x=cache['x16'].flatten(0,1);paired=pair_evaluate(programs['paired384'],x).reshape_as(original);spectral=spectral_evaluate(programs['spectral'],x).reshape_as(original);graph_checks.append(rel(paired,spectral))
   baseline=logits(final)[:,16:255];logp=F.log_softmax(baseline,dim=-1);prob=logp.exp();labels=row[17:256].cuda()[None];ce=F.cross_entropy(baseline.flatten(0,1),labels.flatten())
   for arm in ['exact','ablation','narrow26','parent656','paired384']:
    replacement=exact if arm=='exact' else torch.zeros_like(original) if arm=='ablation' else paired if arm=='paired384' else bank_evaluate(programs[arm],x).reshape_as(original)
    changed=replace_branch(final,h,original,replacement);z=logits(changed)[:,16:255];delta=z-baseline
    if arm=='exact':checks.extend([rel(changed,final),rel(z,baseline)])
    rows.append(dict(domain=domain,document=doc,arm=arm,tokens=labels.numel(),native_ce=float(ce),ce_added=float(F.cross_entropy(z.flatten(0,1),labels.flatten())-ce),kl=float((prob*(logp-F.log_softmax(z,dim=-1))).sum(-1).mean()),error_energy=float(delta.double().square().sum()),argmax_agreement=float((z.argmax(-1)==baseline.argmax(-1)).float().mean())))
   print(json.dumps(dict(domain=domain,document=doc)),flush=True)
 summary=summarize(rows);prim=[r for r in summary if r['arm']=='paired384'];lookup={(r['domain'],r['arm']):r for r in summary};pred=dict(pred_a_instrument=max(checks)<1e-5 and max(graph_checks)<1e-4,pred_b_preservation=all(r['ce_added']<.02 and r['kl']<.02 for r in prim),pred_c_baselines=all(r['effect_relative_error'] is not None and r['effect_relative_error']<=1.1*lookup[(r['domain'],'parent656')]['effect_relative_error'] and r['effect_relative_error']<=.9*lookup[(r['domain'],'narrow26')]['effect_relative_error'] for r in prim))
 result=dict(predictions=pred,summary=summary,rows=rows,maximum_replay=max(checks),maximum_graph_replay=max(graph_checks),sha256=hashes,seconds=time.monotonic()-start,scope='Additive purequartic branch replacement under actual fixedRMSdenominator andnativebackground. Twoalreadyopened domains, notwholeblock orsemanticintervention. Instrument retainsnativecomputation; noend-to-end speedup.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,summary=summary,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
