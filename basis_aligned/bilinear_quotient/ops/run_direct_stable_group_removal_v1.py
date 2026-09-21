#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_groups pred_c_controls
"""Frozen split-derived group removal agreement on16unused FineWeb documents.
pred_a_instrument hashes, finite nonzero effects and residual/readout replay<1e-8.
pred_b_groups all five groups have <=.10 relative centered-logit effect discrepancy
between half0/half1, denominator min aggregate effect norm.
pred_c_controls merged group1 has lower discrepancy than each individual1/4.
No fitting. Group removal consistency, not semantic/native-unit identification.
"""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=16,groups=5,individual_controls=2,contexts=256,fit=False)));return
 import torch
 import torch.nn.functional as F
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from native_feature_capture import capture
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'MIDPOINT_STABLE_GROUP_REMOVAL_NATIVE_V1.json';assert not out.exists();start=time.perf_counter()
 plan=json.loads((P/'MIDPOINT_STABLE_GROUP_REMOVAL_PLAN_V1.json').read_text());artifact=P/'MIDPOINT_STABLE_GROUP_REMOVAL_PROGRAMS_V1.pt';assert hashlib.sha256(artifact.read_bytes()).hexdigest()==plan['program_sha256']
 tokens=torch.load(P/'MIDPOINT_STABLE_GROUP_REMOVAL_TOKENS_V1.pt',weights_only=True);assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==plan['token_sha256']
 programs={k:{n:v.cuda().double() for n,v in e.items()} for k,e in torch.load(artifact,weights_only=True).items()}
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17]
 _,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');invru=torch.linalg.inv(ru)
 logits=lambda x:30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 groups={**plan['groups'],**plan['individual_controls']};records=[];checks=[]
 for doc,row in enumerate(tokens):
  c=capture(model,row[None,:256].cuda());h=c['h17'].double().flatten(0,1);m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1)
  scale=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=((h-m/2)/scale)[16:];m=(m/scale)[16:];state=c['final'].flatten(0,1)[16:];base=logits(state)
  predictions={}
  for name,e in programs.items():predictions[name]=((n@e['A']-e['left_mean'])*(m@e['B']-e['right_mean']))
  for group,ids in groups.items():
   effects=[];writes=[]
   for name,e in programs.items():
    reduced=predictions[name][:,ids]@e['W'][:,ids].T;delta=-reduced@invru.T
    if doc==0:checks.append(float((delta@ru.T+reduced).norm()/reduced.norm()))
    effect=(logits(state+delta.float())-base).double();effect-=effect.mean(1,keepdim=True);effects.append(effect);writes.append(reduced)
   a,b=effects;wa,wb=writes
   records.append(dict(document=int(plan['documents'][doc]),group=group,sites=len(a),effect_energy_half0=float(a.square().sum()),effect_energy_half1=float(b.square().sum()),effect_discrepancy_energy=float((a-b).square().sum()),reduced_write_energy_half0=float(wa.square().sum()),reduced_write_energy_half1=float(wb.square().sum()),reduced_write_discrepancy_energy=float((wa-wb).square().sum())))
 summary={}
 for group in groups:
  rr=[r for r in records if r['group']==group];a=sum(r['effect_energy_half0'] for r in rr);b=sum(r['effect_energy_half1'] for r in rr);err=sum(r['effect_discrepancy_energy'] for r in rr)
  summary[group]=dict(relative_effect_discrepancy=(err/min(a,b))**.5,effect_energy_half0=a,effect_energy_half1=b,sites=sum(r['sites'] for r in rr))
 instrument=max(checks)<1e-8 and all(torch.isfinite(torch.tensor(v['relative_effect_discrepancy'])) and min(v['effect_energy_half0'],v['effect_energy_half1'])>1e-12 for v in summary.values())
 predictions=dict(pred_a_instrument=bool(instrument),pred_b_groups=all(summary[k]['relative_effect_discrepancy']<=.1 for k in plan['groups']),pred_c_controls=all(summary['group1']['relative_effect_discrepancy']<summary[k]['relative_effect_discrepancy'] for k in plan['individual_controls']))
 result=dict(predictions=predictions,summary=summary,records=records,readout_replay=max(checks),program_sha256=plan['program_sha256'],token_sha256=plan['token_sha256'],seconds=time.perf_counter()-start,scope=plan['scope']+' Remove each frozen centered-input subgraph from the original native final residual, with native final normalization and softcap. Individual extracted sums need not be mean-zero. No ground-truth native unit or selective behavioral claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
if __name__=='__main__':main()
