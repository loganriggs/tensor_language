"""Shared frozen quartic write intervention scoring on the established developmental rows.
Caller validates source bindings and constructs reference, initial, candidate writes.
"""
from pathlib import Path
import json,torch
import torch.nn.functional as F
from stable_path_native_cache_v1 import relative
P=Path(__file__).resolve().parent
@torch.no_grad()
def score(writes,h,rows,u,reference_effects=None):
 ref=writes[0]
 targets=torch.tensor([r[s+'_answer_id'] for r in rows for s in ('base','donor')])
 def ce(states):
  values=[]
  for i in range(0,len(states),32):
   logits=30*torch.tanh(F.linear(F.rms_norm(states[i:i+32],(1152,)),u)/30)
   values.append(F.cross_entropy(logits,targets[i:i+32],reduction='none').double())
  return torch.cat(values)
 baseline=ce(h);effects=[]
 for w in writes:
  zero=ce(h-w.float())-baseline;swapped=h[::2]+(w[1::2]-w[::2]).float();values=[]
  for i,r in enumerate(rows):
   readers=u[[r['donor_answer_id'],r['donor_foil_id']]]
   z=30*torch.tanh(F.linear(F.rms_norm(torch.stack([h[2*i],swapped[i]]),(1152,)),readers)/30)
   values.append(float((z[1,0]-z[1,1])-(z[0,0]-z[0,1])))
  effects.append((zero,torch.tensor(values,dtype=torch.float64)))
 replay=None
 if reference_effects is not None:
  prior=reference_effects
  replay=max(float((effects[0][0]-torch.tensor(prior['zero_ce'][0],dtype=torch.float64)).abs().max()),float((effects[0][1]-torch.tensor(prior['swaps'][0],dtype=torch.float64)).abs().max()))
 reports=[]
 for idx,name in [(1,'initial'),(2,'candidate')]:
  zero,swaps=effects[idx];rz,rs=effects[0];families=[]
  for family in sorted({r['family'] for r in rows}):
   ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten();a,b=swaps[ids],rs[ids];live=torch.maximum(a.abs(),b.abs())>=1e-4
   families.append(dict(family=family,swap_relative_rms=relative(a,b),swap_live=int(live.sum()),swap_sign_agreement=float((a[live].sign()==b[live].sign()).double().mean()),zero_ce_meanabs_disagreement=float((zero[ep]-rz[ep]).abs().mean()),zero_ce_mean=[float(zero[ep].mean()),float(rz[ep].mean())]))
  reports.append(dict(name=name,physical_error=float((writes[idx]-ref).norm()/ref.norm()),families=families))
 fams=reports[1]['families']
 return dict(reports=reports,prior_reference_effect_max_error=replay,reference_effects=dict(zero_ce=[effects[0][0].tolist()],swaps=[effects[0][1].tolist()]),
  pred_a=all(bool(torch.isfinite(v).all()) for pair in effects for v in pair) and (replay is None or replay<=1e-5),
  pred_b=all(f['swap_relative_rms']<=.1 and f['swap_sign_agreement']>=.9 and f['swap_live']>=4 for f in fams),
  pred_c=all(f['zero_ce_meanabs_disagreement']<=.02 for f in fams))
