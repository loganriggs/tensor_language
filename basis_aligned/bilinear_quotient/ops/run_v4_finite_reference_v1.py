#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_oracle_selectivity pred_c_oracle_prediction
"""Opened v4 oracle diagnostic; V4_FINITE_REFERENCE_V1_PREREGISTRATION.md."""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';PLAN=dict(prefix=12,native_suffix=28)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import numpy as np
 import torch,tiktoken
 import circuit_fast_screen_producer as producer
 from disk_guard import guard_write,guard_torch_save
 sys.path.insert(0,str(P))
 from refined_native_sources import build_refined,PARENTS
 from native_source_observables import source_observables,source_observables32
 from shared_selective_source_lp import choose
 out=A/'v4_finite_reference_v1_result.json';assert not out.exists();choices=json.loads((P/'V4_FINITE_REFERENCE_CHOICES_V1.json').read_text())
 prior=json.loads((A/'fixed_query_fresh_v4_result.json').read_text());contexts=[]
 for panel in ['opposite','congruent']:
  file=f'SOURCE_OOD_V4_{panel.upper()}_ROWS.json';rows=json.loads((P/file).read_text())
  for template in dict.fromkeys(r['template'] for r in rows):
   n=sum(r['template']==template for r in rows);contexts.append(dict(panel=panel,template=template,rows_file=file,amplitudes={r:[[0,0,1,1,1,0]]*n for r in ['subject','attractor']}))
 torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
 for param in model.parameters():param.requires_grad_(False)
 enc=tiktoken.get_encoding('gpt2');controls=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text());modal=torch.tensor([[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]+controls['token_ids'],device='cuda')
 reference=np.array([0.,0.,1.,1.,1.,0.])[PARENTS];counts={k:0 for k in PLAN};replays=[];checks=[];records=[];groups=[]
 for context in contexts:
  c=build_refined(model,context,modal);counts['prefix']+=c['prefix_calls'];n,t,d=c['raw'].shape;families=[r['family'] for r in c['entries']]
  def native(role,aa):
   counts['native_suffix']+=1;pos,ds=c['source_components'][role]
   return source_observables32(model,c['raw'],c['x0'],c['first'],ds,pos,c['read'],c['pairs'],torch.as_tensor(aa,device='cuda',dtype=torch.float64))
  base=native('subject',np.zeros((n,23)))
  for role in ['subject','attractor']:
   choice=next(v for v in choices['records'] if v['panel']==context['panel'] and v['template']==context['template'] and v['role']==role);weights=np.array(choice['amplitudes']);checks.extend(choice['checks']);oldrows={family:next(r for r in prior['records'] if r['dataset']=='v4' and r['panel']==context['panel'] and r['family']==family and r['role']==role) for family in dict.fromkeys(families)};indices={family:0 for family in oldrows};bank=[]
   for family in families:bank.append(oldrows[family]['amplitudes'][indices[family]]);indices[family]+=1
   unit=base-native(role,np.tile(reference,(n,1)));old=base-native(role,bank);effect=base-native(role,weights);prediction=torch.tensor(choice['prediction'],device='cuda',dtype=torch.float64)
   for family in oldrows:
    ids=[i for i,f in enumerate(families) if f==family];previous=oldrows[family]
    replays.extend([float((old[ids]-torch.tensor(previous['full_effect'],device='cuda',dtype=torch.float64)).abs().max()),float((unit[ids]-torch.tensor(previous['reference_effect'],device='cuda',dtype=torch.float64)).abs().max())])
    v=effect[ids];ref=unit[ids,0];den=v[:,0].norm().clamp_min(1e-30);ret=float(v[:,0]@ref/ref.square().sum().clamp_min(1e-30));leak=float(v[:,1:].norm(dim=0).max()/den);err=(prediction[ids]-v).norm(dim=0)/den
    records.append(dict(panel=context['panel'],family=family,role=role,retention=ret,control_ratio=leak,prediction_errors=err.tolist(),effect=v.tolist(),prediction=prediction[ids].tolist(),amplitudes=weights[ids].tolist()))
 instrument=counts==PLAN and max(replays)<=1e-8
 result=dict(plan=PLAN,counts=counts,max_replay=max(replays),selection_checks=checks,records=records,predictions=dict(pred_a_instrument=instrument,pred_b_oracle_selectivity=instrument and all(r['retention']>=.8 and r['control_ratio']<=.1 for r in records),pred_c_oracle_prediction=instrument and all(r['prediction_errors'][0]<=.1 and max(r['prediction_errors'][1:])<=.05 for r in records)),seconds=time.perf_counter()-tic)
 payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=out.name);out.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','max_replay','seconds']}))
if __name__=='__main__':main()
