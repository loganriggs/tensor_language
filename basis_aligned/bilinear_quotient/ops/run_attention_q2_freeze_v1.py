#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_q2_fidelity pred_c_q2_selectivity
"""Finite baseline Q2 reuse; ATTENTION_Q2_FREEZE_V1_PREREGISTRATION.md."""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';PLAN=dict(prefix=12,native_suffix=32)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch,tiktoken
 import torch.nn.functional as F
 import circuit_fast_screen_producer as producer
 from disk_guard import guard_write
 sys.path.insert(0,str(P))
 from refined_native_sources import build_refined
 from native_source_observables import source_observables32
 out=A/'attention_q2_freeze_v1_result.json';assert not out.exists()
 priorpath=A/'refined_selective_sources_v1_result.json';prior=json.loads(priorpath.read_text());binding=json.loads((P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json').read_text());extra=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text())
 torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model;enc=tiktoken.get_encoding('gpt2');modal=torch.tensor([[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]+extra['token_ids'],device='cuda');counts={k:0 for k in PLAN};replays=[];records=[]
 for context in binding['contexts']:
  c=build_refined(model,context,modal);counts['prefix']+=c['prefix_calls'];n=len(c['raw']);att=model.transformer.h[11].attn
  z=F.rms_norm(c['raw'],(c['raw'].shape[-1],));fixed={name:getattr(att,name)(z).clone() for name in ['c_q','c_q2']}
  def native(role,aa,freeze=None):
   counts['native_suffix']+=1;pos,ds=c['source_components'][role];handle=None
   if freeze is not None:handle=getattr(att,freeze).register_forward_hook(lambda module,args,output:fixed[freeze])
   try:return source_observables32(model,c['raw'],c['x0'],c['first'],ds,pos,c['read'],c['pairs'],torch.as_tensor(aa,device='cuda',dtype=torch.float64))
   finally:
    if handle is not None:handle.remove()
  zeros=torch.zeros(n,23,device='cuda');baseline=native('subject',zeros);replays.append(float((baseline-native('subject',zeros,'c_q2')).abs().max()))
  for role in ['subject','attractor']:
   aa=next(v for v in prior['amplitudes'] if v['panel']==context['panel'] and v['template']==context['template'] and v['role']==role)['amplitudes']['oracle']
   effects={name:baseline-native(role,aa,freeze) for name,freeze in [('full',None),('freeze_Q2','c_q2'),('freeze_Q1','c_q')]}
   families=[r['family'] for r in c['entries']]
   for family in dict.fromkeys(families):
    ids=[i for i,f in enumerate(families) if f==family];full=effects['full'][ids];den=full[:,0].norm().clamp_min(1e-30)
    old=next(r for r in prior['records'] if r['panel']==context['panel'] and r['role']==role and r['family']==family and r['arm']=='oracle');replays.append(float((full-torch.tensor(old['effect'],device='cuda',dtype=torch.float64)).abs().max()))
    unit=next(r for r in prior['records'] if r['panel']==context['panel'] and r['role']==role and r['family']==family and r['arm']=='unitB');reference=torch.tensor(unit['effect'],device='cuda',dtype=torch.float64)[:,0]
    for name,effect in effects.items():
     v=effect[ids];error=(v-full).norm(dim=0)/den;ret=float(v[:,0]@reference/reference.square().sum());leak=float(v[:,1:].norm(dim=0).max()/v[:,0].norm().clamp_min(1e-30))
     records.append(dict(panel=context['panel'],family=family,role=role,arm=name,errors=error.tolist(),retention=ret,control_ratio=leak,effect=v.tolist(),fidelity=bool(error[0]<=.1 and error[1:].max()<=.05),selectivity=bool(ret>=.8 and leak<=.1)))
 instrument=counts==PLAN and max(replays)<=1e-8;q2=[r for r in records if r['arm']=='freeze_Q2']
 result=dict(plan=PLAN,counts=counts,prior_sha256=hashlib.sha256(priorpath.read_bytes()).hexdigest(),max_replay=max(replays),records=records,predictions=dict(pred_a_instrument=instrument,pred_b_q2_fidelity=instrument and all(r['fidelity'] for r in q2),pred_c_q2_selectivity=instrument and all(r['selectivity'] for r in q2)),seconds=time.perf_counter()-tic)
 payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=out.name);out.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','max_replay','seconds']}))
if __name__=='__main__':main()
