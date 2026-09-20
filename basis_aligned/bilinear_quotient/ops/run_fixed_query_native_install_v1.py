#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_q2_fidelity pred_c_q2_selectivity
"""Finite baseline Q2 reuse; FIXED_QUERY_NATIVE_INSTALL_V1_PREREGISTRATION.md."""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';PLAN=dict(prefix=12,native_suffix=16,baseline_attention=4)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch,tiktoken
 import torch.nn.functional as F
 import circuit_fast_screen_producer as producer
 from disk_guard import guard_write
 sys.path.insert(0,str(P))
 from refined_native_sources import build_refined
 from native_fixed_query_source import prepare,evaluate
 from native_source_observables import source_observables32
 out=A/'fixed_query_native_install_v1_result.json';assert not out.exists()
 priorpath=A/'refined_selective_sources_v1_result.json';prior=json.loads(priorpath.read_text());binding=json.loads((P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json').read_text());extra=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text())
 torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model;enc=tiktoken.get_encoding('gpt2');modal=torch.tensor([[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]+extra['token_ids'],device='cuda');counts={k:0 for k in PLAN};replays=[];records=[];compiled_checks=[];joint=json.loads((A/'attention_query_composition_v1_result.json').read_text())
 for context in binding['contexts']:
  c=build_refined(model,context,modal);counts['prefix']+=c['prefix_calls'];n=len(c['raw']);att=model.transformer.h[11].attn
  z=F.rms_norm(c['raw'],(c['raw'].shape[-1],));background=prepare(att,z,c['first']);counts['baseline_attention']+=1
  original=att.forward
  def native(role,aa,compiled=False):
   counts['native_suffix']+=1;pos,ds=c['source_components'][role]
   if compiled:att.forward=lambda normalized,v1=None:(evaluate(att,normalized,pos,background),v1)
   try:return source_observables32(model,c['raw'],c['x0'],c['first'],ds,pos,c['read'],c['pairs'],torch.as_tensor(aa,device='cuda',dtype=torch.float64))
   finally:att.forward=original
  zeros=torch.zeros(n,23,device='cuda');baseline=native('subject',zeros);replays.append(float((baseline-native('subject',zeros,True)).abs().max()))
  for role in ['subject','attractor']:
   aa=next(v for v in prior['amplitudes'] if v['panel']==context['panel'] and v['template']==context['template'] and v['role']==role)['amplitudes']['oracle']
   effects={'compiled':baseline-native(role,aa,True)}
   families=[r['family'] for r in c['entries']]
   for family in dict.fromkeys(families):
    ids=[i for i,f in enumerate(families) if f==family]
    old=next(r for r in prior['records'] if r['panel']==context['panel'] and r['role']==role and r['family']==family and r['arm']=='oracle');full=torch.tensor(old['effect'],device='cuda',dtype=torch.float64);den=full[:,0].norm().clamp_min(1e-30)
    unit=next(r for r in prior['records'] if r['panel']==context['panel'] and r['role']==role and r['family']==family and r['arm']=='unitB');reference=torch.tensor(unit['effect'],device='cuda',dtype=torch.float64)[:,0]
    for name,effect in effects.items():
     v=effect[ids];target=torch.tensor(next(r for r in joint['records'] if r['panel']==context['panel'] and r['role']==role and r['family']==family and r['arm']=='freeze_both')['effect'],device='cuda',dtype=torch.float64);compiled_checks.append(dict(panel=context['panel'],role=role,family=family,max_absolute=float((v-target).abs().max()),relative_errors=((v-target).norm(dim=0)/den).tolist()));error=(v-full).norm(dim=0)/den;ret=float(v[:,0]@reference/reference.square().sum());leak=float(v[:,1:].norm(dim=0).max()/v[:,0].norm().clamp_min(1e-30))
     records.append(dict(panel=context['panel'],family=family,role=role,arm=name,errors=error.tolist(),retention=ret,control_ratio=leak,effect=v.tolist(),fidelity=bool(error[0]<=.1 and error[1:].max()<=.05),selectivity=bool(ret>=.8 and leak<=.1)))
 instrument=counts==PLAN and max(replays)<=1e-4 and max(max(r['relative_errors']) for r in compiled_checks)<=.001;q2=records
 result=dict(compiled_checks=compiled_checks,plan=PLAN,counts=counts,prior_sha256=hashlib.sha256(priorpath.read_bytes()).hexdigest(),max_replay=max(replays),records=records,predictions=dict(pred_a_instrument=instrument,pred_b_q2_fidelity=instrument and all(r['fidelity'] for r in q2),pred_c_q2_selectivity=instrument and all(r['selectivity'] for r in q2)),seconds=time.perf_counter()-tic)
 payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=out.name);out.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','max_replay','seconds']}))
if __name__=='__main__':main()
