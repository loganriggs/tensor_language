#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_weak_composition pred_c_joint_selectivity
"""V4_SELECTIVE_COMPOSITION_V1_PREREGISTRATION.md; unchanged oracle source edits."""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';PLAN=dict(prefix=12,native_suffix=20)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch,tiktoken
 import circuit_fast_screen_producer as producer
 from disk_guard import guard_write
 sys.path.insert(0,str(P))
 from refined_native_sources import build_refined,PARENTS
 from native_source_observables import source_observables32
 out=A/'v4_selective_composition_v1_result.json';assert not out.exists();choicepath=P/'V4_FINITE_REFERENCE_CHOICES_V1.json';choices=json.loads(choicepath.read_text());prior=json.loads((A/'v4_finite_reference_v1_result.json').read_text());contexts=[]
 for panel in ['opposite','congruent']:
  file=f'SOURCE_OOD_V4_{panel.upper()}_ROWS.json';rows=json.loads((P/file).read_text())
  for template in dict.fromkeys(r['template'] for r in rows):
   n=sum(r['template']==template for r in rows);contexts.append(dict(panel=panel,template=template,rows_file=file,amplitudes={r:[[0,0,1,1,1,0]]*n for r in ['subject','attractor']}))
 torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model;enc=tiktoken.get_encoding('gpt2');controls=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text());modal=torch.tensor([[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]+controls['token_ids'],device='cuda');reference=torch.tensor([0.,0.,1.,1.,1.,0.],device='cuda',dtype=torch.float64)[PARENTS];counts={k:0 for k in PLAN};replays=[];records=[]
 for context in contexts:
  c=build_refined(model,context,modal);counts['prefix']+=c['prefix_calls'];n,t,d=c['raw'].shape;families=[r['family'] for r in c['entries']];weights={role:torch.tensor(next(v for v in choices['records'] if all(v[k]==context[k] for k in ['panel','template']) and v['role']==role)['amplitudes'],device='cuda',dtype=torch.float64) for role in ['subject','attractor']}
  def native(s,a,unit=False):
   counts['native_suffix']+=1;raw=c['raw'].clone()
   for role,alpha in [('subject',s),('attractor',a)]:
    pos,ds=c['source_components'][role];aa=reference[None].expand(n,-1) if unit else weights[role];delta=torch.einsum('bk,bkd->bd',aa,ds)
    raw[c['batch'],pos]=(raw[c['batch'],pos].double()+alpha*delta).float()
   return source_observables32(model,raw,c['x0'],c['first'],torch.zeros(n,1,d,device='cuda'),torch.zeros(n,device='cuda',dtype=torch.long),c['read'],c['pairs'],torch.zeros(n,1,device='cuda'))
  base=native(0,0);es=base-native(1,0);ea=base-native(0,1);joint=base-native(1,1);ref=base-native(1,1,True)
  for family in dict.fromkeys(families):
   ids=[i for i,f in enumerate(families) if f==family];ss,aa,jj,rr=[v[ids] for v in [es,ea,joint,ref]]
   for role,effect in [('subject',ss),('attractor',aa)]:
    old=next(v for v in prior['records'] if v['panel']==context['panel'] and v['family']==family and v['role']==role);replays.append(float((effect-torch.tensor(old['effect'],device='cuda',dtype=torch.float64)).abs().max()))
   interaction=jj-ss-aa;sn,an=ss[:,0].norm(),aa[:,0].norm();weak=torch.minimum(sn,an).clamp_min(1e-30);err=interaction.norm(dim=0)/weak;den=jj[:,0].norm().clamp_min(1e-30);ret=float(jj[:,0]@rr[:,0]/rr[:,0].square().sum().clamp_min(1e-30));leak=float(jj[:,1:].norm(dim=0).max()/den)
   records.append(dict(panel=context['panel'],family=family,weak_errors=err.tolist(),subject_increment_errors=(interaction.norm(dim=0)/sn.clamp_min(1e-30)).tolist(),attractor_increment_errors=(interaction.norm(dim=0)/an.clamp_min(1e-30)).tolist(),subject_norm=float(sn),attractor_norm=float(an),joint_norm=float(den),reference_norm=float(rr[:,0].norm()),retention=ret,control_ratio=leak,subject=ss.tolist(),attractor=aa.tolist(),joint=jj.tolist(),reference=rr.tolist(),interaction=interaction.tolist()))
 instrument=counts==PLAN and max(replays)<=1e-8
 result=dict(choices_sha256=hashlib.sha256(choicepath.read_bytes()).hexdigest(),plan=PLAN,counts=counts,max_replay=max(replays),records=records,predictions=dict(pred_a_instrument=instrument,pred_b_weak_composition=instrument and all(r['weak_errors'][0]<=.1 and max(r['weak_errors'][1:])<=.05 for r in records),pred_c_joint_selectivity=instrument and all(r['retention']>=.8 and r['control_ratio']<=.1 for r in records)),seconds=time.perf_counter()-tic)
 payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=out.name);out.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','max_replay','seconds']}))
if __name__=='__main__':main()
