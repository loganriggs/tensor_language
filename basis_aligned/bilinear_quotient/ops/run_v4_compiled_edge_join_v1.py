#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_weak_composition pred_c_joint_selectivity
"""V4_COMPILED_EDGE_JOIN_V1_PREREGISTRATION.md; unchanged oracle source edits."""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';PLAN=dict(prefix=12,native_suffix=16,joined_suffix=20)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch,tiktoken
 import circuit_fast_screen_producer as producer
 from disk_guard import guard_write,guard_torch_save
 sys.path.insert(0,str(P))
 from refined_native_sources import build_refined,PARENTS
 from native_source_observables import source_observables32
 from affine_state_readout import compile,evaluate
 from attention_mixed_edge_core import compile_core,mixed
 out=A/'v4_compiled_edge_join_v1_result.json';assert not out.exists();choicepath=P/'V4_FINITE_REFERENCE_CHOICES_V1.json';choices=json.loads(choicepath.read_text());prior=json.loads((A/'v4_finite_reference_v1_result.json').read_text());contexts=[];composition=json.loads((A/'v4_selective_composition_v1_result.json').read_text());field_errors=[];programs=[];cores=[]
 for panel in ['opposite','congruent']:
  file=f'SOURCE_OOD_V4_{panel.upper()}_ROWS.json';rows=json.loads((P/file).read_text())
  for template in dict.fromkeys(r['template'] for r in rows):
   n=sum(r['template']==template for r in rows);contexts.append(dict(panel=panel,template=template,rows_file=file,amplitudes={r:[[0,0,1,1,1,0]]*n for r in ['subject','attractor']}))
 torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model;enc=tiktoken.get_encoding('gpt2');controls=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text());modal=torch.tensor([[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]+controls['token_ids'],device='cuda');reference=torch.tensor([0.,0.,1.,1.,1.,0.],device='cuda',dtype=torch.float64)[PARENTS];counts={k:0 for k in PLAN};replays=[];records=[]
 for context in contexts:
  c=build_refined(model,context,modal);counts['prefix']+=c['prefix_calls'];n,t,d=c['raw'].shape;families=[r['family'] for r in c['entries']];weights={role:torch.tensor(next(v for v in choices['records'] if all(v[k]==context[k] for k in ['panel','template']) and v['role']==role)['amplitudes'],device='cuda',dtype=torch.float64) for role in ['subject','attractor']}
  def native(s,a,unit=False,capture=None):
   counts['native_suffix']+=1;raw=c['raw'].clone()
   for role,alpha in [('subject',s),('attractor',a)]:
    pos,ds=c['source_components'][role];aa=reference[None].expand(n,-1) if unit else weights[role];delta=torch.einsum('bk,bkd->bd',aa,ds)
    raw[c['batch'],pos]=(raw[c['batch'],pos].double()+alpha*delta).float()
   return source_observables32(model,raw,c['x0'],c['first'],torch.zeros(n,1,d,device='cuda'),torch.zeros(n,device='cuda',dtype=torch.long),c['read'],c['pairs'],torch.zeros(n,1,device='cuda'),capture=capture)
  caps=[{'all_blocks':True} for _ in range(3)];base=native(0,0,capture=caps[0]);ys=native(1,0,capture=caps[1]);ya=native(0,1,capture=caps[2]);es=base-ys;ea=base-ya
  states=[v['final_state'].double() for v in caps];basis=torch.stack([states[0],states[1]-states[0],states[2]-states[0]],1);program=compile(basis,model.lm_head.weight[c['pairs']]);predicted=evaluate(program,0,0)-evaluate(program,1,1)
  numerical=torch.stack([(evaluate(program,s,a)-v).abs() for s,a,v in [(0,0,base),(1,0,ys),(0,1,ya)]]).amax(0);field_errors.append(float(numerical.max()));programs.append(dict(panel=context['panel'],template=context['template'],numerators=program[0].tolist(),gram=program[1].tolist()));joint=base-native(1,1)
  joins={}
  for boundary in [10,11,17]:
   states=[v['initial_state'] if boundary==10 else v['post_blocks'][boundary] for v in caps];combined=(states[1].double()+states[2].double()-states[0].double()).float();counts['joined_suffix']+=1
   joins[boundary]=base-source_observables32(model,combined,c['x0'],c['first'],torch.zeros(n,1,d,device='cuda'),torch.zeros(n,device='cuda',dtype=torch.long),c['read'],c['pairs'],torch.zeros(n,1,device='cuda'),start_layer=boundary+1)
  states=[v['first_mlp_input'] for v in caps];combined=(states[1].double()+states[2].double()-states[0].double()).float();combined=combined+model.transformer.h[11].mlp(torch.nn.functional.rms_norm(combined,(d,)));counts['joined_suffix']+=1
  joins[10.5]=base-source_observables32(model,combined,c['x0'],c['first'],torch.zeros(n,1,d,device='cuda'),torch.zeros(n,device='cuda',dtype=torch.long),c['read'],c['pairs'],torch.zeros(n,1,device='cuda'),start_layer=12)
  att=model.transformer.h[11].attn;pos_s=c['source_components']['subject'][0];pos_a=c['source_components']['attractor'][0];query=torch.maximum(pos_s,pos_a);keypos=torch.minimum(pos_s,pos_a);assert bool((query>keypos).all())
  initial=[v['initial_state'].double() for v in caps];delta=initial[1]+initial[2]-2*initial[0];batch=c['batch']
  cos,sin=att.rotary(torch.zeros(n,t,att.n_head,att.head_dim,device='cuda',dtype=torch.float32))
  native_weights={name:getattr(att,attr).weight.double() for name,attr in [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]}
  core=compile_core(native_weights,initial[0][batch,query],delta[batch,query],initial[0][batch,keypos],delta[batch,keypos],torch.eye(d,device='cuda',dtype=torch.float64)[None].expand(n,-1,-1),c['first'].reshape_as(c['raw'])[batch,keypos].double(),att.lamb.double(),att.n_head,cos[0,query,0].double(),sin[0,query,0].double(),cos[0,keypos,0].double(),sin[0,keypos,0].double(),torch.finfo(torch.float32).eps,torch.finfo(torch.float32).eps)
  edge=mixed(core,1.,1.);states=[v['first_mlp_input'].double() for v in caps];combined=states[1]+states[2]-states[0];combined[batch,query]+=edge;combined=combined.float();combined=combined+model.transformer.h[11].mlp(torch.nn.functional.rms_norm(combined,(d,)));counts['joined_suffix']+=1
  joins[10.6]=base-source_observables32(model,combined,c['x0'],c['first'],torch.zeros(n,1,d,device='cuda'),torch.zeros(n,device='cuda',dtype=torch.long),c['read'],c['pairs'],torch.zeros(n,1,device='cuda'),start_layer=12)
  cores.append(dict(panel=context['panel'],template=context['template'],families=families,core={k:v.cpu() for k,v in core.items()},edge=edge.cpu(),query=query.cpu(),key=keypos.cpu(),coefficients_per_input=sum(v.numel() for v in core.values())//n))
  replays.append(float((joins[10]-joint).abs().max()))
  for family in dict.fromkeys(families):
   ids=[i for i,f in enumerate(families) if f==family];ss,aa,jj=[v[ids] for v in [es,ea,joint]];oldjoint=next(r for r in composition['records'] if r['panel']==context['panel'] and r['family']==family);rr=torch.tensor(oldjoint['reference'],device='cuda',dtype=torch.float64);replays.append(float((jj-torch.tensor(oldjoint['joint'],device='cuda',dtype=torch.float64)).abs().max()))
   for role,effect in [('subject',ss),('attractor',aa)]:
    old=next(v for v in prior['records'] if v['panel']==context['panel'] and v['family']==family and v['role']==role);replays.append(float((effect-torch.tensor(old['effect'],device='cuda',dtype=torch.float64)).abs().max()))
   sn,an=ss[:,0].norm(),aa[:,0].norm();weak=torch.minimum(sn,an).clamp_min(1e-30)
   field_errors.append(float((joins[17][ids]-predicted[ids]).norm(dim=0).max()/weak))
   for boundary,estimate in joins.items():
    errors=(estimate[ids]-jj).norm(dim=0)/weak
    records.append(dict(panel=context['panel'],family=family,boundary=boundary,remaining_blocks=6 if boundary in [10.5,10.6] else 17-boundary,first_mlp=boundary in [10.5,10.6],errors=errors.tolist(),weak_norm=float(weak),passed=bool(errors[0]<=.1 and errors[1:].max()<=.05)))
 instrument=counts==PLAN and max(replays)<=1e-8 and max(field_errors)<=.001
 passing=[boundary for boundary in [10,10.5,10.6,11,17] if all(r['passed'] for r in records if r['boundary']==boundary)]
 result=dict(plan=PLAN,counts=counts,max_replay=max(replays),max_field_check=max(field_errors),records=records,passing_boundaries=passing,predictions=dict(pred_a_instrument=instrument,pred_b_compiled_join=instrument and 10.6 in passing,pred_c_precise_join=instrument and max(max(r['errors']) for r in records if r['boundary']==10.6)<=.001),seconds=time.perf_counter()-tic)
 guard_torch_save(dict(groups=cores,scope='Conditional mixed edge, no native source or suffix closure'),str(A/'v4_compiled_edge_join_v1_cores.pt'))
 payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=out.name);out.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','max_replay','seconds']}))
if __name__=='__main__':main()
