#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_weak_composition pred_c_joint_selectivity
"""V4_EDGE_HEAD3_TRANSFER_V1_PREREGISTRATION.md; unchanged oracle source edits."""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';PLAN=dict(prefix=12,native_suffix=64,joined_suffix=140)
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
 from mixed_edge_writer_features import writers,amplitudes
 from prepared_linear_mlp_response import compile as compile_linear,evaluate as evaluate_linear
 out=A/'v4_edge_head3_transfer_v1_result.json';assert not out.exists();choicepath=P/'V4_FINITE_REFERENCE_CHOICES_V1.json';choices=json.loads(choicepath.read_text());prior=json.loads((A/'v4_finite_reference_v1_result.json').read_text());contexts=[];composition=json.loads((A/'v4_selective_composition_v1_result.json').read_text());field_errors=[];programs=[];cores=[];local_errors=[];export=None
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
  cache={}
  def sample(s,a):
   key=(s,a)
   if key not in cache:
    capture={'all_blocks':True};y=native(s,a,capture=capture);cache[key]=(y,capture)
   return cache[key]
  base,cb=sample(0,0);ys,cs=sample(1,0);ya,ca=sample(0,1);caps=[cb,cs,ca]
  att=model.transformer.h[11].attn;pos_s=c['source_components']['subject'][0];pos_a=c['source_components']['attractor'][0];query=torch.maximum(pos_s,pos_a);keypos=torch.minimum(pos_s,pos_a);assert bool((query>keypos).all())
  initial=[v['initial_state'].double() for v in caps];delta=initial[1]+initial[2]-2*initial[0];batch=c['batch']
  cos,sin=att.rotary(torch.zeros(n,t,att.n_head,att.head_dim,device='cuda',dtype=torch.float32))
  native_weights={name:getattr(att,attr).weight.double() for name,attr in [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]}
  core=compile_core(native_weights,initial[0][batch,query],delta[batch,query],initial[0][batch,keypos],delta[batch,keypos],torch.eye(d,device='cuda',dtype=torch.float64)[None].expand(n,-1,-1),c['first'].reshape_as(c['raw'])[batch,keypos].double(),att.lamb.double(),att.n_head,cos[0,query,0].double(),sin[0,query,0].double(),cos[0,keypos,0].double(),sin[0,keypos,0].double(),torch.finfo(torch.float32).eps,torch.finfo(torch.float32).eps)

  mlp=model.transformer.h[11].mlp
  for sv,av in [(1.,1.),(.5,1.),(1.,.5),(-1.,1.),(1.,-1.),(2.,1.),(1.,2.)]:
   ys,cs=sample(sv,0);ya,ca=sample(0,av);yj,cj=sample(sv,av);es=base-ys;ea=base-ya;joint=base-yj
   u=torch.where(pos_s>pos_a,sv,av);v=torch.where(pos_s>pos_a,av,sv);edge=mixed(core,u,v)
   additive=(cs['first_mlp_input'].double()+ca['first_mlp_input'].double()-cb['first_mlp_input'].double()).float()
   corrected=additive.double();corrected[batch,query]+=edge
   # This local anchor checks the frozen core at new signed/strengthened amplitudes.
   local_errors.append(float((corrected-cj['first_mlp_input'].double()).norm()/(cj['first_mlp_input'].double()-additive.double()).norm().clamp_min(1e-30)))
   bg=additive+mlp(torch.nn.functional.rms_norm(additive,(d,)))
   carry=bg.clone();carry[batch,query]=(carry[batch,query].double()+edge).float()
   exact=corrected.float();exact=exact+mlp(torch.nn.functional.rms_norm(exact,(d,)))
   w=writers(core);beta=amplitudes(core,u,v);g=compile_linear(mlp.Left.weight.double(),mlp.Right.weight.double(),mlp.Down.weight.double(),additive[batch,query].double(),w,torch.finfo(torch.float32).eps)
   linear=bg.clone();linear[batch,query]=(linear[batch,query].double()+evaluate_linear(g,beta)).float()
   estimates={}
   posts=[('carry',carry),('exact',exact),('linear',linear)]
   chosen=(w*beta[:,None,:]).reshape(n,d,3,9).sum(2)[:,:,3]
   for label,update in [('head3',chosen),('without3',edge-chosen)]:
    post=bg.clone();post[batch,query]=(post[batch,query].double()+update).float();posts.append((label,post))
   for label,post in posts:
    counts['joined_suffix']+=1
    estimates[label]=base-source_observables32(model,post,c['x0'],c['first'],torch.zeros(n,1,d,device='cuda'),torch.zeros(n,device='cuda',dtype=torch.long),c['read'],c['pairs'],torch.zeros(n,1,device='cuda'),start_layer=12)
   for family in dict.fromkeys(families):
    ids=[i for i,f in enumerate(families) if f==family];weak=torch.minimum(es[ids,0].norm(),ea[ids,0].norm()).clamp_min(1e-30)
    for label,estimate in estimates.items():
     error=(estimate[ids]-joint[ids]).norm(dim=0)/weak
     records.append(dict(panel=context['panel'],family=family,subject_amplitude=sv,attractor_amplitude=av,arm=label,errors=error.tolist(),weak_norm=float(weak),passed=bool(error[0]<=.1 and error[1:].max()<=.05)))
  # Unit joint replay against the previously frozen native receipt.
  for family in dict.fromkeys(families):
   ids=[i for i,f in enumerate(families) if f==family];old=next(r for r in composition['records'] if r['panel']==context['panel'] and r['family']==family)
   replays.append(float(((base-sample(1,1)[0])[ids]-torch.tensor(old['joint'],device='cuda',dtype=torch.float64)).abs().max()))
 instrument=counts==PLAN and max(replays)<=1e-8 and max(max(r['errors']) for r in records if r['arm']=='exact')<=.001
 result=dict(plan=PLAN,counts=counts,max_replay=max(replays),max_local_edge_relative_error=max(local_errors),records=records,predictions=dict(pred_a_instrument=instrument,pred_b_head3_transfer=instrument and all(r['passed'] for r in records if r['arm']=='head3'),pred_c_head3_needed=instrument and any(not r['passed'] for r in records if r['arm']=='without3')),seconds=time.perf_counter()-tic)
 payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=out.name);out.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','max_replay','seconds']}))
if __name__=='__main__':main()
