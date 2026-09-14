#!/usr/bin/env python3
# BQGATE:360bodyforwards;120prefixes<=248tokens;180seconds;no fitting.
"""pred_a cached states<=1e-4 and producer/Gram identities<=1e-10.
pred_b native relative error within20pct of frozen synthetic mean.
pred_c normalized Gram cosine>=.90 and all3 offdiagonal signs agree.
pred_d five fixed24-row groups within35pct full error and Gram signs stable.
Null: the synthetic Gaussian law misweights the native retained producer.
Price360bodyforwards+360extraMLP17;120prefixes;12outputs;180seconds;no fitting.
"""
import importlib.util,json,os,signal,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import numpy as np
import torch
import torch.nn.functional as F
from additive_head_raw_ports_v1 import EPS,additive,from_projections,project
from retained_contraction_error_control_v1 import components
from three_group_shared_dag_v1 import execute
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate

STEM='NATIVE_RETAINED_LAW_FIDELITY_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(digest(ROOT/k)==v for k,v in binding.items())
 rows=json.loads((P/'MINIMAX_FRESH_CACHE_V1_ROWS.json').read_text())['rows'];validate(rows);assert len(rows)==120
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print('360bodyforwards;120prefixes; native retained-law fidelity; no fitting');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter()
 direction=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda()
 spec=importlib.util.spec_from_file_location('freshhierarchy',P/'extracted_circuits/crossfirst_state_executor_v1/hierarchy.py')
 hier=importlib.util.module_from_spec(spec);spec.loader.exec_module(hier);weights=hier.executor.load_weights(model.state_dict(),'cuda')
 child={};parent={};ctx={};count=0
 W=model.transformer.h[17].attn.c_proj.weight.double().reshape(1152,9,128)[:,2]
 maps=tuple(model.state_dict()['transformer.h.17.attn.'+k+'.weight'].reshape(9,128,1152)[2].double() for k in ('c_q','c_k','c_q2','c_k2','c_v'))
 mix=float(model.transformer.h[17].attn.lamb)
 pairs=sorted(set((r['uk_id'],r['us_id']) for r in json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows']))
 ids=[i for pair in pairs for i in pair];assert len(set(ids))==12
 O=model.lm_head.weight[ids].double();mlp=model.transformer.h[17].mlp
 L,R,D=(getattr(mlp,k).weight.double() for k in ('Left','Right','Down'));C=O@D;LW,RW=L@W,R@W
 exact=torch.stack([L.T@(c[:,None]*RW)+R.T@(c[:,None]*LW) for c in C])
 package=torch.load(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
 mask=torch.from_numpy(np.unpackbits(package['mask'].numpy(),bitorder='little',count=exact.numel()).copy()).bool()
 core=torch.zeros(exact.numel(),dtype=torch.float64);core[mask]=package['values'].double()
 fitted=torch.einsum('op,pih,ah->oia',package['output'].double(),core.reshape(package['shape']),package['head'].double()).cuda()
 error=fitted-exact
 cache=torch.load(P/'MINIMAX_FRESH_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 branch_outputs=[];exact_outputs=[];linear=[];states=[];producer_replays=[]
 def write9(module,args,output):
  arm=ctx['arm'];i=ctx['i']
  if arm==0:
   parts=hier.split_fields(ctx['ids'],ctx['x7'],ctx['x8'],args[0],ctx['R8'],ctx['rho9'],weights);child[i]=parts['child'];parent[i]=parts['parent'];return output
  amplitude=child[i] if arm==1 else parent[i]-child[i]
  return output[0]-(amplitude[...,None]*direction).to(output[0].dtype),output[1]
 def input7(module,args):
  if ctx['arm']==0:ctx['x7']=args[0]
 def pre8(module,args):
  if ctx['arm']==0:ctx['raw8']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def input8(module,args):
  if ctx['arm']==0:ctx['x8']=args[0]
 def after8(module,args,output):
  if ctx['arm']==0:
   z=ctx['raw8']+output[0];ctx['R8']=z.square().mean(-1).double()+EPS
 def pre9(module,args):
  if ctx['arm']==0:
   z=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];ctx['rho9']=(z.square().mean(-1)+EPS).sqrt().double()
 def pre17(module,args):
  raw=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];ctx['raw17']=raw
  ctx['raw_inputs'][ctx['arm']]=raw.double()
 def first_values(module,args,output):ctx['first_values']=output.reshape(1,-1,9,128)[:,:,2].double()
 def after_attention17(module,args,output):ctx['z17'][ctx['arm']]=(ctx['raw17']+output[0])[:,-1].double()
 handles=[model.transformer.h[0].attn.c_v.register_forward_hook(first_values),model.transformer.h[7].mlp.register_forward_pre_hook(input7),model.transformer.h[8].register_forward_pre_hook(pre8),model.transformer.h[8].attn.register_forward_pre_hook(input8),model.transformer.h[8].attn.register_forward_hook(after8),model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(write9),model.transformer.h[17].register_forward_pre_hook(pre17),model.transformer.h[17].attn.register_forward_hook(after_attention17)]
 try:
  for i,row in enumerate(rows):
   ids0=torch.tensor([row['ids']],device='cuda');ctx.update(i=i,ids=ids0,z17={},raw_inputs={})
   for arm in (0,1,3):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids0),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    count+=1
   raw=ctx['raw_inputs'];corners=tuple(raw[k] for k in (0,1,3));dc=corners[1]-corners[0];dr=corners[2]-corners[0]
   projections=tuple(project(x,maps) for x in corners);norms=tuple(x.square().mean(-1)+EPS for x in corners)
   added_norm=norms[1]+norms[2]-norms[0]+2*(dc*dr).mean(-1)
   ports=tuple(from_projections(p,n,ctx['first_values'],mix) for p,n in zip(projections,norms))
   ports+=(from_projections(additive(*projections),added_norm,ctx['first_values'],mix),)
   terms=components(*ports);producer=execute(*ports);producer_replays.append(float((terms.sum(1)-producer).norm()/producer.norm().clamp_min(1e-30)))
   z0=ctx['z17'][1]+ctx['z17'][3]-ctx['z17'][0];z1=z0+producer@W.T
   final=z1.float()+mlp(F.rms_norm(z1.float(),(1152,)))
   denom=(z1.square().mean(-1)+EPS)*(final.double().square().mean(-1)+EPS).sqrt()
   branch_outputs.append((torch.einsum('oih,ni,nkh->nko',error,z0,terms)/denom[:,None,None]).cpu())
   exact_outputs.append((torch.einsum('oih,ni,nh->no',exact,z0,producer)/denom[:,None]).cpu())
   linear.append(z1.cpu());states.append(final.double().cpu())
 finally:
  for h in handles:h.remove()
 branches=torch.cat(branch_outputs);target=torch.cat(exact_outputs);linear=torch.cat(linear);states=torch.cat(states)
 replay_linear=float((linear-cache['linear_parts'][:,1].double()).norm()/cache['linear_parts'][:,1].double().norm())
 replay_state=float((states-cache['states'][:,1].double()).norm()/cache['states'][:,1].double().norm())
 def metrics(indices):
  b=branches[indices];t=target[indices];gram=torch.einsum('nko,nlo->kl',b,b)/len(b);energy=float(b.sum(1).square().sum()/len(b))
  return gram,energy,float(b.sum(1).norm()/t.norm())
 gram,energy,relative=metrics(slice(None));gram_replay=abs(float(gram.sum())-energy)/max(energy,1e-30)
 synthetic=json.loads((P/'NATIVE_RETAINED_ERROR_V1_RESULT.json').read_text());synthetic_relative=sum(c['relative_mixed_error'] for c in synthetic['cells'])/len(synthetic['cells'])
 synthetic_gram=torch.tensor(synthetic['cells'][0]['gram'],dtype=torch.float64)
 synthetic_gram=sum((torch.tensor(c['gram'],dtype=torch.float64) for c in synthetic['cells'][1:]),synthetic_gram)/len(synthetic['cells'])
 sn=synthetic_gram/float(synthetic_gram.sum());nn=gram/energy
 cosine=float((sn.flatten()@nn.flatten())/(sn.norm()*nn.norm()))
 off=[(0,1),(0,2),(1,2)];sg=[int(torch.sign(sn[a,b])) for a,b in off];ng=[int(torch.sign(nn[a,b])) for a,b in off]
 groups=[]
 for j in range(5):
  gg,ee,rr=metrics(slice(24*j,24*(j+1)));signs=[int(torch.sign(gg[a,b])) for a,b in off]
  groups.append({'group':j,'relative_mixed_error':rr,'relative_to_full':rr/relative,'offdiagonal_signs':signs,'signs_stable':signs==ng})
 result={'pred_a':count==360 and max(replay_linear,replay_state,max(producer_replays),gram_replay)<=1e-4 and max(max(producer_replays),gram_replay)<=1e-10,
  'pred_b':abs(relative/synthetic_relative-1)<=.2,
  'pred_c':cosine>=.90 and ng==sg,
  'pred_d':all(abs(g['relative_to_full']-1)<=.35 and g['signs_stable'] for g in groups),
  'body_forwards':count,'rows':len(rows),'cached_linear_replay':replay_linear,'cached_state_replay':replay_state,'max_producer_replay':max(producer_replays),'complete_gram_replay':gram_replay,
  'synthetic_relative_mixed_error':synthetic_relative,'native_relative_mixed_error':relative,'native_over_synthetic':relative/synthetic_relative,
  'normalized_gram_cosine':cosine,'synthetic_offdiagonal_signs':sg,'native_offdiagonal_signs':ng,'native_gram':gram.tolist(),'native_complete_energy':energy,'groups':groups,'seconds':time.perf_counter()-tic,'source_shas':binding,
  'scope':'Frozen sparse-minus-exact setting2 operator on120 native three-trajectory circuit rows with complete retained producer and actual compact background/normalizers. Diagnostic only: no fit, OOD, selectivity, static compression or adoption claim.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','native_gram')},indent=2));signal.alarm(0)

if __name__=='__main__':main()
