#!/usr/bin/env python3
# BQGATE:384bodyforwards;48prefixes<=32tokens;180seconds;no fitting.
"""pred_a native/self/analytic serial effects<=1e-4relative; native capability.
pred_b EACHtemplate single/joint donor shift>=20/24positive,joint>=50%nativecue;
unrelated meanabs<=.5regional eacharm.
pred_c EACHinformative template directmixed serial effect error<=10%relative.
Null: zero-removal success fails signed donor-variable interchange.
Price384bodyforwards48rows8arms,180seconds,no fitting; pair-local cache.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from compiled_scalar_producers_v1 import head_scalar
from directional_mlp_bridge_v1 import execute
from regional_cue_row_check_v1 import validate
STEM='SCALAR_PRODUCERS_INTERCHANGE_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/'SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_ROWS.json').read_text())['rows'];validate(rows)
 donors=json.loads((P/(STEM+'_DONORS.json')).read_text())['donors'];assert len(rows)==len(donors)==48 and all(d['recipient']==i and d['donor']==i^1 and d['self_donor']==i for i,d in enumerate(donors))
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('384bodyforwards48rows; frozen scalar donor interchange and signed analytic bridge');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();tic=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();producer={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()}
 program={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCER_DIRECTIONAL_MLP_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()}
 writers=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['writers'].cuda()
 context={};native={};values=torch.zeros(48,8,2,dtype=torch.float64);count=0;eps=torch.finfo(torch.float32).eps;amplitude_ratios=[]
 def before8(module,args):context['r8']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def after8(module,args,out):
  scalar=head_scalar(args[0],context['tokens'],producer,0);i=context['row'];arm=context['arm']
  if arm==0:native[i].update(z8=context['r8']+out[0],a8=scalar)
  if arm in (1,3,4,5,6,7):
   donor=i if arm==7 else i^1;amplitude=scalar-native[donor]['a8'];return out[0]-(amplitude[...,None]*writers[0]).to(out[0].dtype),out[1]
  return out
 def block8out(module,args,out):
  if context['arm']==0:
   n=native[context['row']];n['u8']=out[0]-n['z8']-module.mlp.Down_bias
 def before9(module,args):
  i=context['row'];arm=context['arm'];n=native[i]
  if arm==0:n['r9']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];return
  if arm not in (5,6) or 'exact9' in n:return
  z=n['z8'].double();a=(n['a8']-native[i^1]['a8'])[...,None];d=program['direction'];J=program['mixed_map'];rho=(z-a*d).square().mean(-1,keepdim=True)+eps
  exact=execute(z,n['u8'].double(),a,program);mixed=-a*d-a/rho*(z@J.T)
  amplitude_ratios.extend((a[...,0].abs()*d.norm()/z.norm(dim=-1)).flatten().cpu().tolist())
  for name,delta in (('exact9',exact),('mixed9',mixed)):
   x=F.rms_norm((n['r9'].double()+module.lambdas[0].double()*delta).float(),(1152,),eps=eps);n[name]=head_scalar(x,context['tokens'],producer,1)
 def after9(module,args,out):
  scalar=head_scalar(args[0],context['tokens'],producer,1);i=context['row'];arm=context['arm']
  if arm==0:native[i]['a9']=scalar;return out
  if arm==1:return out
  if arm in (2,3):target=native[i^1]['a9']
  elif arm in (4,7):target=native[i]['a9']
  elif arm==5:target=native[i]['exact9']
  else:target=native[i]['mixed9']
  return out[0]+((target-scalar)[...,None]*writers[1]).to(out[0].dtype),out[1]
 handles=[model.transformer.h[8].register_forward_pre_hook(before8),model.transformer.h[8].attn.register_forward_hook(after8),model.transformer.h[8].register_forward_hook(block8out),model.transformer.h[9].register_forward_pre_hook(before9),model.transformer.h[9].attn.register_forward_hook(after9)]
 def forward(i,arm):
  nonlocal count
  row=rows[i];tokens=torch.tensor([row['ids']],device='cuda');context.update(tokens=tokens,row=i,arm=arm);x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
  for block in model.transformer.h:x,v1=block(x,v1,x0)
  count+=1;logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
  values[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();values[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 try:
  for pair in range(0,48,2):
   native.clear()
   for i in (pair,pair+1):native[i]={};forward(i,0)
   for i in (pair,pair+1):
    for arm in range(1,8):forward(i,arm)
 finally:
  for h in handles:h.remove()
 assert count==384
 old=torch.load(P/'SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['margins'][:,0]
 def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 cells=[]
 for family in range(2):
  ix=[i for i,r in enumerate(rows) if r['family']==family];m=values[ix,:,0];contrast=m[::2,0]-m[1::2,0];effect=m-m[:,:1];sign=torch.tensor([donors[i]['desired_margin_sign'] for i in ix],dtype=torch.float64);directed=effect*sign[:,None]
  reference=m[:,1]-m[:,4];refnorm=float(reference.norm());informative=refnorm>=1e-5
  exact=rel(m[:,5]-m[:,4],reference);mixed=rel(m[:,6]-m[:,4],reference);exact_abs=float((m[:,5]-m[:,1]).abs().max())
  baseline=rel(values[ix,0],old[ix]);selferr=rel(values[ix,7],values[ix,0]);selfabs=float((values[ix,7]-values[ix,0]).abs().max())
  cap=float(contrast.mean())>=.2 and int((contrast>0).sum())>=10
  signs=[int((directed[:,a]>0).sum()) for a in (1,2,3)];fraction=float(directed[:,3].mean()/contrast.mean());controls=[float((values[ix,a,1]-values[ix,0,1]).abs().mean()/effect[:,a].abs().mean().clamp_min(1e-30)) for a in (1,2,3)]
  cells.append(dict(family=family,native_capability=cap,baseline_replay=baseline,self_donor_replay=selferr,self_donor_maxabs=selfabs,native_mean_contrast=float(contrast.mean()),serial_effect_norm=refnorm,serial_informative=informative,exact_serial_effect_error=exact,exact_serial_maxabs=exact_abs,directmixed_serial_effect_error=mixed,positive_directed_rows=signs,joint_native_transfer_fraction=fraction,unrelated_to_regional=controls,mean_directed_by_arm=directed.mean(0).tolist(),individual_sum_joint_error=rel(effect[:,1]+effect[:,2],effect[:,3]),instrument=baseline<=1e-4 and selferr<=1e-4 and (exact<=1e-4 if informative else exact_abs<=1e-6),interchange_pass=min(signs)>=20 and fraction>=.5 and max(controls)<=.5))
 A=all(c['native_capability'] and c['instrument'] for c in cells)
 result={'pred_a':A,'pred_b':A and all(c['interchange_pass'] for c in cells),'pred_c':A and all(c['serial_informative'] and c['directmixed_serial_effect_error']<=.1 for c in cells),'cells':cells,'arms':['native','swap8','swap9','swap_joint','swap8_frozen9','swap8_exact9','swap8_mixed9','self_swap_joint'],'amplitude_ratio_min_median_max':[min(amplitude_ratios),float(torch.tensor(amplitude_ratios).median()),max(amplitude_ratios)],'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Explicit single/joint donor scalar interchange, signed analytic interaction on known contexts. Native background/input dependencies and newline collateral remain; not newOOD or full circuit completion.','source_shas':binding}
 torch.save(dict(margins=values),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
