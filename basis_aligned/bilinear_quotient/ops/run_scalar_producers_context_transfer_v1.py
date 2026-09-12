#!/usr/bin/env python3
# BQGATE:336bodyforwards;48freshprefixes<=32tokens;180seconds;no fitting.
"""pred_a EACHtemplate native meancontrast>=.2,>=10/12positive;
exact serial-effect replay<=1e-4relative, scalar-change<=1e-5relative.
pred_b physical8/9/joint>=10/12cue reductions,joint>=50%nativecontrast;
unrelated meanabs<=.5regional eachphysicalarm.
pred_c EACHtemplate directmixed serial effect error<=10%relative.
Null: prior conditional component does not transfer to these new contexts.
Price336bodyforwards48rows7arms<=32tokens180seconds;no fitting.
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
STEM='SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows'];validate(rows);assert len(rows)==48 and max(len(r['ids']) for r in rows)<=32
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('336bodyforwards48fresh email/letter contexts; frozen physical pair and bridge');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();tic=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();producer={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()}
 program={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCER_DIRECTIONAL_MLP_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()}
 writers=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['writers'].cuda()
 context={};values=torch.zeros(48,7,2,dtype=torch.float64);scalar_num=[0.,0.];scalar_den=[0.,0.];counts=0;eps=torch.finfo(torch.float32).eps
 def before8(module,args):context['r8']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def after8(module,args,out):
  scalar=head_scalar(args[0],context['tokens'],producer,0)
  if context['arm']==0:context.update(z8=context['r8']+out[0],a8=scalar)
  if context['arm'] in (1,3,4,5,6):return out[0]-(scalar[...,None]*writers[0]).to(out[0].dtype),out[1]
  return out
 def block8out(module,args,out):
  if context['arm']==0:context['u8']=out[0]-context['z8']-module.mlp.Down_bias
 def before9(module,args):
  if context['arm']!=0:return
  raw=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];z=context['z8'].double();a=context['a8'][...,None];d=program['direction'];J=program['mixed_map'];rho=(z-a*d).square().mean(-1,keepdim=True)+eps
  exact=execute(z,context['u8'].double(),a,program);mixed=-a*d-a/rho*(z@J.T)
  for name,delta in (('exact9',exact),('mixed9',mixed)):
   x=F.rms_norm((raw.double()+module.lambdas[0].double()*delta).float(),(1152,),eps=eps)
   context[name]=head_scalar(x,context['tokens'],producer,1)
 def after9(module,args,out):
  scalar=head_scalar(args[0],context['tokens'],producer,1);arm=context['arm']
  if arm==0:context['native9']=scalar;return out
  if arm==1:
   family=rows[context['row']]['family'];scalar_num[family]+=float((context['exact9']-scalar).square().sum());scalar_den[family]+=float((scalar-context['native9']).square().sum());return out
  if arm==4:scalar=context['native9']
  elif arm==5:scalar=context['mixed9']
  elif arm==6:scalar=context['exact9']
  return out[0]-(scalar[...,None]*writers[1]).to(out[0].dtype),out[1]
 handles=[model.transformer.h[8].register_forward_pre_hook(before8),model.transformer.h[8].attn.register_forward_hook(after8),model.transformer.h[8].register_forward_hook(block8out),model.transformer.h[9].register_forward_pre_hook(before9),model.transformer.h[9].attn.register_forward_hook(after9)]
 try:
  for i,row in enumerate(rows):
   tokens=torch.tensor([row['ids']],device='cuda')
   for arm in range(7):
    context.update(tokens=tokens,row=i,arm=arm);x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    counts+=1;logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    values[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();values[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for handle in handles:handle.remove()
 assert counts==336
 def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 cells=[]
 for family in range(2):
  ix=[i for i,r in enumerate(rows) if r['family']==family];m=values[ix,:,0];contrast=m[::2]-m[1::2];reduction=contrast[:,:1]-contrast;effect=m-m[:,:1]
  reference=m[:,3]-m[:,4];exact=rel(m[:,6]-m[:,4],reference);mixed=rel(m[:,5]-m[:,4],reference);scalar=(scalar_num[family]/max(scalar_den[family],1e-30))**.5
  cap=float(contrast[:,0].mean())>=.2 and int((contrast[:,0]>0).sum())>=10
  directions=[int((reduction[:,a]>0).sum()) for a in (1,2,3)];fraction=float(reduction[:,3].mean()/contrast[:,0].mean());controls=[float((values[ix,a,1]-values[ix,0,1]).abs().mean()/effect[:,a].abs().mean().clamp_min(1e-30)) for a in (1,2,3)]
  cells.append(dict(family=family,native_capability=cap,native_mean_contrast=float(contrast[:,0].mean()),native_positive_pairs=int((contrast[:,0]>0).sum()),exact_serial_effect_error=exact,exact_scalar_change_error=scalar,directmixed_serial_effect_error=mixed,serial_effect_norm=float(reference.norm()),positive_reduction_pairs=directions,joint_native_contrast_fraction=fraction,unrelated_to_regional=controls,mean_cue_reduction_by_arm=reduction.mean(0).tolist(),physical_joint_nonadditivity=rel(effect[:,1]+effect[:,2],effect[:,3]),removal_pass=min(directions)>=10 and fraction>=.5 and max(controls)<=.5))
 instrument=all(c['exact_serial_effect_error']<=1e-4 and c['exact_scalar_change_error']<=1e-5 for c in cells);capability=all(c['native_capability'] for c in cells);A=instrument and capability
 result={'pred_a':A,'pred_b':A and all(c['removal_pass'] for c in cells),'pred_c':A and all(c['directmixed_serial_effect_error']<=.1 for c in cells),'instrument':instrument,'native_capability':capability,'cells':cells,'arms':['native','physical8','physical9','dynamicjoint','frozen9joint','directmixedjoint','exactjoint'],'body_forwards':counts,'seconds':time.perf_counter()-tic,'scope':'Frozen physical components and analytic interaction bridge on new cities/endpoints/email-letter templates. Explicit native z/u/a/r9 inputs remain. Not corpusOOD or repaired newline selectivity.','source_shas':binding}
 torch.save(dict(margins=values),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
