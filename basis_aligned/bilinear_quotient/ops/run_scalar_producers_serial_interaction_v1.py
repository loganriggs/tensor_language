#!/usr/bin/env python3
# BQGATE:144bodyforwards;48fixedprefixes<=22tokens;120seconds;no fitting.
"""pred_a scalar/native<=1e-5relative,baseline/joint margin replay<=1e-4relative.
pred_b EACHtemplate frozen-second change norm>=.5interaction and cosine>=.8.
pred_c EACHtemplate remaining interaction norm<=.5original.
Null: serial second-component changes do not explain nonadditivity.
Price144bodyforwards48rows3arms;120seconds;no refit or OOD claim.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from compiled_scalar_producers_v1 import head_scalar
STEM='SCALAR_PRODUCERS_SERIAL_INTERACTION_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];assert len(rows)==48 and max(len(r['ids']) for r in rows)<=22
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('144bodyforwards48rows; native/dynamicjoint/frozen-secondjoint');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();tic=time.perf_counter();signal.alarm(120);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();producer={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()}
 writers=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['writers'].cuda()
 scalars=torch.load(P/'SCALAR_PRODUCERS_SERIAL_NATIVE_SCALARS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['scalars'].cuda()
 context={};errors=[];counts=0;values=torch.zeros(48,3,2,dtype=torch.float64)
 def hook(index,module,args,output):
  scalar=head_scalar(args[0],context['tokens'],producer,index)
  frozen=scalars[index,context['row'],:args[0].shape[1]][None,:]
  if context['arm']==0:
   errors.append(float((scalar-frozen).norm()/frozen.norm().clamp_min(1e-30)));return output
  if context['arm']==2 and index==1:scalar=frozen
  return output[0]-(scalar[...,None]*writers[index]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[layer].attn.register_forward_hook(lambda m,a,o,i=index:hook(i,m,a,o)) for index,layer in enumerate((8,9))]
 try:
  for i,row in enumerate(rows):
   tokens=torch.tensor([row['ids']],device='cuda')
   for arm in range(3):
    context.update(tokens=tokens,row=i,arm=arm);x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    counts+=1;logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    values[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();values[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for h in handles:h.remove()
 assert counts==144
 old=torch.load(P/'SCALAR_PRODUCERS_RECURSIVE_REMOVAL_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['regional']
 def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 checks=dict(scalar_replay=max(errors),baseline_replay=rel(values[:,0],old[:,0]),dynamic_joint_replay=rel(values[:,1],old[:,3]))
 cells=[]
 for family in range(2):
  ix=[i for i,r in enumerate(rows) if r['family']==family];e=old[ix,:,0]-old[ix,:1,0];I=e[:,1]+e[:,2]-e[:,3];D=values[ix,2,0]-values[ix,1,0]
  ratio=float(D.norm()/I.norm().clamp_min(1e-30));cos=float(F.cosine_similarity(D,I,dim=0));remaining=float((I-D).norm()/I.norm().clamp_min(1e-30))
  cells.append(dict(family=family,change_to_interaction_norm=ratio,change_interaction_cosine=cos,remaining_interaction_fraction=remaining,original_interaction_norm=float(I.norm()),frozen_joint_cue_reduction=float(((values[ix,0,0]-values[ix,2,0])[::2]-(values[ix,0,0]-values[ix,2,0])[1::2]).mean())))
 A=checks['scalar_replay']<=1e-5 and max(checks['baseline_replay'],checks['dynamic_joint_replay'])<=1e-4
 result={'pred_a':A,'pred_b':A and all(c['change_to_interaction_norm']>=.5 and c['change_interaction_cosine']>=.8 for c in cells),'pred_c':A and all(c['remaining_interaction_fraction']<=.5 for c in cells),'checks':checks,'cells':cells,'body_forwards':counts,'seconds':time.perf_counter()-tic,'scope':'Hybrid frozen-second-write diagnostic; later native state recomputes. Not direct-edge isolation, component adoption, fresh OOD, or repair of newline outlier.','source_shas':binding}
 torch.save(dict(margins=values),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
