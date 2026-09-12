#!/usr/bin/env python3
# BQGATE:240bodyforwards;48fixedprefixes<=22tokens;120seconds;no fitting.
"""pred_a baseline/joint/exact bridge serial effects<=1e-4relative.
pred_b EACHtemplate source-only serial effect error<=10%relative.
pred_c EACHtemplate query-only serial effect error>=25%relative.
Null: intermediate accuracy fails native signed-effect prediction.
Price240bodyforwards48rows5arms;120seconds;no fitting.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from compiled_scalar_producers_v1 import head_scalar
STEM='SCALAR_SERIAL_JOINT_PORTS_EFFECT_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];assert len(rows)==48 and max(len(r['ids']) for r in rows)<=22
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('240bodyforwards48rows; joint query/source generated-component effect validation');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();tic=time.perf_counter();signal.alarm(120);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();producer={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()}
 writers=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['writers'].cuda()
 parts=torch.load(P/'SCALAR_SERIAL_JOINT_PORTS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['terms'].cuda()
 predictions=torch.stack((parts[0],parts.sum(0),parts[0]+parts[2]+parts[4]+parts[6],parts[0]+parts[1]))
 context={};errors=[];counts=0;values=torch.zeros(48,5,2,dtype=torch.float64)
 def hook(index,module,args,output):
  scalar=head_scalar(args[0],context['tokens'],producer,index)
  if context['arm']==0:return output
  if context['arm']>=2 and index==1:
   scalar=predictions[{2:1,3:2,4:3}[context['arm']],context['row'],:args[0].shape[1]][None,:]
  return output[0]-(scalar[...,None]*writers[index]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[layer].attn.register_forward_hook(lambda m,a,o,i=index:hook(i,m,a,o)) for index,layer in enumerate((8,9))]
 try:
  for i,row in enumerate(rows):
   tokens=torch.tensor([row['ids']],device='cuda')
   for arm in range(5):
    context.update(tokens=tokens,row=i,arm=arm);x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    counts+=1;logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    values[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();values[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for h in handles:h.remove()
 assert counts==240
 old=torch.load(P/'SCALAR_PRODUCERS_RECURSIVE_REMOVAL_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['regional']
 def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 checks=dict(baseline_replay=rel(values[:,0],old[:,0]),dynamic_joint_replay=rel(values[:,1],old[:,3]))
 frozen=torch.load(P/'SCALAR_PRODUCERS_SERIAL_INTERACTION_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['margins'][:,2]
 cells=[]
 for family in range(2):
  ix=[i for i,r in enumerate(rows) if r['family']==family];reference=values[ix,1,0]-frozen[ix,0]
  errors=[rel(values[ix,a,0]-frozen[ix,0],reference) for a in (2,3,4)]
  effects=values[ix,:,0]-values[ix,:1,0];reduction=effects[1::2]-effects[::2]
  cells.append(dict(family=family,serial_effect_errors=errors,reference_serial_effect_norm=float(reference.norm()),mean_pair_cue_reduction=reduction.mean(0).tolist(),control_meanabs_changes=(values[ix,:,1]-values[ix,:1,1]).abs().mean(0).tolist()))
 A=max(checks.values())<=1e-4 and all(c['serial_effect_errors'][0]<=1e-4 for c in cells)
 result={'pred_a':A,'pred_b':A and all(c['serial_effect_errors'][1]<=.1 for c in cells),'pred_c':A and all(c['serial_effect_errors'][2]>=.25 for c in cells),'checks':checks,'cells':cells,'arms':['native','dynamicjoint','joint_field','source_field','query_field'],'body_forwards':counts,'seconds':time.perf_counter()-tic,'scope':'Conditional joint-query/source generation tested at signed native serial-effect boundary; removed component field is changed, not the full head routing. Reused contexts; no newline repair or global extraction.','source_shas':binding}
 torch.save(dict(margins=values),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
