#!/usr/bin/env python3
# BQGATE:12 body batches;96 sequences21-24tokens;6suffix arms;no fit;180sec.
"""pred_a priorwrite/pre/baseline relative<=1e-5 and partition<=1e-6.
pred_b city-only removal target AND distractor contrasts error<=.1 each4cells.
pred_c city-specific removal >=.8 own and <=.2 other fullcomponent contrast norm.
Null: contextual/query dependence distributes cue influence beyond city source positions.
12bodybatches96rows21-24tokens6suffixarms180sec,~25MB; frozen mixedpackage.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from folded_normalized_router_v1 import rotary
from compiled_mixed_token_head_v1 import execute_mixed_token_head
from regional_cue_row_check_v1 import validate
STEM='REGIONAL_SOURCE_POSITIONS_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 names=['REGIONAL_COMPETING_CUES_V1','REGIONAL_CITY_ROLE_CROSSOVER_V1'];rows=[];previous=[]
 for assignment,name in enumerate(names):
  rr=json.loads((P/(name+'_ROWS.json')).read_text())['rows'];validate(rr)
  for r in rr:rows.append(dict(r,assignment=assignment))
  previous.append(torch.load(P/(name+'_ARTIFACT.pt'),weights_only=True,map_location='cpu'))
 positions=[];batches=[]
 for i,r in enumerate(rows):
  target=[j for j,(a,b) in enumerate(zip(r['ids'],rows[i^1]['ids'])) if a!=b]
  tourist=[j for j,(a,b) in enumerate(zip(r['ids'],rows[i^2]['ids'])) if a!=b]
  assert len(target)==len(tourist)==1 and target!=tourist;positions.append([target[0],tourist[0]])
 for assignment in range(2):
  for length in sorted(set(len(r['ids']) for r in rows)):
   ids=[i for i,r in enumerate(rows) if r['assignment']==assignment and len(r['ids'])==length]
   batches.extend((length,ids[off:off+8]) for off in range(0,len(ids),8))
 assert len(rows)==96 and len(batches)==12
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('12 body batches96rows21-24tokens6suffix arms;frozen package;~25MB');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();last=model.transformer.h[17]
 program={k:v.cuda() for k,v in torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()}
 pre=torch.empty(96,1152,device='cuda');writes=torch.zeros(96,5,1152,device='cuda');pos_writes=torch.zeros(96,24,1152);currents=torch.zeros_like(pos_writes)
 for length,ids in batches:
  tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda');x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
  for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
  r=last.lambdas[0]*x+last.lambdas[1]*x0;current=F.rms_norm(r,(1152,));attention,_=last.attn(current,v1);pre[ids]=(r+attention)[:,-1];currents[ids,:length]=current.cpu()
  query=current[:,-1];qr=rotary(length-1,128).cuda()
  for pos in range(length):
   rotation=(qr.T@rotary(pos,128).cuda()).float()
   value=execute_mixed_token_head(query,current[:,pos],tokens[:,pos],rotation,program);pos_writes[ids,pos]=value.cpu();writes[ids,0]+=value
   for local,i in enumerate(ids):
    group=1 if pos==positions[i][0] else 2 if pos==positions[i][1] else 3
    writes[i,group]+=value[local]
 writes[:,4]=writes[:,1]+writes[:,2]
 def margins(delta):
  z=pre+delta;h=z+last.mlp(F.rms_norm(z,(1152,)));logits=30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
  return torch.stack([torch.stack([logits[i,r['uk_id']]-logits[i,r['us_id']],logits[i,r['control_ids'][0]]-logits[i,r['control_ids'][1]]]) for i,r in enumerate(rows)]).double()
 baseline=margins(torch.zeros_like(pre));arms=torch.stack([margins(-writes[:,j]) for j in range(5)]);effects=baseline[None]-arms
 def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 prior_w=torch.cat([a['write_vertices'][:,3] for a in previous]).cuda();prior_p=torch.cat([a['pre'] for a in previous]).cuda();prior_m=torch.cat([a['baseline_margins'] for a in previous]).cuda()
 checks=dict(write=rel(writes[:,0].double(),prior_w),pre=rel(pre,prior_p),baseline=rel(baseline,prior_m),partition=rel(writes[:,1:4].sum(1),writes[:,0]))
 aa=max(checks[k] for k in ('write','pre','baseline'))<=1e-5 and checks['partition']<=1e-6
 cells=[]
 for assignment in range(2):
  for family in range(2):
   ids=[i for i,r in enumerate(rows) if r['assignment']==assignment and r['family']==family]
   e=effects[:,ids,0].reshape(5,6,2,2);t=e[:,:,:,0]-e[:,:,:,1];d=e[:,:,0,:]-e[:,:,1,:]
   cells.append(dict(assignment=assignment,family=family,target_city_only_error=rel(t[4],t[0]),distractor_city_only_error=rel(d[4],d[0]),
    target_own_ratio=float(t[1].norm()/t[0].norm()),target_cross_ratio=float(d[1].norm()/d[0].norm()),
    tourist_own_ratio=float(d[2].norm()/d[0].norm()),tourist_cross_ratio=float(t[2].norm()/t[0].norm()),
    target_means=[float(v.mean()) for v in t],distractor_means=[float(v.mean()) for v in d]))
 bb=aa and all(c['target_city_only_error']<=.1 and c['distractor_city_only_error']<=.1 for c in cells)
 cc=aa and all(c['target_own_ratio']>=.8 and c['tourist_own_ratio']>=.8 and c['target_cross_ratio']<=.2 and c['tourist_cross_ratio']<=.2 for c in cells)
 torch.save(dict(pre=pre.cpu(),write_vertices=writes.cpu(),baseline_margins=baseline.cpu(),removal_margins=arms.cpu(),position_writes=pos_writes,current_states=currents,city_positions=positions),art)
 result={'pred_a':aa,'pred_b':bb,'pred_c':cc,'checks':checks,'cells':cells,'seconds':time.perf_counter()-tic,'body_forwards':len(batches),'artifact_sha':digest(art),'source_shas':binding,'arms':['all','editor_city','tourist_city','other','both_cities'],'scope':'Native read-site component removals. Query/contextual descendants can carry either cue at any position; not upstream causal mediation or fitting.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
