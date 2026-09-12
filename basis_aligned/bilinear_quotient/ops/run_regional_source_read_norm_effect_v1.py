#!/usr/bin/env python3
# BQGATE:0 body forwards;96 cached rows;7 suffix forwards;no fit;180sec.
"""pred_a baseline relative<=1e-5 against saved native margins.
pred_b feature-only signed margin-effect relative<=.1 against both source ports EACH8cells.
pred_c normalizer-only same. Null: source normalization effect remains functionally material.
0bodyforwards96cachedrows7suffixarms180sec,~2MB; native background fixed.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
STEM='REGIONAL_SOURCE_READ_NORM_EFFECT_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 a=torch.load(P/'REGIONAL_SOURCE_POSITIONS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');b=torch.load(P/'REGIONAL_SOURCE_READ_NORM_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 rows=sum([json.loads((P/(s+'_ROWS.json')).read_text())['rows'] for s in ('REGIONAL_COMPETING_CUES_V1','REGIONAL_CITY_ROLE_CROSSOVER_V1')],[]);assert len(rows)==96
 assert b['port_writes'].shape==(2,3,96,1152)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('0body96cachedrows7suffixforwards180sec~2MB;no fitting');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2)
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();last=model.transformer.h[17];pre=a['pre'].cuda();base=b['base_write']
 def margins(delta):
  z=pre+delta.cuda();h=z+last.mlp(F.rms_norm(z,(1152,)));logits=30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
  return torch.stack([torch.stack([logits[i,r['uk_id']]-logits[i,r['us_id']],logits[i,r['control_ids'][0]]-logits[i,r['control_ids'][1]]]) for i,r in enumerate(rows)]).double().cpu()
 baseline=margins(torch.zeros_like(base));effects=[]
 for k,bit in enumerate((1,2)):
  donor=torch.arange(96)^bit
  effects.append(torch.stack([margins(b['port_writes'][k,j]-base)-baseline for j in (0,2,1)]))
 effects=torch.stack(effects)
 def rel(x,y):return float((x-y).norm()/y.norm().clamp_min(1e-30))
 prior=torch.load(P/'REGIONAL_QUERY_SOURCE_EFFECT_V1_ARTIFACT.pt',weights_only=True)['effects'][:,2]
 replay=max(rel(baseline,a['baseline_margins']),rel(effects[:,0],prior));cells=[]
 for k,cue in enumerate(('editor','tourist')):
  for assignment in range(2):
   for family in range(2):
    ids=list(range(assignment*48+family*24,assignment*48+family*24+24));v=effects[k,:,ids,0]
    direction=torch.tensor([1 if ((i%4)//(1 if k==0 else 2))%2 else -1 for i in ids])
    cells.append(dict(cue=cue,assignment=assignment,family=family,feature_only_error=rel(v[2],v[0]),normalizer_only_error=rel(v[1],v[0]),full_effect_norm=float(v[0].norm()),full_direction_positive=int((v[0]*direction>0).sum()),feature_direction_positive=int((v[2]*direction>0).sum()),normalizer_direction_positive=int((v[1]*direction>0).sum()),nonlinear_interaction_relative=rel(v[1]+v[2],v[0])))
 aa=replay<=1e-5;result={'pred_a':aa,'pred_b':aa and all(c['feature_only_error']<=.1 for c in cells),'pred_c':aa and all(c['normalizer_only_error']<=.1 for c in cells),'baseline_replay':replay,'cells':cells,'seconds':time.perf_counter()-tic,'body_forwards':0,'source_shas':binding,'scope':'Conditional source feature/key-normalizer replacements with query fixed. Full arm equals previous source-only swap. No upstream closure or broad OOD.'}
 torch.save(dict(baseline=baseline,effects=effects),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
