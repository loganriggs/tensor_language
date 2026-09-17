#!/usr/bin/env python3
# BQGATE:160bodyforwards;16prefixes;300seconds;no fitting.
"""pred_a native/source anchors<=1e-5abs/1e-6rel and live RMS>=1e-5.
pred_b carry6 effect error<=.35; pred_c carry6+attn7<=.20; pred_d closure<=1e-12.
Null: earlier source omission misses the full causal increment. 160 forwards.
Opened source screen; no selective, fresh, or complete-circuit certification.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';D=P/'extracted_circuits/odd_attention8h2_mlp7_donor_v1'
sys.path[:0]=[str(D),str(Path(__file__).parent),str(P),str(ROOT)]
import execute,native
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v2 import measure
from run_even_value_factorial_native_v1 import setup
STEM='MLP7_INPUT_SOURCE_V1';ARMS=['native']+[f'source:{i}' for i in range(8)]+['direct']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  assert len(groups)==16;print('160bodyforwards;16prefixes;full MLP7 source factorial');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda')
 program={part:{k:v.to('cuda') for k,v in values.items()} for part,values in torch.load(D/'program.pt',weights_only=True).items()}
 sources=[];state={};errors=[]
 def pre(module,args):
  if len(sources)<len(groups):state['c']=module.lambdas[0]*args[0];state['e']=module.lambdas[1]*args[2]
 def after(module,args,result):
  if len(sources)<len(groups):
   city=groups[len(sources)]['city_position'];state['parts']=torch.stack([state['c'][:,city],result[0][:,city],state['e'][:,city]])
 def mlp_pre(module,args):
  if len(sources)<len(groups):
   import torch.nn.functional as F
   parts=state['parts'];z=F.rms_norm(parts.sum(0),(1152,));city=groups[len(sources)]['city_position'];target=args[0][:,city]
   errors.append(float((z-target).norm()/target.norm()));sources.append(parts.detach().clone())
 index={(r['context_id'],r['cue']):i for i,r in enumerate(groups)}
 def write(arm,row,donor_row,current,donor,mask):
  city=row['city_position'];rt=row['ids'][city];dt=donor_row['ids'][city]
  if arm=='direct':return native.execute(program['head'],current,donor[:,city],rt,dt,city,mask)
  i=index[(row['context_id'],row['cue'])];bits=int(arm.split(':')[1]);g=sum(sources[i^1][j] if bits&(1<<j) else sources[i][j] for j in range(3))
  return execute.execute(program,current,g,rt,dt,city,mask)
 b=model.transformer.h[7];handles=[b.register_forward_pre_hook(pre),b.attn.register_forward_hook(after),b.mlp.register_forward_pre_hook(mlp_pre)]
 try:m=measure(model,graph,groups,ARMS,write)
 finally:
  for h in handles:h.remove()
 v=expand(m['values'],mapping,6)
 old=torch.load(P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_ARTIFACT.pt',weights_only=True)['values']
 expected=old[[5,10]];difference=v[[1,8]]-expected
 anchor_abs=float(difference.abs().max());anchor_rel=float(difference.norm()/expected.norm())
 corners=v[1:9]-v[1:2];target=corners[7,:,0]
 cells={cue:[i for i,r in enumerate(rows) if r['cue']==cue] for cue in ['British','American']};cells.update({f'endpoint_{e}':[i for i,r in enumerate(rows) if r['endpoint']==e] for e in range(6)})
 metrics={name:{str(mask):float((corners[mask,idx,0]-target[idx]).norm()/target[idx].norm().clamp_min(1e-8)) for mask in range(1,7)} for name,idx in cells.items()}
 div=corners.clone()
 for bit in range(3):
  for mask in range(8):
   if mask&(1<<bit):div[mask]-=div[mask^(1<<bit)]
 closure=float((div.sum(0)-corners[7]).abs().max())
 stats={str(mask):{'target_rms_logits':float(div[mask,:,0].square().mean().sqrt()),'aligned_fraction':float(div[mask,:,0]@target/(target@target)),'all_readout_rms':div[mask].square().mean(0).sqrt().tolist()} for mask in range(1,8)}
 result={'pred_a':max(errors)<=1e-5 and anchor_abs<=1e-5 and anchor_rel<=1e-6 and float(target.square().mean().sqrt())>=1e-5,'pred_b':all(x['1']<=.35 for x in metrics.values()),'pred_c':all(x['3']<=.20 for x in metrics.values()),'pred_d':closure<=1e-12 and bool(torch.isfinite(v).all()),
  'source_errors':errors,'anchor_max_abs':anchor_abs,'anchor_relative':anchor_rel,'direct_full_error':float((v[8]-v[9]).abs().max()),'cell_errors':metrics,'term_stats':stats,'mobius_closure':closure,'full_increment_rms_logits':float(target.square().mean().sqrt()),'body_forwards':m['body_forwards'],'seconds':time.perf_counter()-start,
  'panel_status':'opened eight-context source screen','source_shas':binding,'scope':'g7 source interventions through full MLP7, both keys and conditional odd-value suffix. Native g7 sources still external. No fresh/source-selectivity or special composition claim.'}
 torch.save({'values':v,'dividends':div,'sources':torch.stack(sources).cpu()},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
