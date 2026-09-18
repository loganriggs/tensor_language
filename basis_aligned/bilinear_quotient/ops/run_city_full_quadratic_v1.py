#!/usr/bin/env python3
# BQGATE:120bodyforwards;40prefixes;120seconds;no fitting.
"""pred_a full replay abs/rel<=1e-5; pred_b full error<=.8 linear error;
pred_c full error<=.8 secant error.120forwards, opened local-term ablation.
Null comparator: omit B or halve B at identical block9 site and native suffix.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_FULL_QUADRATIC_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_FULL_PILE_V2_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows)
 assert len(groups)==40
 if os.environ.get('BQLIB_NO_MODEL') or os.environ.get('BQLIB_DRYRUN'):
  print('120bodyforwards;40prefixes;full/secant/linear');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(120)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval()
 half=torch.load(P/'CITY_FULL_PILE_V3_ARTIFACT.pt',weights_only=True)
 full=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)
 writes=[]
 for a,b in zip(half['fixtures'],full['fixtures'],strict=True):
  assert torch.equal(a['candidate_inputs']['token_ids'],b['candidate_inputs']['token_ids'])
  h=a['expected_candidate_delta'].cuda();f=b['expected_candidate_delta'].cuda()
  writes.append([f,2*h,4*h-f])
 state={}
 def pre9(module,args):
  x,first,x0=args;return x+writes[state['i']][state['arm']].to(x.dtype),first,x0
 handle=model.transformer.h[9].register_forward_pre_hook(pre9)
 values=torch.zeros(3,40,10,dtype=torch.float64);count=0
 try:
  for arm in range(3):
   state['arm']=arm
   for i,row in enumerate(groups):
    state['i']=i
    assert row['ids']==full['fixtures'][i]['candidate_inputs']['token_ids'][0].tolist()
    ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
    for block in model.transformer.h:x,first=block(x,first,x0)
    scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=(scores[left]-scores[right]).cpu()
    count+=1
 finally:handle.remove()
 v=expand(values,mapping,6);native=full['values'][0];actual=full['values'][1]-native
 records={};base=native[:,0][::2]-native[:,0][1::2];cap=base>=.1
 for i,name in enumerate(['full','secant','linear']):
  effect=v[i]-native;rms=effect.square().mean(0).sqrt();pair=v[i,:,0][::2]-v[i,:,0][1::2];atten=(base[cap]-pair[cap])/base[cap]
  error=float((effect[:,0]-actual[:,0]).norm()/actual[:,0].norm());coll=(rms[1:]/rms[0]).tolist();positive=float((atten>0).double().mean());mean=float(atten.mean())
  records[name]={'target_error':error,'control_over_target':coll,'positive_fraction':positive,'mean_attenuation':mean,'screen_pass':error<=.35 and max(coll)<=.5 and positive>=.90 and mean>=.02}
 diff=v[0]-full['values'][3];absolute=float(diff.abs().max());relative=float(diff.norm()/full['values'][3].norm())
 r={'pred_a':absolute<=1e-5 and relative<=1e-5 and bool(torch.isfinite(v).all()) and count==120,
    'pred_b':records['full']['target_error']<=.8*records['linear']['target_error'],
    'pred_c':records['full']['target_error']<=.8*records['secant']['target_error'],
    'arms':records,'replay_max_abs':absolute,'replay_relative':relative,'body_forwards':count,'seconds':time.perf_counter()-start,
    'scope':'Opened term-necessity screen, full suffix recomputed; not fresh/selectivity-null/composition confirmation','source_shas':binding}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
