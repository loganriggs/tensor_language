#!/usr/bin/env python3
# BQGATE:160bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a anchors<=1e-5/finite/outsidezero/160forwards;
pred_b both attenuation>=.02 and positive>=.90 everyfamily;
pred_c current live>=1e-5/fourcontrols<=.5; pred_d interaction<=.35smaller.
Opened native value-source factorial; no fresh or random-split claim.
"""
from pathlib import Path
import os,sys,json,hashlib,time,signal
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
from typed_face_write_atoms_v1 import native
STEM='CITY_VALUE_BRANCH_FACTORIAL_V1';ARMS=['native','inherited','current','both']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 doc=json.loads((P/'CITY_REMOVAL_ENDPOINT_FRESH_V1_ROWS.json').read_text());rows=doc['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('160bodyforwards;40prefixes;4nativearms');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt',weights_only=True).items()}
 tables={k:v.cuda() for k,v in torch.load(P/'CITY_REMOVAL_ENDPOINT_FRESH_V1_TABLES.pt',weights_only=True).items()};p['token_ids']=tables['token_ids'];p['first_table']=tables['first_table'][:,256:384]
 cache=[{} for _ in groups];state={};outside=[]
 def pre(module,args):
  if state['arm']=='native':
   row=groups[state['i']];x=args[0];city=row['city_position'];route=native.routing(p,x,x[:,city],city);iv=native.inherited(p,row['ids'][city]).to(x.dtype);cv=F.linear(x,p['current_value'])[:,city];mask=torch.zeros(x.shape[1],device=x.device,dtype=torch.bool);mask[row['destination_positions']]=True
   di=-.5*F.linear(route[...,None]*p['mixture']*iv[:,None],p['output'])*mask[None,:,None]
   dc=-.5*F.linear(route[...,None]*(1-p['mixture'])*cv[:,None],p['output'])*mask[None,:,None]
   cache[state['i']]={'inherited':di,'current':dc,'both':di+dc}
   outside.extend([float(d[:,~mask].abs().max()) for d in [di,dc]])
 def post(module,args,result):
  if state['arm']!='native':return result[0]+cache[state['i']][state['arm']].to(result[0].dtype),result[1]
  return result
 handles=[model.transformer.h[8].attn.register_forward_pre_hook(pre),model.transformer.h[8].attn.register_forward_hook(post)];values=torch.zeros(4,40,10,dtype=torch.float64);count=0
 try:
  for ai,arm in enumerate(ARMS):
   state['arm']=arm
   for i,row in enumerate(groups):
    state['i']=i;ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
    for block in model.transformer.h:x,first=block(x,first,x0)
    scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[ai,i,j]=(scores[left]-scores[right]).cpu()
    count+=1
 finally:
  for h in handles:h.remove()
 v=expand(values,mapping,6);e=v-v[:1];old=torch.load(P/'CITY_REMOVAL_ENDPOINT_FRESH_V1_ARTIFACT.pt',weights_only=True)['values'];anchor=float((v[:2]-old[:2]).abs().max());records={};cells=[]
 def stats(idx,arm):
  rms=e[arm,idx].square().mean(0).sqrt();base=v[0,idx,0][::2]-v[0,idx,0][1::2];pair=v[arm,idx,0][::2]-v[arm,idx,0][1::2];cap=base>=.1;atten=(base[cap]-pair[cap])/base[cap]
  return {'target_rms':float(rms[0]),'control_over_target':(rms[1:]/rms[0].clamp_min(1e-30)).tolist(),'capable_pairs':int(cap.sum()),'positive_fraction':float((atten>0).double().mean()) if len(atten) else 0.,'mean_attenuation':float(atten.mean()) if len(atten) else 0.,'attenuations':atten.tolist()}
 for family in doc['variants']:
  idx=[i for i,r in enumerate(rows) if r['variant']==family];record={arm:stats(idx,ai) for ai,arm in enumerate(ARMS[1:],1)};ei=e[1,idx,0];ec=e[2,idx,0];joint=e[3,idx,0];record['interaction_over_smaller']=float((joint-ei-ec).norm()/torch.minimum(ei.norm(),ec.norm()).clamp_min(1e-30));records[family]=record
 for cell in doc['contexts']:
  idx=[i for i,r in enumerate(rows) if r['context_id']==cell['context_id']];cells.append({**cell,'native_pair_margins':(v[0,idx,0][::2]-v[0,idx,0][1::2]).tolist(),'arms':{arm:stats(idx,ai) for ai,arm in enumerate(ARMS[1:],1)}})
 result={'pred_a':anchor<=1e-5 and bool(torch.isfinite(v).all()) and max(outside)==0 and count==160,'pred_b':all(r['both']['capable_pairs']>=18 and r['both']['positive_fraction']>=.90 and r['both']['mean_attenuation']>=.02 for r in records.values()),'pred_c':all(r['current']['target_rms']>=1e-5 and max(r['current']['control_over_target'])<=.5 for r in records.values()),'pred_d':all(r['interaction_over_smaller']<=.35 and min(r['inherited']['target_rms'],r['current']['target_rms'])>=1e-5 for r in records.values()),'families':records,'cells':cells,'anchor_max_abs':anchor,'max_outside':max(outside),'body_forwards':count,'seconds':time.perf_counter()-start,'source_shas':binding,'scope':'Opened native source factorial. No donor or approximate MLP response. No fresh/null-selectivity/random-split specificity claim.'}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','cells')},indent=2));signal.alarm(0)
if __name__=='__main__':main()
