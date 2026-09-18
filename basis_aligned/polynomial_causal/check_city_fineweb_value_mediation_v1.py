"""Five-arm native-suffix causal screen at head9.8 value boundary."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_FINEWEB_VALUE_MEDIATION_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_ATTENTION7_DROP3_FINEWEB_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('200sequence equivalents;90block calls;CPU value mediation');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();signal.alarm(180)
 out=P/(STEM+'_RESULT.json');assert not out.exists()
 from fastload import load_model_fast
 model=load_model_fast().eval();old=torch.load(P/'CITY_DROP3_FINEWEB_INTERCHANGE_V1_ARTIFACT.pt',weights_only=True)
 fold=torch.load(P/'CITY_FINEWEB_MLP8_VALUE_FOLD_V1_ARTIFACT.pt',weights_only=True)
 ids=torch.tensor([r['ids'] for r in groups]);assert ids.shape==(40,32)
 edit=torch.cat([f['native_delta'] for f in old['fixtures']]);pieces=torch.cat([f['pieces'] for f in fold['fixtures']],dim=1)
 changes={2:pieces[1:5].sum(0).float(),3:pieces[3].float(),4:pieces[0].float()}
 outside=max(float(d[:,[i for i in range(32) if i not in groups[0]['destination_positions']]].abs().max()) for d in changes.values())
 state={'arm':0};errors=[];other_errors=[]
 def post8(module,args,out):return (out[0]+edit,out[1]) if state['arm'] else out
 h=model.transformer.h[8].attn.register_forward_hook(post8);attn=model.transformer.h[9].attn;original=attn.squared_attention
 def changed(module,q,k,v,q2,k2):
  arm=state['arm']
  if arm>=2:
   new=v.clone();new[:,:,8]-=changes[arm]
   actual=v[:,:,8]-new[:,:,8];errors.append(float((actual.double()-changes[arm].double()).norm()/changes[arm].double().norm()))
   other_errors.append(float((new[:,:,:8]-v[:,:,:8]).abs().max()));v=new
  return original(q,k,v,q2,k2)
 attn.squared_attention=MethodType(changed,attn);values=torch.zeros(5,40,10,dtype=torch.float64);calls=0
 try:
  for arm in range(5):
   state['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
   for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
   scores=30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30)
   for i,row in enumerate(groups):
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=scores[i,left]-scores[i,right]
 finally:h.remove();attn.squared_attention=original
 v=expand(values,mapping,6);e=v-v[:1];diff=v[:2]-old['values'][[0,1]];absolute=float(diff.abs().max());relative=float(diff.norm()/old['values'][[0,1]].norm())
 prior=json.loads((P/'CITY_FINEWEB_RESPONSE_V1_RESULT.json').read_text());reversed_docs=set(prior['reversed_documents']);records={};names=['native','swap','minus_mlp8_value','minus_quadratic_value','minus_direct_value']
 for label,idx in [('all',list(range(240))),('reversed',[i for i,r in enumerate(rows) if r['context_id'] in reversed_docs]),('other',[i for i,r in enumerate(rows) if r['context_id'] not in reversed_docs])]:
  base=v[0,idx,0][::2]-v[0,idx,0][1::2];cap=base>=.1;target=e[1,idx,0]
  records[label]={}
  for arm,name in enumerate(names[1:],1):
   delta=v[arm,idx]-v[1,idx];pair=v[arm,idx,0][::2]-v[arm,idx,0][1::2];atten=(base[cap]-pair[cap])/base[cap]
   records[label][name]={'target_change_over_swap':float(delta[:,0].norm()/target.norm()),'target_change_rms':float(delta[:,0].square().mean().sqrt()),'positive_fraction':float((atten>0).double().mean()),'mean_attenuation':float(atten.mean()),'control_change_rms':delta[:,1:].square().mean(0).sqrt().tolist(),'signed_change_aligned_with_swap':float(delta[:,0].dot(target)/target.square().sum())}
 docs=[]
 for context in range(20):
  idx=[i for i,r in enumerate(rows) if r['context_id']==context];b=v[0,idx,0][::2]-v[0,idx,0][1::2];cap=b>=.1
  docs.append({'context_id':context,'attenuations':{name:((b[cap]-(v[a,idx,0][::2]-v[a,idx,0][1::2])[cap])/b[cap]).tolist() for a,name in enumerate(names[1:],1)}})
 r={'pred_a':absolute<=1e-4 and relative<=1e-5 and outside==0 and max(other_errors)==0 and max(errors)<=1e-5 and bool(torch.isfinite(v).all()) and calls==90 and not torch.cuda.is_initialized(),
 'pred_b':records['reversed']['minus_mlp8_value']['target_change_over_swap']>=.20 and records['reversed']['minus_mlp8_value']['target_change_rms']>=1e-5,
 'pred_c':all(records[g]['minus_quadratic_value']['target_change_over_swap']>=.10 for g in ['reversed','other']),
 'pred_d':all(records[g]['minus_direct_value']['target_change_over_swap']>=.20 for g in ['reversed','other']),
 'groups':records,'documents':docs,'anchor_max_abs':absolute,'anchor_relative':relative,'max_subtraction_relative_error':max(errors),'other_head_error':max(other_errors),'outside':outside,'body_forwards':200,'batched_block_calls':calls,'seconds':time.perf_counter()-start,'scope':'Opened factor-boundary mediation with full native suffix. Not MLP8 ablation, independent composition, fresh or norm-null-confirmed selectivity.','source_shas':binding}
 torch.save({'values':v,'arm_names':names},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['source_shas','documents']},indent=2));signal.alarm(0)
if __name__=='__main__':main()
