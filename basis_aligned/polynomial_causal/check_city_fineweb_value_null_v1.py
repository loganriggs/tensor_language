"""Same-boundary norm-matched specificity screen at head9.8 value boundary."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_FINEWEB_VALUE_NULL_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_ATTENTION7_DROP3_FINEWEB_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('760sequence equivalents;342block calls;CPU value nulls');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();signal.alarm(180)
 out=P/(STEM+'_RESULT.json');assert not out.exists()
 from fastload import load_model_fast
 model=load_model_fast().eval();old=torch.load(P/'CITY_DROP3_FINEWEB_INTERCHANGE_V1_ARTIFACT.pt',weights_only=True)
 fold=torch.load(P/'CITY_FINEWEB_MLP8_VALUE_FOLD_V1_ARTIFACT.pt',weights_only=True)
 ids=torch.tensor([r['ids'] for r in groups]);assert ids.shape==(40,32)
 edit=torch.cat([f['native_delta'] for f in old['fixtures']]);pieces=torch.cat([f['pieces'] for f in fold['fixtures']],dim=1)
 changes={2:pieces[1:5].sum(0).float()};norm_errors=[]
 for k in range(16):
  null=[]
  for i,row in enumerate(groups):
   gen=torch.Generator().manual_seed(18101000+1000*k+row['context_id'])
   random=torch.randn(32,128,generator=gen).double();norm=changes[2][i].double().norm(dim=-1,keepdim=True)
   d=(random*norm/random.norm(dim=-1,keepdim=True)*(1 if row['cue']=='British' else -1)).float()
   norm_errors.append(float(((d.double().norm(dim=-1,keepdim=True)-norm).abs()/norm.clamp_min(1e-8)).max()));null.append(d)
  changes[3+k]=torch.stack(null)
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
 attn.squared_attention=MethodType(changed,attn);values=torch.zeros(19,40,10,dtype=torch.float64);calls=0
 try:
  for arm in range(19):
   state['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
   for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
   scores=30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30)
   for i,row in enumerate(groups):
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=scores[i,left]-scores[i,right]
 finally:h.remove();attn.squared_attention=original
 v=expand(values,mapping,6)
 old_scores=torch.load(P/'CITY_FINEWEB_VALUE_MEDIATION_V1_ARTIFACT.pt',weights_only=True)['values'][:3]
 diff=v[:3]-old_scores;absolute=float(diff.abs().max());relative=float(diff.norm()/old_scores.norm())
 effect=v-v[1:2];rms=effect.square().mean(1).sqrt();null=rms[3:,0];ordered=null.sort().values;median=float((ordered[7]+ordered[8])/2)
 prior=json.loads((P/'CITY_FINEWEB_RESPONSE_V1_RESULT.json').read_text());reversed_docs=set(prior['reversed_documents']);groups_out={}
 for label,idx in [('all',list(range(240))),('reversed',[i for i,r in enumerate(rows) if r['context_id'] in reversed_docs]),('other',[i for i,r in enumerate(rows) if r['context_id'] not in reversed_docs])]:
  rr=effect[:,idx].square().mean(1).sqrt();groups_out[label]={'real_target_rms':float(rr[2,0]),'real_control_over_target':(rr[2,1:]/rr[2,0]).tolist(),'null_readout_rms':rr[3:].tolist(),'nulls_beaten':int((rr[3:,0]<rr[2,0]).sum())}
 r={'pred_a':absolute<=1e-4 and relative<=1e-5 and max(norm_errors)<=1e-5 and outside==0 and max(other_errors)==0 and max(errors)<=1e-5 and bool(torch.isfinite(v).all()) and calls==342 and not torch.cuda.is_initialized(),
 'pred_b':float(rms[2,0])>=2*median and bool((null<rms[2,0]).all()),'pred_c':bool((rms[2,1:]<=.5*rms[2,0]).all()),
 'groups':groups_out,'target_over_null_median':float(rms[2,0])/median,'nulls_beaten':int((null<rms[2,0]).sum()),'max_null_norm_error':max(norm_errors),'anchor_max_abs':absolute,'anchor_relative':relative,'max_subtraction_relative_error':max(errors),'other_head_error':max(other_errors),'outside':outside,'body_forwards':760,'batched_block_calls':calls,'seconds':time.perf_counter()-start,'scope':'Opened same-head9.8-value-boundary specificity; swap background fixed. No fresh or uniformly selective subgroup promotion; no composition claim.','source_shas':binding}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['source_shas','groups']},indent=2));signal.alarm(0)
if __name__=='__main__':main()
