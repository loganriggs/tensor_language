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
STEM='CITY_VALUE_PATH_SPLIT_NULL_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_ATTENTION7_DROP3_FINEWEB_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('1480sequence equivalents;666block calls;CPU norm-preserving split nulls');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();signal.alarm(600)
 out=P/(STEM+'_RESULT.json');assert not out.exists()
 from fastload import load_model_fast
 model=load_model_fast().eval();old=torch.load(P/'CITY_DROP3_FINEWEB_INTERCHANGE_V1_ARTIFACT.pt',weights_only=True)
 fold=torch.load(P/'CITY_FINEWEB_MLP8_VALUE_FOLD_V1_ARTIFACT.pt',weights_only=True)
 ids=torch.tensor([r['ids'] for r in groups]);assert ids.shape==(40,32)
 edit=torch.cat([f['native_delta'] for f in old['fixtures']]);pieces=torch.cat([f['pieces'] for f in fold['fixtures']],dim=1)
 real_a=pieces[0];real_b=pieces[1:5].sum(0);parent=real_a+real_b
 changes={2:real_b.float(),3:real_a.float(),4:parent.float()};norm_errors=[];sum_errors=[]
 for k in range(16):
  aa=[];bb=[]
  for i,row in enumerate(groups):
   a,b,total=real_a[i],real_b[i],parent[i];s2=total.square().sum(-1,keepdim=True)
   projection=(a*total).sum(-1,keepdim=True)/s2.clamp_min(1e-30)*total
   residual=a-projection;length=residual.norm(dim=-1,keepdim=True)
   gen=torch.Generator().manual_seed(18103000+1000*k+row['context_id'])
   random=torch.randn(a.shape,generator=gen).double()*(1 if row['cue']=='British' else -1)
   perpendicular=random-(random*total).sum(-1,keepdim=True)/s2.clamp_min(1e-30)*total
   ar=projection+length*perpendicular/perpendicular.norm(dim=-1,keepdim=True).clamp_min(1e-30);br=total-ar
   af,bf=ar.float(),br.float();aa.append(af);bb.append(bf)
   for got,ref in [(af,a),(bf,b)]:norm_errors.append(float(((got.double().norm(dim=-1)-ref.norm(dim=-1)).abs()/ref.norm(dim=-1).clamp_min(1e-8)).max()))
   sum_errors.append(float((af.double()+bf.double()-total).norm()/total.norm().clamp_min(1e-30)))
  changes[5+2*k]=torch.stack(aa);changes[6+2*k]=torch.stack(bb)
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
 attn.squared_attention=MethodType(changed,attn);values=torch.zeros(37,40,10,dtype=torch.float64);calls=0
 try:
  for arm in range(37):
   state['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
   for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
   scores=30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30)
   for i,row in enumerate(groups):
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=scores[i,left]-scores[i,right]
 finally:h.remove();attn.squared_attention=original
 v=expand(values,mapping,6)
 prior_scores=torch.load(P/'CITY_VALUE_PATH_COMPOSITION_V1_ARTIFACT.pt',weights_only=True)['values']
 diff=v[:5]-prior_scores;absolute=float(diff.abs().max());relative=float(diff.norm()/prior_scores.norm());e=v-v[1:2]
 prior=json.loads((P/'CITY_FINEWEB_RESPONSE_V1_RESULT.json').read_text());reversed_docs=set(prior['reversed_documents'])
 def metric(idx):
  a,b,j=[e[k,idx,0] for k in [3,2,4]];real=float((j-a-b).norm()/min(a.norm(),b.norm()));ratios=[];live=[]
  for k in range(16):
   ra,rb=e[5+2*k,idx,0],e[6+2*k,idx,0];ratios.append(float((j-ra-rb).norm()/min(ra.norm(),rb.norm())));live.append([float(ra.norm()/j.norm()),float(rb.norm()/j.norm())])
  order=sorted(ratios);median=(order[7]+order[8])/2
  return {'real_interaction':real,'single_over_joint':[float(a.norm()/j.norm()),float(b.norm()/j.norm())],'random_interactions':ratios,'random_single_over_joint':live,'random_median':median,'real_over_random_median':real/median,'random_splits_beaten':sum(real<x for x in ratios)}
 groups_out={label:metric(idx) for label,idx in [('all',list(range(240))),('reversed',[i for i,r in enumerate(rows) if r['context_id'] in reversed_docs]),('other',[i for i,r in enumerate(rows) if r['context_id'] not in reversed_docs])]}
 r={'pred_a':absolute<=1e-4 and relative<=1e-5 and max(norm_errors)<=1e-5 and max(sum_errors)<=1e-6 and outside==0 and max(other_errors)==0 and max(errors)<=1e-5 and bool(torch.isfinite(v).all()) and calls==666 and not torch.cuda.is_initialized() and min(x for pair in groups_out['all']['random_single_over_joint'] for x in pair)>=.01,
 'pred_b':all(g['real_interaction']<=.35 for g in groups_out.values()) and min(groups_out['all']['single_over_joint'])>=.10,
 'pred_c':groups_out['all']['real_interaction']<=groups_out['all']['random_median'],
 'groups':groups_out,'anchor_max_abs':absolute,'anchor_relative':relative,'max_piece_norm_error':max(norm_errors),'max_sum_error':max(sum_errors),'max_subtraction_relative_error':max(errors),'other_head_error':max(other_errors),'outside':outside,'body_forwards':1480,'batched_block_calls':calls,'seconds':time.perf_counter()-start,'scope':'Opened same-total-write random split specificity with both piece norms preserved. No freshjoint confirmation, unique decomposition or whole-circuit simplicity claim.','source_shas':binding}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['source_shas','groups']},indent=2));signal.alarm(0)
if __name__=='__main__':main()
