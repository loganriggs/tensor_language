"""Fresh native-weight value mediation with independent MLP8 reference and16nulls."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
from mlp8_current_value_response_v1 import execute
STEM='CITY_VALUE_PATH_FRESH_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 audit=json.loads((P/(STEM+'_ROW_AUDIT.json')).read_text());assert audit['passes'] and audit['row_sha256']==hashlib.sha256((P/(STEM+'_ROWS.json')).read_bytes()).hexdigest()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('200sequence equivalents;90block calls;5arms;CPU fresh value paths');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();signal.alarm(180)
 out=P/(STEM+'_RESULT.json');assert not out.exists()
 from fastload import load_model_fast
 model=load_model_fast().eval();program=torch.load(P/'CITY_FINEWEB_MLP8_VALUE_FOLD_V1_PROGRAM.pt',weights_only=True)
 ids=torch.tensor([r['ids'] for r in groups]);assert ids.shape==(40,32)
 city=groups[0]['city_position'];masks=torch.zeros(40,32,dtype=torch.bool)
 for i,row in enumerate(groups):masks[i,row['destination_positions']]=True;assert row['city_position']==city
 state={'arm':0};cache={};contexts={};changes={};errors=[];other_errors=[];norm_errors=[]
 attn8=model.transformer.h[8].attn;original8=attn8.squared_attention
 def capture8(module,q,k,v,q2,k2):
  if state['arm']==0:
   donor=torch.arange(40)^1
   route=((q[:,:,2]*k[:,city,None,2]).sum(-1)/128)*((q2[:,:,2]*k2[:,city,None,2]).sum(-1)/128)
   dr=((q[:,:,2]*k[donor,city,None,2]).sum(-1)/128)*((q2[:,:,2]*k2[donor,city,None,2]).sum(-1)/128)
   cache['edit']=F.linear(dr[...,None]*v[donor,city,None,2]-route[...,None]*v[:,city,None,2],module.c_proj.weight[:,256:384])*masks[...,None]
  return original8(q,k,v,q2,k2)
 attn8.squared_attention=MethodType(capture8,attn8)
 def pre8(module,args):
  if state['arm']<2:contexts[(state['arm'],'mixed8')]=(module.lambdas[0]*args[0]+module.lambdas[1]*args[2]).clone()
 def post8(module,args,out):return (out[0]+cache['edit'],out[1]) if state['arm'] else out
 def postz(module,args,out):
  if state['arm']<2:contexts[(state['arm'],'z')]=(contexts[(state['arm'],'mixed8')]+out[0]).clone()
 def mlp8(module,args,out):
  if state['arm']<2:contexts[(state['arm'],'mlp8')]=out.clone()
 def pre9(module,args):
  if state['arm']<2:contexts[(state['arm'],'mixed9')]=(module.lambdas[0]*args[0]+module.lambdas[1]*args[2]).clone()
 handles=[model.transformer.h[8].register_forward_pre_hook(pre8),attn8.register_forward_hook(post8),attn8.register_forward_hook(postz),model.transformer.h[8].mlp.register_forward_hook(mlp8),model.transformer.h[9].register_forward_pre_hook(pre9)]
 attn9=model.transformer.h[9].attn;original9=attn9.squared_attention
 def changed(module,q,k,v,q2,k2):
  arm=state['arm']
  if arm>=2:
   new=v.clone();new[:,:,8]-=changes[arm]
   actual=v[:,:,8]-new[:,:,8];errors.append(float((actual.double()-changes[arm].double()).norm()/changes[arm].double().norm()))
   other_errors.append(float((new[:,:,:8]-v[:,:,:8]).abs().max()));v=new
  return original9(q,k,v,q2,k2)
 attn9.squared_attention=MethodType(changed,attn9)
 values=torch.zeros(5,40,10,dtype=torch.float64);calls=0;fixtures=[];local_errors=[];outside=None
 try:
  for arm in range(5):
   state['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
   for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
   scores=30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30)
   for i,row in enumerate(groups):
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=scores[i,left]-scores[i,right]
   if arm==1:
    inputs={'z':contexts[(0,'z')],'delta':cache['edit'],'mixed9_native':contexts[(0,'mixed9')],'mixed9_edited':contexts[(1,'mixed9')]}
    pieces=execute(program,**inputs);candidate=pieces[1:5].sum(0)
    rho=(contexts[(1,'mixed9')].double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
    native=(1-attn9.lamb.double())*model.transformer.h[9].lambdas[0].double()/rho*F.linear(contexts[(1,'mlp8')].double()-contexts[(0,'mlp8')].double(),attn9.c_v.weight[1024:1152].double())
    direct=(1-attn9.lamb.double())*model.transformer.h[9].lambdas[0].double()/rho*F.linear(cache['edit'].double(),attn9.c_v.weight[1024:1152].double())
    joint_reference=(1-attn9.lamb.double())*model.transformer.h[9].lambdas[0].double()/rho*F.linear(cache['edit'].double()+contexts[(1,'mlp8')].double()-contexts[(0,'mlp8')].double(),attn9.c_v.weight[1024:1152].double())
    local_errors=[float((c-n).norm()/n.norm()) for c,n in zip(candidate+direct,joint_reference)]
    changes[2]=direct.float();changes[3]=candidate.float();changes[4]=(direct+candidate).float()
    outside=max(float(d[~masks].abs().max()) for d in changes.values())
    preflight={'passes':max(local_errors)<=1e-4 and outside==0,'max_local_error':max(local_errors),'local_errors':local_errors,'outside':outside}
    (P/(STEM+'_PREFLIGHT.json')).write_text(json.dumps(preflight,indent=2)+'\n');assert preflight['passes'],preflight
    fixtures=[{'inputs':{k:v[i:i+1] for k,v in inputs.items()},'native_joint_value_delta':joint_reference[i:i+1],'direct_value_delta':direct[i:i+1],'mlp8_value_delta':candidate[i:i+1]} for i in range(40)]
   print(f'arm{arm+1}/5 elapsed{time.perf_counter()-start:.2f}s',flush=True)
 finally:
  for h in handles:h.remove()
  attn8.squared_attention=original8;attn9.squared_attention=original9
 v=expand(values,mapping,6);total=v-v[:1];e=v-v[1:2]
 base=v[0,:,0][::2]-v[0,:,0][1::2];cap=base>=.1;records={}
 for arm,name in [(1,'swap'),(2,'minus_direct'),(3,'minus_mlp8'),(4,'minus_joint')]:
  pair=v[arm,:,0][::2]-v[arm,:,0][1::2];atten=(base[cap]-pair[cap])/base[cap]
  records[name]={'positive_fraction':float((atten>0).double().mean()),'mean_attenuation':float(atten.mean())}
 documents=[];reversed_docs=[]
 def metric(idx):
  if not idx:return {'rows':0}
  a,b,j=[e[k,idx,0] for k in [2,3,4]];parent=total[1,idx,0];rms=e[4,idx].square().mean(0).sqrt()
  return {'rows':len(idx),'interaction_over_smaller':float((j-a-b).norm()/min(a.norm(),b.norm())),'single_over_joint':[float(a.norm()/j.norm()),float(b.norm()/j.norm())],'joint_change_over_parent':float(j.norm()/parent.norm()),'joint_target_rms':float(rms[0]),'control_over_joint_target':(rms[1:]/rms[0]).tolist(),'parent_suppression_aligned_fraction':float(-j.dot(parent)/parent.square().sum())}
 for context in range(20):
  idx=[i for i,r in enumerate(rows) if r['context_id']==context];b=v[0,idx,0][::2]-v[0,idx,0][1::2];capable=b>=.1
  a={name:((b[capable]-(v[arm,idx,0][::2]-v[arm,idx,0][1::2])[capable])/b[capable]).tolist() for arm,name in [(1,'swap'),(2,'minus_direct'),(3,'minus_mlp8'),(4,'minus_joint')]}
  if any(x<0 for x in a['swap']):reversed_docs.append(context)
  documents.append({'context_id':context,'capable_pairs':int(capable.sum()),'attenuations':a,**metric(idx)})
 groups_out={label:metric(idx) for label,idx in [('all',list(range(240))),('reversed',[i for i,r in enumerate(rows) if r['context_id'] in reversed_docs]),('other',[i for i,r in enumerate(rows) if r['context_id'] not in reversed_docs])]}
 g=groups_out['all']
 r={'pred_a':audit['passes'] and len(rows)==240 and max(local_errors)<=1e-4 and outside==0 and max(other_errors)==0 and max(errors)<=1e-5 and bool(torch.isfinite(v).all()) and calls==90 and not torch.cuda.is_initialized(),
 'pred_b':g['interaction_over_smaller']<=.35,
 'pred_c':min(g['single_over_joint'])>=.10 and all(groups_out[k]['rows']>0 and groups_out[k]['interaction_over_smaller']<=.35 for k in ['reversed','other']),
 'pred_d':g['joint_change_over_parent']>=.10 and g['joint_target_rms']>=1e-5 and max(g['control_over_joint_target'])<=.5,
 'pred_e':g['parent_suppression_aligned_fraction']>=.20,
 'groups':groups_out,'arms':records,'capable_pairs':int(cap.sum()),'documents':documents,'max_local_error':max(local_errors),'max_subtraction_relative_error':max(errors),'other_head_error':max(other_errors),'outside':outside,'body_forwards':200,'batched_block_calls':calls,'seconds':time.perf_counter()-start,'scope':'Fresh direct/MLP8 value-path composition and suppression of parent-swap transmission. Native normalization contexts/fullsuffix; no fresh random-split specificity, unique semantic-unit or directional-rescue claim.','source_shas':binding}
 torch.save({'values':v,'fixtures':fixtures},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['source_shas','documents']},indent=2));signal.alarm(0)
if __name__=='__main__':main()
