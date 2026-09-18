"""Registered CPU finite-change suffix accounting with attention9 split by head."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_FINEWEB_RESPONSE_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_ATTENTION7_DROP3_FINEWEB_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('80sequence equivalents;36block calls;CPU native and exact swap');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();signal.alarm(180)
 out=P/(STEM+'_RESULT.json');assert not out.exists()
 from fastload import load_model_fast
 model=load_model_fast().eval();old=torch.load(P/'CITY_DROP3_FINEWEB_INTERCHANGE_V1_ARTIFACT.pt',weights_only=True)
 ids=torch.tensor([r['ids'] for r in groups]);assert ids.shape==(40,32)
 edit=torch.cat([f['native_delta'] for f in old['fixtures']]);state={'arm':0};saved={};handles=[]
 def post8(module,args,out):return (out[0]+edit,out[1]) if state['arm'] else out
 handles.append(model.transformer.h[8].attn.register_forward_hook(post8))
 def capture_mlp(layer):
  def hook(module,args,value):saved[(state['arm'],layer,'mlp')]=value[:,-1].double().clone()
  return hook
 def capture_attention(layer):
  def hook(module,args,value):saved[(state['arm'],layer,'attention')]=value[0][:,-1].double().clone()
  return hook
 def capture_block(layer):
  def hook(module,args,value):saved[(state['arm'],layer,'residual')]=value[0][:,-1].double().clone()
  return hook
 for layer in range(8,18):
  b=model.transformer.h[layer];handles.extend([b.mlp.register_forward_hook(capture_mlp(layer)),b.attn.register_forward_hook(capture_attention(layer)),b.register_forward_hook(capture_block(layer))])
 def heads(module,args):
  channels=args[0][:,-1].double().reshape(40,9,128);w=module.weight.double().reshape(1152,9,128)
  saved[(state['arm'],9,'heads')]=torch.einsum('bhi,dhi->hbd',channels,w)
 handles.append(model.transformer.h[9].attn.c_proj.register_forward_pre_hook(heads))
 values=torch.zeros(2,40,10,dtype=torch.float64);calls=0
 try:
  for arm in range(2):
   state['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
   for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
   scores=30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30)
   for i,row in enumerate(groups):
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=scores[i,left]-scores[i,right]
 finally:
  for h in handles:h.remove()
 v=expand(values,mapping,6);anchor=v-old['values'][[0,1]];absolute=float(anchor.abs().max());relative=float(anchor.norm()/old['values'][[0,1]].norm())
 gamma=1.;scales={}
 for l in range(17,8,-1):scales[l]=gamma;gamma*=float(model.transformer.h[l].lambdas[0])
 raw=[];names=[]
 for h in range(9):raw.append(scales[9]*(saved[(1,9,'heads')][h]-saved[(0,9,'heads')][h]));names.append(f'attention9.head{h}')
 for l in range(9,18):
  for name in (['mlp'] if l==9 else ['attention','mlp']):raw.append(scales[l]*(saved[(1,l,name)]-saved[(0,l,name)]));names.append(f'{name}{l}')
 raw=torch.stack(raw);final=torch.stack([saved[(a,17,'residual')] for a in range(2)]);delta=final[1]-final[0]
 residual_error=float((raw.sum(0)-delta).norm()/delta.norm());head_error=max(float((saved[(a,9,'heads')].sum(0)-saved[(a,9,'attention')]).norm()/saved[(a,9,'attention')].norm()) for a in range(2))
 block8=float((saved[(1,8,'residual')]-saved[(0,8,'residual')]).abs().max())
 rho=(final.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
 normalized=torch.cat([raw/rho[1][None],(final[0]*(1/rho[1]-1/rho[0]))[None]]);names.append('output_normalization')
 contributions=torch.zeros(len(names),40,10,dtype=torch.float64)
 for i,row in enumerate(groups):
  w=model.lm_head.weight[torch.tensor(row['endpoint_pairs']+CONTROL_PAIRS)].double()
  u=torch.einsum('ad,jkd->ajk',final[:,i]/rho[:,i],w);du=u[1]-u[0];cap=30*torch.tanh(u/30)
  secant=torch.where(du.abs()>1e-12,(cap[1]-cap[0])/du,1-torch.tanh(u[0]/30).square());reader=secant[:,0,None]*w[:,0]-secant[:,1,None]*w[:,1]
  contributions[:,i]=normalized[:,i]@reader.T
 c=expand(contributions,mapping,6);effect=v[1]-v[0];err=c.sum(0)-effect
 prior=json.loads((P/'CITY_DROP3_FINEWEB_INTERCHANGE_V1_RESULT.json').read_text());reversed_docs={d['context_id'] for d in prior['document_diagnostics'] if any(a<0 for a in d['native_attenuation'])}
 records={}
 for label,idx in [('reversed',[i for i,r in enumerate(rows) if r['context_id'] in reversed_docs]),('other',[i for i,r in enumerate(rows) if r['context_id'] not in reversed_docs])]:
  target=effect[idx,0];parts=c[:,idx,0]
  records[label]={'attention9_error':float((parts[:9].sum(0)-target).norm()/target.norm()),'early_error':float((parts[:10].sum(0)+parts[-1]-target).norm()/target.norm()),'terms':{name:{'norm_ratio':float(part.norm()/target.norm()),'aligned_fraction':float(part.dot(target)/target.square().sum())} for name,part in zip(names,parts)}}
 docs=[]
 for context in sorted({r['context_id'] for r in rows}):
  idx=[i for i,r in enumerate(rows) if r['context_id']==context];b=v[0,idx,0][::2]-v[0,idx,0][1::2];capable=b>=.1
  part=c[:,idx,0];docs.append({'context_id':context,'capable_count':int(capable.sum()),'paired_attenuation_contributions':{name:((-(x[::2]-x[1::2]))[capable]/b[capable]).tolist() for name,x in zip(names,part)}})
 r={'pred_a':absolute<=1e-4 and relative<=1e-5 and residual_error<=1e-4 and float(err.abs().max())<=1e-4 and float(err.norm()/effect.norm())<=1e-3 and head_error<=1e-4 and block8==0 and bool(torch.isfinite(c).all()) and calls==36 and not torch.cuda.is_initialized(),'pred_b':all(x['attention9_error']<=.35 for x in records.values()),'pred_c':all(x['early_error']<=.35 for x in records.values()),'groups':records,'documents':docs,'reversed_documents':sorted(reversed_docs),'anchor_max_abs':absolute,'anchor_relative':relative,'residual_error':residual_error,'head9_closure':head_error,'block8_last_token_delta':block8,'margin_max_abs':float(err.abs().max()),'margin_relative':float(err.norm()/effect.norm()),'body_forwards':80,'batched_block_calls':calls,'seconds':time.perf_counter()-start,'scope':'Opened native-swap response attribution; no downstream intervention or sufficiency proof.','source_shas':binding}
 torch.save({'values':v,'contributions':c,'term_names':names,'transported_raw_responses':raw,'final_states':final},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['groups','documents','source_shas']},indent=2));signal.alarm(0)
if __name__=='__main__':main()
