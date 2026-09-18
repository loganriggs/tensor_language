"""Exact seven-term native head9.8 response fold; no downstream intervention."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_FINEWEB_HEAD9_FOLD_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_ATTENTION7_DROP3_FINEWEB_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('80sequence equivalents;36block calls;CPU seven-term fold');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();signal.alarm(180)
 out=P/(STEM+'_RESULT.json');assert not out.exists()
 from fastload import load_model_fast
 model=load_model_fast().eval();old=torch.load(P/'CITY_DROP3_FINEWEB_INTERCHANGE_V1_ARTIFACT.pt',weights_only=True)
 census=torch.load(P/'CITY_FINEWEB_RESPONSE_V1_ARTIFACT.pt',weights_only=True)
 ids=torch.tensor([r['ids'] for r in groups]);assert ids.shape==(40,32)
 edit=torch.cat([f['native_delta'] for f in old['fixtures']]);state={'arm':0};saved={}
 def post8(module,args,out):return (out[0]+edit,out[1]) if state['arm'] else out
 h=model.transformer.h[8].attn.register_forward_hook(post8);attn=model.transformer.h[9].attn;original=attn.squared_attention
 def capture(module,q,k,v,q2,k2):
  saved[state['arm']]={'q':q[:,-1,8].double().clone(),'q2':q2[:,-1,8].double().clone(),'k':k[:,:,8].double().clone(),'k2':k2[:,:,8].double().clone(),'v':v[:,:,8].double().clone()}
  return original(q,k,v,q2,k2)
 attn.squared_attention=MethodType(capture,attn);values=torch.zeros(2,40,10,dtype=torch.float64);calls=0
 try:
  for arm in range(2):
   state['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
   for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
   scores=30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30)
   for i,row in enumerate(groups):
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=scores[i,left]-scores[i,right]
 finally:h.remove();attn.squared_attention=original
 scores=expand(values,mapping,6);diff=scores-old['values'][[0,1]];anchor=float(diff.abs().max());anchor_rel=float(diff.norm()/old['values'][[0,1]].norm())
 qerr=max(float((saved[1][k]-saved[0][k]).norm()/saved[0][k].norm()) for k in ['q','q2'])
 a=[(saved[t]['q'][:,None]*saved[t]['k']).sum(-1)/128 for t in [0,1]]
 b=[(saved[t]['q2'][:,None]*saved[t]['k2']).sum(-1)/128 for t in [0,1]]
 av,bv,v=a[0],b[0],saved[0]['v'];da,db,dv=a[1]-av,b[1]-bv,saved[1]['v']-v
 terms=torch.stack([da[...,None]*bv[...,None]*v,av[...,None]*db[...,None]*v,av[...,None]*bv[...,None]*dv,da[...,None]*db[...,None]*v,da[...,None]*bv[...,None]*dv,av[...,None]*db[...,None]*dv,da[...,None]*db[...,None]*dv])
 names=['deltaK1','deltaK2','deltaV','deltaK1_deltaK2','deltaK1_deltaV','deltaK2_deltaV','deltaK1_deltaK2_deltaV']
 final=census['final_states'];rho=(final.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();gamma=1.
 for layer in range(10,18):gamma*=float(model.transformer.h[layer].lambdas[0])
 contributions=torch.zeros(7,40,32,10,dtype=torch.float64)
 for i,row in enumerate(groups):
  w=model.lm_head.weight[torch.tensor(row['endpoint_pairs']+CONTROL_PAIRS)].double()
  u=torch.einsum('ad,jkd->ajk',final[:,i]/rho[:,i],w);du=u[1]-u[0];capped=30*torch.tanh(u/30)
  secant=torch.where(du.abs()>1e-12,(capped[1]-capped[0])/du,1-torch.tanh(u[0]/30).square());reader=secant[:,0,None]*w[:,0]-secant[:,1,None]*w[:,1]
  folded=gamma/rho[1,i]*(reader@attn.c_proj.weight[:,1024:1152].double())
  contributions[:,i]=terms[:,i]@folded.T
 c=expand(contributions.sum(2),mapping,6);ref=census['contributions'][8];err=c.sum(0)-ref
 prior=json.loads((P/'CITY_FINEWEB_RESPONSE_V1_RESULT.json').read_text());reversed_docs=set(prior['reversed_documents']);records={}
 for label,idx in [('reversed',[i for i,r in enumerate(rows) if r['context_id'] in reversed_docs]),('other',[i for i,r in enumerate(rows) if r['context_id'] not in reversed_docs])]:
  target=ref[idx,0];parts=c[:,idx,0];route=parts[[0,1,3,4,5,6]].sum(0)
  records[label]={'routing_only_error':float((route-target).norm()/target.norm()),'triple_over_head':float(parts[6].norm()/target.norm()),'terms':{name:{'norm_ratio':float(part.norm()/target.norm()),'aligned_fraction':float(part.dot(target)/target.square().sum())} for name,part in zip(names,parts)}}
 r={'pred_a':anchor<=1e-4 and anchor_rel<=1e-5 and qerr<=1e-6 and float(err.abs().max())<=1e-4 and float(err.norm()/ref.norm())<=1e-4 and bool(torch.isfinite(c).all()) and calls==36 and not torch.cuda.is_initialized(),'pred_b':all(g['routing_only_error']<=.35 for g in records.values()),'pred_c':all(g['triple_over_head']<=.10 for g in records.values()),'groups':records,'anchor_max_abs':anchor,'anchor_relative':anchor_rel,'query_relative_change':qerr,'closure_max_abs':float(err.abs().max()),'closure_relative':float(err.norm()/ref.norm()),'body_forwards':80,'batched_block_calls':calls,'seconds':time.perf_counter()-start,'scope':'Opened exact fold of head9.8 finite-change response. Native fields and final response-conditioned reader supplied; not an independent causal component or extracted token-only model.','source_shas':binding}
 torch.save({'values':scores,'contributions':c,'position_contributions':contributions,'term_names':names,'captured_fields':saved},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
