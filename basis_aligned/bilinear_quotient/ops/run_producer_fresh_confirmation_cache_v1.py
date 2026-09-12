#!/usr/bin/env python3
# BQGATE:8 body forwards;48 sequences18-22tokens;cache and native capability;180sec;under16MB.
"""pred_a producer fold/headsum and residual/RMS replay <=1e-5 relative.
pred_b eachtemplate native cue mean>=.2 and >=10/12 positive contrasts.
pred_c finite complete cache with all48rows, same frozen weights and three producers.
Null: native capability absent or instrument invalid; no semantic circuit conclusion.
Price8nativebodyforwards, no optimization,180seconds,under16MBcache.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='PRODUCER_FRESH_CONFIRMATION_CACHE_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];validate(rows);batches=[]
 for family in range(2):
  for length in sorted(set(len(r['ids']) for r in rows)):
   ids=[i for i,r in enumerate(rows) if r['family']==family and len(r['ids'])==length]
   batches.extend((length,ids[o:o+8]) for o in range(0,len(ids),8))
 assert len(rows)==48 and len(batches)==8 and max(n for n,_ in batches)==22
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('8bodyforwards48freshsequences18-22tokens; fixedproducer/cache/nativecapability; nofit');return
 result_path=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not result_path.exists() and not art.exists()
 tic=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 from jacclust.tt_model import apply_rotary_emb
 from compiled_mixed_token_head_v1 import execute_mixed_token_head
 from folded_normalized_router_v1 import rotary
 model=load_model_fast().cuda().eval();state=model.state_dict();last=model.transformer.h[17]
 p={k:v.cuda() for k,v in torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()};C=p['current_readers'];folds={}
 for layer in (8,9,13):
  prefix=f'transformer.h.{layer}.attn.';scale=1.
  for j in range(layer+1,18):scale*=float(state[f'transformer.h.{j}.lambdas'][0])
  mix=float(state[prefix+'lamb']);out=(scale*C@state[prefix+'c_proj.weight'].double()).reshape(4,9,128)
  E=torch.einsum('ahk,hkd->ahd',out,(1-mix)*state[prefix+'c_v.weight'].double().reshape(9,128,1152));H=torch.einsum('ahk,hkd->ahd',out,mix*state['transformer.h.0.attn.c_v.weight'].double().reshape(9,128,1152));folds[layer]=(scale,E,H)
 current_cache=torch.zeros(48,22,1152);position_writes=torch.zeros_like(current_cache);heads=torch.zeros(27,48,22,4,dtype=torch.float64);group=torch.zeros(48,22,4,dtype=torch.float64);rho_cache=torch.zeros(48,22,1,dtype=torch.float64);pre=torch.zeros(48,1152);margins=torch.zeros(48,3,dtype=torch.float64);errors=[];rhos=[];captured={}
 handles=[model.transformer.h[0].attn.register_forward_pre_hook(lambda module,args:captured.update(first=args[0].detach()))]
 def hook(layer,module,args,output):
  x=args[0];n,length,_=x.shape;scale,E,H=folds[layer]
  q,k,q2,k2=[getattr(module,name)(x).reshape(n,length,9,128) for name in ('c_q','c_k','c_q2','c_k2')];cos,sin=module.rotary(q)
  q,k,q2,k2=[apply_rotary_emb(F.rms_norm(v,(128,)),cos,sin).double() for v in (q,k,q2,k2)]
  gamma=(torch.einsum('nthd,nshd->nhts',q,k)/128)*(torch.einsum('nthd,nshd->nhts',q2,k2)/128);gamma*=torch.ones(length,length,device='cuda',dtype=torch.bool).tril()[None,None]
  values=torch.einsum('nsd,ahd->nsha',x.double(),E)+torch.einsum('nsd,ahd->nsha',captured['first'].double(),H)
  contributions=torch.einsum('nhts,nsha->ntha',gamma,values);native=scale*(output[0].double()@C.T)
  errors.append(float((contributions.sum(2)-native).norm()/native.norm()));captured['heads'][layer]=contributions;captured['group'].append(native)
 for layer in (8,9,13):handles.append(model.transformer.h[layer].attn.register_forward_hook(lambda module,args,out,layer=layer:hook(layer,module,args,out)))
 try:
  for length,ids in batches:
   captured['heads']={};captured['group']=[];tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda');x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
   for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
   residual=last.lambdas[0]*x+last.lambdas[1]*x0;current=F.rms_norm(residual,(1152,));attention,_=last.attn(current,v1);z=(residual+attention)[:,-1];rho=(residual.double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
   rhos.append(float((residual.double()/rho-current).norm()/current.norm()));heads[:,ids,:length]=torch.cat([captured['heads'][j] for j in (8,9,13)],2).permute(2,0,1,3).cpu();group[ids,:length]=sum(captured['group']).cpu();current_cache[ids,:length]=current.cpu();rho_cache[ids,:length]=rho.cpu();pre[ids]=z.cpu()
   qr=rotary(length-1,128).cuda()
   for pos in range(length):position_writes[ids,pos]=execute_mixed_token_head(current[:,-1],current[:,pos],tokens[:,pos],(qr.T@rotary(pos,128).cuda()).float(),p).cpu()
   h=z+last.mlp(F.rms_norm(z,(1152,)));logits=30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
   for local,index in enumerate(ids):
    row=rows[index];pairs=[(row['uk_id'],row['us_id']),row['control_ids'],row['newline_control_ids']];margins[index]=torch.stack([logits[local,a]-logits[local,b] for a,b in pairs]).double().cpu()
 finally:
  for handle in handles:handle.remove()
 head_error=float((heads.sum(0)-group).norm()/group.norm());cells=[]
 for family in range(2):
  uk=[i for i,r in enumerate(rows) if r['family']==family and r['cue']=='British'];us=[i^1 for i in uk];difference=margins[uk,0]-margins[us,0]
  cells.append(dict(family=family,mean_native_cue_contrast=float(difference.mean()),positive_native_contrasts=int((difference>0).sum()),count=len(uk)))
 finite=all(bool(torch.isfinite(t).all()) for t in (heads,group,rho_cache,pre,current_cache,position_writes,margins))
 torch.save(dict(head_reads=heads,group_reads=group,rho=rho_cache,pre=pre,current_states=current_cache,position_writes=position_writes,baseline_margins=margins),art)
 result={'pred_a':max(errors+rhos+[head_error])<=1e-5,'pred_b':all(c['mean_native_cue_contrast']>=.2 and c['positive_native_contrasts']>=10 for c in cells),'pred_c':finite,'producer_replay_max':max(errors),'head_sum_replay':head_error,'rho_replay_max':max(rhos),'capability_cells':cells,'seconds':time.perf_counter()-tic,'body_forwards':len(batches),'artifact_bytes':art.stat().st_size,'artifact_sha':digest(art),'source_shas':binding,'scope':'Fresh fixed panel native cache/capability only; full QK and states retained. Behavioral confirmation separately registered; no data fitting or capability-based row selection.'}
 result_path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
