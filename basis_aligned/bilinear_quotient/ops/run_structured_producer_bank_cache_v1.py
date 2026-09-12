#!/usr/bin/env python3
# BQGATE:12 body forwards;96 sequences21-24tokens;cache only;180sec;no optimization.
"""pred_a folded four-read producer replay <=1e-5 relative each batch/layer.
pred_b native pre/current replay <=1e-5 relative and rho reconstruction <=1e-5.
pred_c projector idempotence <=1e-12 and retained/complement sum <=1e-12.
Null: invalid native folding instrument prevents behavioral interpretation.
Price: 12 native body forwards, cache under3MB, 180sec cap; all native weights retained.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='STRUCTURED_PRODUCER_BANK_CACHE_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(digest(k)==v for k,v in binding.items())
 rows=sum([json.loads((P/(s+'_ROWS.json')).read_text())['rows'] for s in ('REGIONAL_COMPETING_CUES_V1','REGIONAL_CITY_ROLE_CROSSOVER_V1')],[])
 validate(rows);batches=[]
 for assignment in range(2):
  for length in sorted(set(len(r['ids']) for r in rows)):
   ids=[i for i in range(48*assignment,48*(assignment+1)) if len(rows[i]['ids'])==length]
   batches.extend((length,ids[o:o+8]) for o in range(0,len(ids),8))
 assert len(rows)==96 and len(batches)==12 and max(x[0] for x in batches)==24
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print('12 body forwards;96 fixed contexts;three producer four-reading cache; no text fit');return
 result_path=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt')
 assert not result_path.exists() and not artifact.exists()
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;signal.alarm(180);tic=time.perf_counter()
 from fastload import load_model_fast
 from jacclust.tt_model import apply_rotary_emb
 model=load_model_fast().cuda().eval();state=model.state_dict();last=model.transformer.h[17]
 program=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
 C=program['current_readers'].cuda().double();assert C.shape==(4,1152)
 cached=torch.load(P/'REGIONAL_SOURCE_POSITIONS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 folds={};projector=None
 for layer in (8,9,13):
  prefix=f'transformer.h.{layer}.attn.';scale=1.
  for j in range(layer+1,18):scale*=float(state[f'transformer.h.{j}.lambdas'][0])
  mix=float(state[prefix+'lamb']);out=(scale*C@state[prefix+'c_proj.weight'].double()).reshape(4,9,128)
  E=torch.einsum('ahk,hkd->ahd',out,(1-mix)*state[prefix+'c_v.weight'].double().reshape(9,128,1152))
  H=torch.einsum('ahk,hkd->ahd',out,mix*state['transformer.h.0.attn.c_v.weight'].double().reshape(9,128,1152))
  folds[layer]=(scale,E,H)
  if layer==13:
   M=torch.cat([E[:,0],H[:,0]],1);u,s,v=torch.linalg.svd(M,full_matrices=False);projector=u[:,:1]@u[:,:1].T
 head_reads=torch.zeros(27,96,24,4,dtype=torch.float64)
 reads=torch.zeros(3,96,24,4,dtype=torch.float64);rhos=torch.zeros(96,24,1,dtype=torch.float64)
 pre=torch.zeros(96,1152);checks=[];state_checks=[];rho_checks=[];captured={}
 handles=[model.transformer.h[0].attn.register_forward_pre_hook(lambda module,args:captured.update(first=args[0].detach()))]
 def hook(layer,module,args,output):
  inp=args[0];n,length,_=inp.shape;scale,E,H=folds[layer]
  q,k,q2,k2=[getattr(module,name)(inp).reshape(n,length,9,128) for name in ('c_q','c_k','c_q2','c_k2')]
  cos,sin=module.rotary(q)
  q,k,q2,k2=[apply_rotary_emb(F.rms_norm(v,(128,)),cos,sin).double() for v in (q,k,q2,k2)]
  gamma=(torch.einsum('nthd,nshd->nhts',q,k)/128)*(torch.einsum('nthd,nshd->nhts',q2,k2)/128)
  gamma*=torch.ones(length,length,device='cuda',dtype=torch.bool).tril()[None,None]
  values=torch.einsum('nsd,ahd->nsha',inp.double(),E)+torch.einsum('nsd,ahd->nsha',captured['first'].double(),H)
  folded=torch.einsum('nhts,nsha->nta',gamma,values)
  native=scale*(output[0].double()@C.T)
  checks.append(dict(layer=layer,error=float((folded-native).norm()/native.norm())))
  captured['group'].append(native)
  captured['all_heads'][layer]=torch.einsum('nhts,nsha->ntha',gamma,values)
  if layer==13:
   head=torch.einsum('nts,nsa->nta',gamma[:,0],values[:,:,0]);captured['head']=head;captured['rank1']=head@projector
 for layer in (8,9,13):handles.append(model.transformer.h[layer].attn.register_forward_hook(lambda module,args,out,layer=layer:hook(layer,module,args,out)))
 try:
  for length,ids in batches:
   captured['group']=[];captured['all_heads']={};tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda')
   x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
   for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
   residual=last.lambdas[0]*x+last.lambdas[1]*x0;current=F.rms_norm(residual,(1152,))
   attention,_=last.attn(current,v1);native_pre=(residual+attention)[:,-1]
   rho=(residual.double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
   for a,b in [(native_pre.cpu(),cached['pre'][ids]),(current.cpu(),cached['current_states'][ids,:length])]:state_checks.append(float((a-b).norm()/b.norm()))
   rho_checks.append(float((residual.double()/rho-current.double()).norm()/current.norm()))
   reads[:,ids,:length]=torch.stack([sum(captured['group']),captured['head'],captured['rank1']]).cpu()
   head_reads[:,ids,:length]=torch.cat([captured['all_heads'][j] for j in (8,9,13)],2).permute(2,0,1,3).cpu()
   rhos[ids,:length]=rho.cpu();pre[ids]=native_pre.cpu()
 finally:
  for h in handles:h.remove()
 projection_error=float((projector@projector-projector).norm()/projector.norm())
 complement=reads[1]-reads[2];sum_error=float((reads[2]+complement-reads[1]).norm()/reads[1].norm())
 head_sum_error=float((head_reads.sum(0)-reads[0]).norm()/reads[0].norm())
 torch.save(dict(head_reads=head_reads,reads=reads,rho=rhos,pre=pre,projector=projector.cpu(),value_singular_values=s.cpu(),arm_names=['three_producer_group','head13_0','head13_0_value_rank1']),artifact)
 result={'pred_a':max(r['error'] for r in checks)<=1e-5 and head_sum_error<=1e-5,'pred_b':max(state_checks+rho_checks)<=1e-5,'pred_c':max(projection_error,sum_error)<=1e-12,'head_sum_replay':head_sum_error,'producer_replay':checks,'state_replay_max':max(state_checks),'rho_replay_max':max(rho_checks),'projector_idempotence':projection_error,'retained_complement_sum_error':sum_error,'body_forwards':len(batches),'seconds':time.perf_counter()-tic,'artifact_bytes':artifact.stat().st_size,'artifact_sha':digest(artifact),'source_shas':binding,'scope':'Native producer contribution to four downstream source readings. Full routing and native states retained; cache instrument only, behavioral tests separately registered.'}
 result_path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
