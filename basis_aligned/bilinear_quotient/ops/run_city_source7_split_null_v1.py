#!/usr/bin/env python3
# BQGATE:1440bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a replay abs<=1e-4/rel<=1e-5 and live random pieces>=.01full;
pred_b real interaction<=.35; pred_c real interaction<=16random split median.
1440forwards; balanced coordinate partitions of the same full write.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_SOURCE7_SPLIT_NULL_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_FULL_PILE_V2_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_NO_MODEL') or os.environ.get('BQLIB_DRYRUN'):
  print('1440bodyforwards;40prefixes;source7 partition and16randomsplits');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval()
 writes=[w.cuda() for w in torch.load(P/'CITY_SOURCE7_EDIT_V1_WRITES.pt',weights_only=True)['writes']]
 sum_error=max(float((w[1]+w[2]-w[0]).norm()/w[0].norm()) for w in writes)
 outside=max(float(w[:,:, [i for i in range(len(r['ids'])) if i not in r['destination_positions']]].abs().max()) for w,r in zip(writes,groups))
 masks=[]
 for k in range(16):
  mask=torch.zeros(1152,dtype=torch.bool,device='cuda');order=torch.randperm(1152,generator=torch.Generator().manual_seed(18094900+k));mask[order[:576].cuda()]=True;masks.append(mask)
 state={}
 def post8(module,args,out):
  arm=state['arm'];i=state['i']
  if arm==0:return out
  if arm<=3:d=writes[i][arm-1].to(out[0].dtype)
  else:
   k=(arm-4)//2;mask=masks[k] if (arm-4)%2==0 else ~masks[k]
   d=(writes[i][0]*mask).to(out[0].dtype)
  return out[0]+d,out[1]
 handle=model.transformer.h[8].attn.register_forward_hook(post8)
 values=torch.zeros(36,40,10,dtype=torch.float64);count=0
 try:
  for arm in range(36):
   state['arm']=arm
   for i,row in enumerate(groups):
    state['i']=i;ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
    for block in model.transformer.h:x,first=block(x,first,x0)
    scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=(scores[left]-scores[right]).cpu()
    count+=1
 finally:handle.remove()
 v=expand(values,mapping,6);e=v-v[:1];old=torch.load(P/'CITY_SOURCE7_EDIT_V1_ARTIFACT.pt',weights_only=True)['values']
 diff=v[:4]-old[:4];absolute=float(diff.abs().max());relative=float(diff.norm()/old[:4].norm())
 rms=e.square().mean(1).sqrt();base=v[0,:,0][::2]-v[0,:,0][1::2];cap=base>=.1;records={}
 for arm,name in [(1,'full'),(2,'mlp7_group'),(3,'complement')]:
  pair=v[arm,:,0][::2]-v[arm,:,0][1::2];atten=(base[cap]-pair[cap])/base[cap]
  records[name]={'target_rms':float(rms[arm,0]),'target_over_full':float(rms[arm,0]/rms[1,0]),'control_over_target':(rms[arm,1:]/rms[arm,0]).tolist(),'positive_fraction':float((atten>0).double().mean()),'mean_attenuation':float(atten.mean())}
 interaction=float((e[1,:,0]-e[2,:,0]-e[3,:,0]).square().mean().sqrt()/torch.minimum(rms[2,0],rms[3,0]))
 ratios=[]
 for k in range(16):
  a=4+2*k;b=a+1
  ratios.append(float((e[1,:,0]-e[a,:,0]-e[b,:,0]).square().mean().sqrt()/torch.minimum(rms[a,0],rms[b,0])))
 ordered=sorted(ratios);median=(ordered[7]+ordered[8])/2
 r={'pred_a':absolute<=1e-4 and relative<=1e-5 and sum_error<=1e-10 and outside==0 and bool(torch.isfinite(v).all()) and count==1440 and bool((rms[4:,0]>=.01*rms[1,0]).all()),
    'pred_b':interaction<=.35,'pred_c':interaction<=median,'arms':records,
    'interaction_over_smaller':interaction,'random_interactions':ratios,'random_median':median,'randoms_beaten':sum(interaction<x for x in ratios),'random_piece_target_rms':rms[4:,0].tolist(),
    'replay_max_abs':absolute,'replay_relative':relative,'sum_relative_error':sum_error,'max_outside':outside,'body_forwards':count,'seconds':time.perf_counter()-start,
    'scope':'Opened source-partition specificity against native-coordinate random splits. Not fresh confirmation or port closure; prior independent-branch failures preserved.','source_shas':binding}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
