#!/usr/bin/env python3
# BQGATE:800bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a replay abs<=1e-4/rel<=1e-5; pred_b effecterror<=.35/livecap>=90;
pred_c controls<=.5target; pred_d exactnorm replay/effects<=1e-3;
pred_e target>=2nullmedian/16beaten; pred_f positive>=.90/mean>=.02.800forwards.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_MLP7_GENERATED_NORM_EDIT_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_FULL_PILE_V2_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_NO_MODEL') or os.environ.get('BQLIB_DRYRUN'):
  print('800bodyforwards;40prefixes;generated city norms and16nulls');return
 cert=json.loads((P/'CITY_MLP7_READERS_V1_RESULT.json').read_text());assert all(cert[k] for k in ['pred_a','pred_b','pred_c'])
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval()
 norm_fixtures=torch.load(P/'CITY_MLP7_GENERATED_NORM_V1_CPU_ARTIFACT.pt',weights_only=True)['fixtures'];refs=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 writes=[torch.stack([r['inputs']['delta'].double(),f['exact'],f['approximate']]).cuda() for f,r in zip(norm_fixtures,refs,strict=True)]
 outside=max(float(w[:,:, [i for i in range(len(r['ids'])) if i not in r['destination_positions']]].abs().max()) for w,r in zip(writes,groups))
 state={};norm_errors=[]
 def post8(module,args,out):
  arm=state['arm'];i=state['i']
  if arm==0:return out
  if arm<=3:d=writes[i][arm-1].to(out[0].dtype)
  else:
   row=groups[i];gen=torch.Generator(device=out[0].device).manual_seed(18091300+1000*(arm-4)+row['context_id'])
   random=torch.randn(out[0].shape,device=out[0].device,dtype=out[0].dtype,generator=gen).double()
   norm=writes[i][2].norm(dim=-1,keepdim=True)
   d=(random*norm/random.norm(dim=-1,keepdim=True).clamp_min(1e-30)*(1 if row['cue']=='British' else -1)).to(out[0].dtype)
   norm_errors.append(float(((d.double().norm(dim=-1,keepdim=True)-norm).abs()/norm.clamp_min(1e-8)).max()))
  return out[0]+d,out[1]
 handle=model.transformer.h[8].attn.register_forward_hook(post8)
 values=torch.zeros(20,40,10,dtype=torch.float64);count=0
 try:
  for arm in range(20):
   state['arm']=arm
   for i,row in enumerate(groups):
    state['i']=i;ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
    for block in model.transformer.h:x,first=block(x,first,x0)
    scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=(scores[left]-scores[right]).cpu()
    count+=1
 finally:handle.remove()
 v=expand(values,mapping,6);e=v-v[:1];old=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['values']
 diff=v[:2]-old[:2];absolute=float(diff.abs().max());relative=float(diff.norm()/old[:2].norm())
 rms=e.square().mean(1).sqrt();base=v[0,:,0][::2]-v[0,:,0][1::2];cap=base>=.1;records={}
 for arm,name in [(1,'full'),(2,'exact_norm'),(3,'approximate_norm')]:
  pair=v[arm,:,0][::2]-v[arm,:,0][1::2];atten=(base[cap]-pair[cap])/base[cap]
  records[name]={'target_rms':float(rms[arm,0]),'target_over_full':float(rms[arm,0]/rms[1,0]),'control_over_target':(rms[arm,1:]/rms[arm,0]).tolist(),'positive_fraction':float((atten>0).double().mean()),'mean_attenuation':float(atten.mean())}
 actual=e[1];approx=e[3];error=float((approx[:,0]-actual[:,0]).norm()/actual[:,0].norm())
 exact_diff=v[2]-v[1];exact_abs=float(exact_diff.abs().max());exact_rel=float(exact_diff.norm()/v[1].norm())
 exact_effect=((e[2]-e[1]).square().sum(0).sqrt()/e[1].square().sum(0).sqrt()).tolist()
 nr=rms[4:,0];ordered=nr.sort().values;median=float((ordered[7]+ordered[8])/2);g=records['approximate_norm']
 strata={}
 for natural in [True,False]:
  idx=[i for i,row in enumerate(rows) if row['is_untouched_natural_arm']==natural]
  strata['untouched' if natural else 'substituted']=float((approx[idx,0]-actual[idx,0]).norm()/actual[idx,0].norm())
 r={'pred_a':absolute<=1e-4 and relative<=1e-5 and outside==0 and max(norm_errors)<=1e-5 and bool(torch.isfinite(v).all()) and count==800,
    'pred_b':error<=.35 and g['target_rms']>=1e-5 and int(cap.sum())>=90,
    'pred_c':max(g['control_over_target'])<=.5,
    'pred_d':exact_abs<=1e-4 and exact_rel<=1e-5 and max(exact_effect)<=1e-3,
    'pred_e':g['target_rms']>=2*median and bool((nr<g['target_rms']).all()),
    'pred_f':g['positive_fraction']>=.90 and g['mean_attenuation']>=.02,
    'arms':records,'target_error':error,'prediction_error_strata':strata,'capable_pairs':int(cap.sum()),
    'exact_norm_max_abs':exact_abs,'exact_norm_relative':exact_rel,'exact_norm_effect_errors':exact_effect,
    'target_over_null_median':g['target_rms']/median,'nulls_beaten':int((nr<g['target_rms']).sum()),'null_readout_rms':rms[4:].tolist(),
    'replay_max_abs':absolute,'replay_relative':relative,'max_outside':outside,'max_null_norm_error':max(norm_errors),'body_forwards':count,'seconds':time.perf_counter()-start,
    'scope':'Opened generated-city-normalizer screen, native attention8 intervention and full native MLP8/suffix. Remaining upstream/query states external. No fresh or independent composition claim.','source_shas':binding}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
