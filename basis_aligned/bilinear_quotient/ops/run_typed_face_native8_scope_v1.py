#!/usr/bin/env python3
# BQGATE:760bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a finite/closure<=1e-5; pred_b>=18/24capable each variant.
pred_c conditional/native8 effect error<=.35; pred_d>=.75 positive attenuation and mean>=.02.
pred_e>=2xmedian null and beats15/16; pred_f four collateral ratios<=.5.
760 forwards. Fixed full product and nulls, no source reselection.
"""
import hashlib,importlib.util,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import measure
from run_even_value_factorial_native_v1 import setup
import typed_face_write_atoms_v1 as atoms
native=atoms.native
STEM='TYPED_FACE_NATIVE8_SCOPE_V1';ARMS=['native','conditional','native8_midpoint']+[f'native8_null:{i}' for i in range(16)]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 doc=json.loads((P/'TYPED_FACE_PROSPECTIVE_V1_ROWS.json').read_text());rows=doc['rows'];groups,mapping=group_rows(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  assert len(groups)==40 and len(ARMS)==19;print('760bodyforwards;40prefixes;five opened construction families;16matched nulls');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');p={k:v.to('cuda') for k,v in torch.load(P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt',weights_only=True).items()}
 norm_errors=[];closure=[]
 def write(arm,row,donor_row,current,donor,mask):
  city=row['city_position'];rt=row['ids'][city];dt=donor_row['ids'][city]
  candidate=native.execute(p,current,donor[:,city],rt,dt,city,mask)
  candidate=.5*candidate
  if arm in ('conditional','native8_midpoint'):return candidate
  seed=17092300+1000*int(arm.split(':')[1])+row['context_id'];g=torch.Generator(device=current.device).manual_seed(seed)
  channels=torch.randn((*candidate.shape[:2],128),generator=g,device=current.device,dtype=current.dtype)
  direction=F.linear(channels,p['output']);norm=candidate.norm(dim=-1,keepdim=True)
  direction=direction*(norm/direction.norm(dim=-1,keepdim=True).clamp_min(1e-30))*(1 if row['cue']=='British' else -1)
  norm_errors.append(float(((direction.norm(dim=-1,keepdim=True)-norm).abs()/norm.clamp_min(1e-8)).max()))
  return direction
 m=measure(model,graph,groups,ARMS,write);v=expand(m['values'],mapping,6);effects=v-v[:1];records={}
 for variant in doc['variants']:
  idx=[i for i,r in enumerate(rows) if r['variant']==variant];values=v[:,idx];e=effects[:,idx]
  native_pair=values[0,::2,0]-values[0,1::2,0];mid_pair=values[2,::2,0]-values[2,1::2,0];cap=native_pair>=.1
  attenuation=(native_pair[cap]-mid_pair[cap])/native_pair[cap]
  rms=e.square().mean(1).sqrt();target=float(rms[2,0]);nr=rms[3:,0];median=float(nr.median())
  # Even-sample median is mean of central order statistics.
  sorted_nr=nr.sort().values;median=float((sorted_nr[7]+sorted_nr[8])/2)
  records[variant]={'capable_pairs':int(cap.sum()),'native_pairs':24,'parent_rms_logits':float(rms[1,0]),'prediction_error':float((e[2,:,0]-e[1,:,0]).norm()/e[2,:,0].norm().clamp_min(1e-8)),
   'attenuation_positive_fraction':float((attenuation>0).double().mean()) if len(attenuation) else 0.,'attenuation_mean':float(attenuation.mean()) if len(attenuation) else 0.,
   'target_rms_logits':target,'target_over_null_median':target/max(median,1e-30),'nulls_beaten':int((nr<target).sum()),'null_readout_rms_logits':rms[3:].tolist(),'control_over_target_rms':(rms[2,1:]/max(target,1e-30)).tolist()}
 anchor=torch.load(P/'TYPED_FACE_PROSPECTIVE_V1_ARTIFACT.pt',weights_only=True)['values'][[0,3]]
 anchor_error=float((v[:2]-anchor).abs().max())
 result={'pred_a':bool(torch.isfinite(v).all()) and max(m['outside'])==0 and max(norm_errors)<=1e-5 and anchor_error<=1e-5 and m['body_forwards']==760,
 'pred_b':all(r['capable_pairs']>=18 and r['parent_rms_logits']>=1e-5 and r['target_rms_logits']>=1e-5 for r in records.values()),
 'pred_c':max(r['prediction_error'] for r in records.values())<=.35,
 'pred_d':all(r['attenuation_positive_fraction']>=.75 and r['attenuation_mean']>=.02 for r in records.values()),
 'pred_e':all(r['target_over_null_median']>=2 and r['nulls_beaten']>=15 for r in records.values()),
 'pred_f':max(max(r['control_over_target_rms']) for r in records.values())<=.5,
 'variants':records,'max_null_norm_error':max(norm_errors),'anchor_max_abs':anchor_error,'body_forwards':m['body_forwards'],'seconds':time.perf_counter()-start,'source_shas':binding,
 'scope':'Opened prospective rows. Same frozen half-write applied before MLP8 with all native consumers recomputed. Conditional odd-only route is a distinct counterfactual. No fresh-row confirmation or composition promotion.'}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
