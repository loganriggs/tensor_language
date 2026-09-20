#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_cross pred_c_cancel
"""Full last-MLP source census, LAST_MLP_SOURCE_PLAN_V1.md.
pred_a_replay: native sum/state/mm replay<1e-4, toy<1e-12.
pred_b_cross: max(rm,ma) centered removalnorm>.5mm bothdomains.
pred_c_cancel: m-dependent observable norm<.75 root-sum-square componentnorm bothdomains.
Null: selfterm is sufficient target or cross interactions do notcancel.
Price48nativeforwards; nativeweights6implicit blocks, no compactstudent orspeedupclaim.
"""
import os,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(native_forwards=48,context=256,source_blocks=6)));return
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from native_quartic_branch import pure_branch
 from last_mlp_source_blocks import terms,toy_check
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'LAST_MLP_SOURCE_BLOCKS_V1.json';assert not out.exists();start=time.perf_counter();toy=toy_check();model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];L,R,D=[v.weight.double() for v in [b17.mlp.Left,b17.mlp.Right,b17.mlp.Down]];_,ru=torch.linalg.qr(model.lm_head.weight.float());ru=ru.double();eps=torch.finfo(torch.float32).eps;checks=[];records=[];gram_records=[];names=['r_r','r_m','r_a','m_m','m_a','a_a']
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 logits=lambda state:30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
 for domain in ['fineweb','code']:
  tokens=torch.load(P/f'SELECTIVE_CONFIRMATION_{domain.upper()}_V1.pt',weights_only=True);totalgram=torch.zeros(6,6,dtype=torch.float64,device='cuda')
  for doc,row in enumerate(tokens):
   ids=row.cuda()[None];c=capture(model,ids[:,:256]);h=c['h17'].double();m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double();a=c['attn17'].double();r=h-m-a;square=h.square().mean(-1,keepdim=True)+eps;scale=square.sqrt();parts=terms(L,R,D,{k:v.flatten(0,1)/scale.flatten(0,1) for k,v in [('r',r),('m',m),('a',a)]});polynomial=sum(parts.values());nativepoly=(b17.mlp(F.rms_norm(c['h17'],(1152,)))-b17.mlp.Down_bias).double().flatten(0,1);checks.append(rel(polynomial,nativepoly));checks.append(rel(h.flatten(0,1)+polynomial+b17.mlp.Down_bias.double(),c['final'].double().flatten(0,1)));pure=pure_branch(c['m16'],b16.mlp.Down_bias,b17.lambdas[0],b17.mlp.Left.weight,b17.mlp.Right.weight,b17.mlp.Down.weight).double().flatten(0,1)/square.flatten(0,1);checks.append(rel(parts['m_m'],pure))
   observable=torch.stack([parts[n]@ru.T for n in names]).flatten(1);gram=observable@observable.T;totalgram+=gram;groups={**parts,'mlp_dependent':parts['m_m']+parts['r_m']+parts['m_a'],'attention_dependent':parts['a_a']+parts['r_a']+parts['m_a'],'all_polynomial':polynomial};state=c['final'].flatten(0,1);native=logits(state);basece=F.cross_entropy(native,ids[:,1:257].flatten(),reduction='none');logp=F.log_softmax(native,dim=-1);prob=logp.exp()
   for group,value in groups.items():
    z=logits(state-value.float());effect=(z-native).double();centered=effect-effect.mean(-1,keepdim=True);ce=F.cross_entropy(z,ids[:,1:257].flatten(),reduction='none')-basece;records.append(dict(domain=domain,document=doc,group=group,raw_effect_energy=float(effect.square().sum()),centered_effect_energy=float(centered.square().sum()),ce_added=float(ce.mean()),kl=float((prob*(logp-F.log_softmax(z,dim=-1))).sum(-1).mean())))
  indices=[names.index(n) for n in ['m_m','r_m','m_a']];sub=totalgram[indices][:,indices];ratio=float((sub.sum()/sub.diag().sum()).clamp_min(0).sqrt());gram_records.append(dict(domain=domain,names=names,gram=totalgram.cpu().tolist(),m_dependent_cancellation_ratio=ratio));print(domain,'done',flush=True)
 summary={}
 for domain in ['fineweb','code']:
  summary[domain]={}
  for group in [*names,'mlp_dependent','attention_dependent','all_polynomial']:
   rr=[r for r in records if r['domain']==domain and r['group']==group];summary[domain][group]={k:sum(r[k] for r in rr)/(len(rr) if k in ['ce_added','kl'] else 1) for k in ['raw_effect_energy','centered_effect_energy','ce_added','kl']}
  for row in summary[domain].values():row['centered_effect_norm_over_mm']=(row['centered_effect_energy']/summary[domain]['m_m']['centered_effect_energy'])**.5
 pred=dict(pred_a_replay=max(checks)<1e-4 and max(toy.values())<1e-12,pred_b_cross=all(max(summary[d][g]['centered_effect_norm_over_mm'] for g in ['r_m','m_a'])>.5 for d in summary),pred_c_cancel=all(r['m_dependent_cancellation_ratio']<.75 for r in gram_records))
 result=dict(predictions=pred,replay_max=max(checks),toy=toy,summary=summary,source_grams=gram_records,records=records,seconds=time.perf_counter()-start,scope='Exact native source-block numerator with shared frozen normalization. Component removals are not full upstream source interventions. Reused data, no student fit or semantic claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
