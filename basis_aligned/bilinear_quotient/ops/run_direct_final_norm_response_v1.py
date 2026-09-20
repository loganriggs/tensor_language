#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_direct pred_c_norm
"""Final normalization response; FINAL_NORM_NATIVE_PLAN_V1.md.
pred_a_replay: native/formula effect<1e-4, archivedenergies<1e-5, sumidentity<1e-10.
pred_b_direct: direct-only centered relativeerror<.5 modes0/1 bothdomains/interventions.
pred_c_norm: norm-only centered relativeerror<.5 modes0/1 bothdomains/interventions.
Null: direct writing andnormalization interact; neither alone suffices.
Price48nativeforwards; frozen13916scalar+4608writercoeff10products, backgroundextra.
"""
import os,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(native_forwards=48,context=256,modes=5)));return
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from native_quartic_branch import pure_branch
 from final_norm_response import response
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'FINAL_NORM_NATIVE_V1.json';assert not out.exists();start=time.perf_counter();model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];_,ru=torch.linalg.qr(model.lm_head.weight.float());ru=ru.double();artifact=torch.load(P/'SELECTIVE_SCALAR_READOUT_V1.pt',weights_only=True);writer=artifact['residual_writer'].cuda().double();scale=float(artifact['teacher_scale']);view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);U=view['output_directions'].cuda().double();mu=view['output_mean'].cuda().double();donors=torch.load(P/'SELECTIVE_CONFIRMATION_DONORS_V1.pt',weights_only=True);records=[];replay=[];archived=[];identities=[];eps=torch.finfo(torch.float32).eps
 center=lambda a:a-a.mean(-1,keepdim=True)
 for domain in ['fineweb','code']:
  tokens=torch.load(P/f'SELECTIVE_CONFIRMATION_{domain.upper()}_V1.pt',weights_only=True);states=[];den=[];amplitudes=[]
  for row in tokens:
   c=capture(model,row[None,:256].cuda());pure=pure_branch(c['m16'],b16.mlp.Down_bias,b17.lambdas[0],b17.mlp.Left.weight,b17.mlp.Right.weight,b17.mlp.Down.weight).flatten(0,1).double();amplitudes.append((pure@ru.T/scale-mu)@U);states.append(c['final'].flatten(0,1));den.append(c['h17'].square().mean(-1,keepdim=True).flatten(0,1)+eps)
  stateall=torch.cat(states);denall=torch.cat(den);amp=torch.cat(amplitudes);targets=tokens[:,1:257].reshape(-1).cuda();oldrem=json.loads((P/f'CENTERED_EFFECT_REMOVAL_{domain.upper()}_V1.json').read_text())['records'];oldswap=json.loads((P/'CENTERED_EFFECT_SWAP_V1.json').read_text())['records'];lookup={(r['document'],str(r['mode'])):r for r in oldrem};swaplookup={(r['document'],str(r['mode'])):r for r in oldswap if r['domain']==domain and r['family']=='same_token'}
  for intervention in ['removal','same_token']:
   mapping=donors[domain]['same_token']
   for doc in range(len(tokens)):
    sites=torch.arange(doc*256,(doc+1)*256)
    if intervention=='same_token':sites=sites[mapping[sites]>=0]
    ids=sites.cuda();state=stateall[ids];pre=model.lm_head(F.rms_norm(state,(1152,)));base=30*torch.tanh(pre/30);basece=F.cross_entropy(base,targets[ids],reduction='none');old=lookup if intervention=='removal' else swaplookup
    for mode in [0,1,2,3,'joint']:
     sl=list(range(4)) if mode=='joint' else [mode];a=-amp[ids][:,sl] if intervention=='removal' else amp[mapping[sites].cuda()][:,sl]-amp[ids][:,sl];delta=(a@writer[:,sl].T).float()/denall[ids];changed=state+delta;native=30*torch.tanh(model.lm_head(F.rms_norm(changed,(1152,)))/30);nativeeffect=(native-base).double();effective=changed.double()-state.double();dl=model.lm_head(effective.float()).double();r=response(pre.double(),state.double(),effective,dl,eps);full=r['full']-r['baseline'];direct=r['direct_fixed_normalizer']-r['baseline'];norm=r['normalizer_only']-r['baseline'];interaction=full-direct-norm
     replay.append(float((full-nativeeffect).norm()/nativeeffect.norm()));en=float(nativeeffect.square().sum());archived.append(abs(en-old[(doc,str(mode))]['native_effect_energy'])/max(en,1e-30));identities.append(float((full-direct-norm-interaction).norm()/full.norm()));fc=center(full);row=dict(domain=domain,intervention=intervention,document=doc,mode=mode,sites=len(ids),native_effect_energy=en,full_centered_energy=float(fc.square().sum()),scale_ratio_mean=float(r['scale_ratio'].mean()),scale_ratio_deviation_rms=float((r['scale_ratio']-1).square().mean().sqrt()),components={})
     for name,effect in [('direct',direct),('normalizer',norm),('interaction',interaction)]:
      ec=center(effect);dot=float((ec*fc).sum());energy=float(ec.square().sum());row['components'][name]=dict(centered_energy=energy,full_centered_dot=dot,error_to_full_centered=float((ec-fc).square().sum()),raw_energy=float(effect.square().sum()))
     for name,z in [('full',r['full']),('direct',r['direct_fixed_normalizer']),('normalizer',r['normalizer_only'])]:row[name+'_ce_added']=float((F.cross_entropy(z.float(),targets[ids],reduction='none')-basece).mean())
     records.append(row)
  print(domain,'done',flush=True)
 summary={}
 for domain in ['fineweb','code']:
  summary[domain]={}
  for intervention in ['removal','same_token']:
   summary[domain][intervention]={}
   for mode in [0,1,2,3,'joint']:
    rr=[r for r in records if r['domain']==domain and r['intervention']==intervention and r['mode']==mode];full=sum(r['full_centered_energy'] for r in rr);entry=dict(components={})
    for name in ['direct','normalizer','interaction']:
     en=sum(r['components'][name]['centered_energy'] for r in rr);dot=sum(r['components'][name]['full_centered_dot'] for r in rr);error=sum(r['components'][name]['error_to_full_centered'] for r in rr);entry['components'][name]=dict(energy_over_full=en/full,signed_energy_fraction=dot/full,cosine=dot/(en*full)**.5,relative_error_to_full=(error/full)**.5)
    for name in ['full','direct','normalizer']:entry[name+'_ce_added']=sum(r[name+'_ce_added']*r['sites'] for r in rr)/sum(r['sites'] for r in rr)
    summary[domain][intervention][str(mode)]=entry
 major=[summary[d][t][str(g)] for d in summary for t in ['removal','same_token'] for g in [0,1]]
 pred=dict(pred_a_replay=max(replay)<1e-4 and max(archived)<1e-5 and max(identities)<1e-10,pred_b_direct=all(r['components']['direct']['relative_error_to_full']<.5 for r in major),pred_c_norm=all(r['components']['normalizer']['relative_error_to_full']<.5 for r in major))
 result=dict(predictions=pred,summary=summary,records=records,native_formula_replay=max(replay),archived_energy_replay=max(archived),sum_identity=max(identities),seconds=time.perf_counter()-start,scope='Diagnostic direct/normalizer controls with explicit interaction, not independent causal effects. Native amplitude only; no new candidate or fitting. Reused panels.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
