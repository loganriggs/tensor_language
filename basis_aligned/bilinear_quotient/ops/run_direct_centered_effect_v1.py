#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_noncommon pred_c_centered
"""Common/centered native effects; CENTERED_EFFECT_PLAN_V1.md.
pred_a_replay: original raw energies replay<1e-5, partitionidentity<1e-10.
pred_b_noncommon: mode1nativecommonfraction<.5 bothinterventions/domains.
pred_c_centered: mode1centerederror<.4/cos>.9 bothinterventions/domains.
Null: raw-logit fidelity overstates probability-relevant intervention fidelity.
Price96forwards;18524coeff10products includingwriters, nativebackground extra.
"""
import os,json
import run_direct_native_mode_intervention_v1 as mode
import run_direct_feature_swap_v1 as swap

def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(native_forwards=96,products=10)));return
 P=mode.P;out=P/'CENTERED_EFFECT_V1.json';assert not out.exists();checks=[];partitionchecks=[];results={}
 def audit(records,previous):
  assert len(records)==len(previous)
  for r,o in zip(records,previous):
   assert (r['document'],r['mode'])==(o['document'],o['mode'])
   for field in ['native_effect_energy','predicted_effect_energy','effect_error_energy']:checks.append(abs(r[field]-o[field])/max(abs(o[field]),1e-30))
   for prefix in ['native','predicted']:partitionchecks.append(abs(r[prefix+'_centered_effect_energy']+r[prefix+'_common_effect_energy']-r[prefix+'_effect_energy'])/r[prefix+'_effect_energy'])
   partitionchecks.append(abs(r['centered_effect_error_energy']+r['common_effect_error_energy']-r['effect_error_energy'])/max(r['effect_error_energy'],1e-30))
 for domain,n in [('fineweb',32),('code',16)]:
  mode.PANEL_PATH=P/f'SELECTIVE_CONFIRMATION_{domain.upper()}_V1.pt';mode.PROGRAM_FILE='SELECTIVE_READOUT_NATIVE_ADAPTER_V1.pt';mode.PROGRAM_KEY='selective';mode.EXTRACTED_FILE=P/'SELECTIVE_SCALAR_READOUT_V1.pt';mode.OUTPUT_STEM=f'CENTERED_EFFECT_REMOVAL_{domain.upper()}_V1';mode.PLAN=dict(mode.PLAN,documents=list(range(n)),context=256,native_forwards=n,program_file=mode.PROGRAM_FILE,program_key='selective');mode.main();r=json.loads((P/(mode.OUTPUT_STEM+'.json')).read_text());old=json.loads((P/f'SELECTIVE_CONFIRMATION_REMOVAL_SELECTIVE_{domain.upper()}_V1.json').read_text());assert r['token_hash']==old['token_hash'];checks.append(r['replay_max']);audit(r['records'],old['records']);results[domain]=dict(removal=r['summary'])
 swap.PROGRAM_FILE='SELECTIVE_SCALAR_READOUT_V1.pt';swap.OUTPUT_FILE='CENTERED_EFFECT_SWAP_V1.json';swap.PANEL_PREFIX='SELECTIVE_CONFIRMATION';swap.DONOR_PREFIX='SELECTIVE_CONFIRMATION_DONORS';swap.main();r=json.loads((P/swap.OUTPUT_FILE).read_text());old=json.loads((P/'SELECTIVE_CONFIRMATION_SWAP_SELECTIVE_V1.json').read_text());audit(r['records'],old['records'])
 for domain in results:results[domain]['same_token']=r['summary'][domain]['same_token']
 major=[results[d][t]['1'] for d in results for t in ['removal','same_token']]
 pred=dict(pred_a_replay=max(checks)<1e-5 and max(partitionchecks)<1e-10,pred_b_noncommon=all(r['native_common_fraction']<.5 for r in major),pred_c_centered=all(r['centered_effect_relative_error']<.4 and r['centered_effect_cosine']>.9 for r in major))
 result=dict(results=results,predictions=pred,raw_replay_max=max(checks),partition_replay_max=max(partitionchecks),scope='Post-softcap centered effects are probability-relevant; raw effects retained. Reused panels, no semantic selectivity or new fitting.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
