#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_major pred_c_joint
"""Independent confirmation; SELECTIVE_CONFIRMATION_PLAN_V1.md.
pred_a_instrument: target/untouched effects replay<1e-5 and hashes match.
pred_b_major: selective mode1 removal/same-token error<.4/cos>.9 bothdomains.
pred_c_joint: joint error no worse than original for both interventions/domains.
pred_d_gain: code mode1 MSEgain>=10% for bothinterventions.
Null: diagnostic-selected scalar readout fails fresh transfer.
Price192forwards;18524coeff10products incl residualwriters, nativebackground extra.
"""
import os,json,hashlib
import run_direct_native_mode_intervention_v1 as mode
import run_direct_feature_swap_v1 as swap

SPECS=None
OUTPUT_PREFIX='SELECTIVE_CONFIRMATION'
RESULT_SCOPE='Independent document/file confirmation of frozen diagnostic-selected scalar readout. Related local code; pretrained overlap unknown. Not semantic selectivity or full model replacement.'

def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(native_forwards=192,context=256,products=10)));return
 import torch
 P=mode.P;out=P/f'{OUTPUT_PREFIX}_V1.json';assert not out.exists();meta=json.loads((P/'SELECTIVE_CONFIRMATION_PANELS_V1.json').read_text());assert hashlib.sha256((P/'SELECTIVE_SCALAR_READOUT_V1.pt').read_bytes()).hexdigest()==meta['program_sha256'];checks=[];removal={};swaps={}
 specs=SPECS if SPECS is not None else {'original':('SKIP_METRIC_BLEND_PROGRAM_V1.pt',.5,'EXTRACTED_SCALAR_INTERVENTIONS_V1.pt'),'selective':('SELECTIVE_READOUT_NATIVE_ADAPTER_V1.pt','selective','SELECTIVE_SCALAR_READOUT_V1.pt')}
 assert len(specs)==2
 baseline,primary=list(specs)
 for candidate,(file,key,artifact) in specs.items():
  removal[candidate]={}
  for domain,n in [('fineweb',32),('code',16)]:
   panel=P/f'SELECTIVE_CONFIRMATION_{domain.upper()}_V1.pt';tokens=torch.load(panel,weights_only=True);assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==meta['panels'][domain]['token_sha256']
   mode.PANEL_PATH=panel;mode.PROGRAM_FILE=file;mode.PROGRAM_KEY=key;mode.EXTRACTED_FILE=P/artifact;mode.OUTPUT_STEM=f'{OUTPUT_PREFIX}_REMOVAL_{candidate.upper()}_{domain.upper()}_V1';mode.PLAN=dict(mode.PLAN,documents=list(range(n)),context=256,native_forwards=n,program_file=file,program_key=key);mode.main();r=json.loads((P/(mode.OUTPUT_STEM+'.json')).read_text());checks.append(r['replay_max']);removal[candidate][domain]=r
  swap.PROGRAM_FILE=artifact;swap.OUTPUT_FILE=f'{OUTPUT_PREFIX}_SWAP_{candidate.upper()}_V1.json';swap.PANEL_PREFIX='SELECTIVE_CONFIRMATION';swap.DONOR_PREFIX='SELECTIVE_CONFIRMATION_DONORS';swap.main();swaps[candidate]=json.loads((P/swap.OUTPUT_FILE).read_text());assert swaps[candidate]['predictions']['pred_a_instrument']
 def compare(a,b):
  assert len(a)==len(b)
  for r,o in zip(a,b):
   assert (r['document'],r['mode'])==(o['document'],o['mode'])
   checks.append(abs(r['native_effect_energy']-o['native_effect_energy'])/max(abs(o['native_effect_energy']),1e-30))
   if r['mode'] in [0,2,3]:
    for field in ['predicted_effect_energy','effect_error_energy']:checks.append(abs(r[field]-o[field])/max(abs(o[field]),1e-30))
 for d in ['fineweb','code']:compare(removal[primary][d]['records'],removal[baseline][d]['records'])
 compare(swaps[primary]['records'],swaps[baseline]['records'])
 results={}
 for domain in ['fineweb','code']:
  results[domain]={candidate:dict(removal=removal[candidate][domain]['summary'],same_token=swaps[candidate]['summary'][domain]['same_token']) for candidate in specs}
 major=[results[d][primary][t]['1'] for d in results for t in ['removal','same_token']]
 pred=dict(pred_a_instrument=max(checks)<1e-5,pred_b_major=all(r['effect_relative_error']<.4 and r['effect_cosine']>.9 for r in major),pred_c_joint=all(results[d][primary][t]['joint']['effect_relative_error']<=results[d][baseline][t]['joint']['effect_relative_error'] for d in results for t in ['removal','same_token']),pred_d_gain=all(results['code'][primary][t]['1']['effect_relative_error']**2<=.9*results['code'][baseline][t]['1']['effect_relative_error']**2 for t in ['removal','same_token']))
 result=dict(predictions=pred,results=results,replay_max=max(checks),manifest=meta,scope=RESULT_SCOPE)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
