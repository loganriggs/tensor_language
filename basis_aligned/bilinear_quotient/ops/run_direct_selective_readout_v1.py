#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_swap pred_c_joint
"""Selective scalar decoder test, SELECTIVE_READOUT_PLAN_V1.md.
pred_a_instrument: native and unchanged-feature effects replay<1e-5.
pred_b_swap: mode1 same-token error<.4/cosine>.9 in both domains.
pred_c_joint: joint removal and same-token swap errors<=baseline both domains.
Null: selective readout gains fail interchange or joint nonlinear behavior.
Price:96forwards;13916scalar+4608writercoefficients/10products.
"""
import os,json,hashlib
import run_direct_native_mode_intervention_v1 as mode
import run_direct_feature_swap_v1 as swap

def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(native_forwards=96,context=256,products=10)));return
 import torch
 from disk_guard import guard_torch_save
 P=mode.P;out=P/'SELECTIVE_READOUT_NATIVE_V1.json';assert not out.exists()
 scalar=torch.load(P/'SELECTIVE_SCALAR_READOUT_V1.pt',weights_only=True);s={k:v.double() for k,v in scalar['program'].items()};view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);U=view['output_directions'].double();mu=view['output_mean'].double();a={k:s[k] for k in ['A','B','root_left','root_right']};a.update(output_writer=U@s['quartic_readout'],skip_writer=U@s['quadratic_readout'],constant=mu+U@s['constant'])
 guard_torch_save(dict(programs={'selective':a}),str(P/'SELECTIVE_READOUT_NATIVE_ADAPTER_V1.pt'))
 removal={};checks=[]
 def compare(records,oldrecords,fields):
  assert len(records)==len(oldrecords)
  for r,o in zip(records,oldrecords):
   assert (r['document'],r['mode'])==(o['document'],o['mode'])
   checks.append(abs(r['native_effect_energy']-o['native_effect_energy'])/max(abs(o['native_effect_energy']),1e-30))
   if r['mode'] in [0,2,3]:
    for field in fields:checks.append(abs(r[field]-o[field])/max(abs(o[field]),1e-30))
 for domain,n in [('FINEWEB',32),('CODE',16)]:
  mode.PANEL_PATH=P/f'BLEND_CONFIRMATION_{domain}_V1.pt';mode.PROGRAM_FILE='SELECTIVE_READOUT_NATIVE_ADAPTER_V1.pt';mode.PROGRAM_KEY='selective';mode.EXTRACTED_FILE=P/'SELECTIVE_SCALAR_READOUT_V1.pt';mode.OUTPUT_STEM=f'SELECTIVE_READOUT_REMOVAL_{domain}_V1';mode.PLAN=dict(mode.PLAN,documents=list(range(n)),context=256,native_forwards=n,program_file=mode.PROGRAM_FILE,program_key='selective');mode.main()
  r=json.loads((P/(mode.OUTPUT_STEM+'.json')).read_text());old=json.loads((P/f'BLEND_CONFIRMATION_MODE_{domain}_PRIMARY_V1.json').read_text());assert r['token_hash']==old['token_hash'];checks.append(r['replay_max']);compare(r['records'],old['records'],['predicted_effect_energy','effect_error_energy']);removal[domain.lower()]=dict(current=r['summary'],baseline=old['summary'])
 swap.PROGRAM_FILE='SELECTIVE_SCALAR_READOUT_V1.pt';swap.OUTPUT_FILE='SELECTIVE_READOUT_SWAP_V1.json';swap.main();r=json.loads((P/swap.OUTPUT_FILE).read_text());old=json.loads((P/'FEATURE_SWAP_V1.json').read_text());assert r['predictions']['pred_a_instrument']
 for a,b in zip(r['records'],old['records']):assert (a['domain'],a['family'])==(b['domain'],b['family'])
 compare(r['records'],old['records'],['predicted_effect_energy','effect_error_energy'])
 pred=dict(pred_a_instrument=max(checks)<1e-5,pred_b_swap=all(r['summary'][d]['same_token']['1']['effect_relative_error']<.4 and r['summary'][d]['same_token']['1']['effect_cosine']>.9 for d in removal),pred_c_joint=all(removal[d]['current']['joint']['effect_relative_error']<=removal[d]['baseline']['joint']['effect_relative_error'] and r['summary'][d]['same_token']['joint']['effect_relative_error']<=old['summary'][d]['same_token']['joint']['effect_relative_error'] for d in removal))
 result=dict(predictions=pred,replay_max=max(checks),removal=removal,swap=r['summary'],swap_baseline=old['summary'],artifact_sha256=hashlib.sha256((P/swap.PROGRAM_FILE).read_bytes()).hexdigest(),scope='Diagnostic-selected feature1 readout, reused data; native background explicit. Not fresh confirmation or semantic/task selectivity.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
