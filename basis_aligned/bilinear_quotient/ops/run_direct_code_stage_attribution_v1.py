#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_original pred_c_intermediate
"""Archived stage comparison; CODE_STAGE_ATTRIBUTION_PLAN_V1.md.
pred_a_replay: native target energies/hash and replay agree <1e-5.
pred_b_original: 26-product mode1 error<.4/cosine>.9.
pred_c_intermediate:12-product mode1 error<.4/cosine>.9.
Null: errors enter before final compression or shared assumptions fail.
Price:26products47312coefficients,12products21960; original background retained.
"""
import os,json
import run_direct_native_mode_intervention_v1 as run
ARMS=[('quartic26','NATIVE_QUARTIC_MEAN_V1.pt',8),('quartic12','FUSED_ROOT_PROGRAM_V1.pt',6)]
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(arms=ARMS,native_forwards=32,panel='CODE_SHIFT_PANEL_V1.pt')));return
 out=run.P/'CODE_STAGE_ATTRIBUTION_V1.json';assert not out.exists()
 baseline=json.loads((run.P/'CODE_SHIFT_MODES_V1.json').read_text());results={};checks=[]
 for name,file,key in ARMS:
  run.PANEL_PATH=run.P/'CODE_SHIFT_PANEL_V1.pt';run.PROGRAM_FILE=file;run.PROGRAM_KEY=key;run.OUTPUT_STEM='CODE_STAGE_'+name.upper()+'_V1';run.PLAN=dict(run.PLAN,documents=list(range(16)),panel='CODE_SHIFT_PANEL_V1.pt',program_file=file,program_key=key)
  run.main();r=json.loads((run.P/(run.OUTPUT_STEM+'.json')).read_text());checks.append(r['replay_max']);assert r['token_hash']==baseline['token_hash']
  for a,b in zip(r['records'],baseline['records']):
   assert (a['document'],a['mode'])==(b['document'],b['mode']);checks.append(abs(a['native_effect_energy']-b['native_effect_energy'])/max(b['native_effect_energy'],1e-30))
  results[name]=r['summary']
 passed=lambda name:results[name]['1']['effect_relative_error']<.4 and results[name]['1']['effect_cosine']>.9
 pred=dict(pred_a_replay=max(checks)<1e-5,pred_b_original=passed('quartic26'),pred_c_intermediate=passed('quartic12'))
 result=dict(arms=ARMS,summary=results,predictions=pred,native_replay_max=max(checks),scope='Archived-stage attribution on reused code panel, fixed common feature basis, no refitting.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
