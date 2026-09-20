#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_calibration pred_c_transfer
"""Frozen metric-blend native test, SKIP_METRIC_BLEND_PLAN_V1.md.
pred_a_replay: calibration endpoint replay and all native replays<1e-5.
pred_b_calibration: exact fixed-rank endpoint objective ordering holds.
pred_c_transfer: alpha.5 code mode1 error<.4, FineWeb branch CE<.025.
Null: calibration higher moments do not repair frozen intervention transfer.
Price21948coefficients/10products for every arm; primaryalpha.5 fixed in advance.
All panels reused diagnostics, no rank or alpha selection during this test.
"""
import os,json
import run_direct_native_quartic_branch_v1 as branch
import run_direct_native_mode_intervention_v1 as mode
ALPHAS=[0.,.25,.5,.75,1.]
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(alphas=ALPHAS,domains=['fineweb','code'],native_forwards=192,primary=.5)));return
 P=branch.P;out=P/'SKIP_METRIC_BLEND_NATIVE_V1.json';assert not out.exists();fit=json.loads((P/'SKIP_METRIC_BLEND_FIT_V1.json').read_text());records={};replays=[]
 label=lambda a:f'blend{int(100*a)}'
 for domain in ['fineweb','code']:
  branch.PANEL_PATH=None if domain=='fineweb' else P/'CODE_SHIFT_PANEL_V1.pt';branch.PROGRAM_SPECS={label(a):('SKIP_METRIC_BLEND_PROGRAM_V1.pt',a) for a in ALPHAS};branch.PRIMARY='blend50';branch.OUTPUT_STEM='SKIP_METRIC_BLEND_BRANCH_'+domain.upper()+'_V1';branch.PLAN=dict(branch.PLAN,documents=list(range(64,80)) if domain=='fineweb' else list(range(16)),arms=['exact','ablation']+[label(a) for a in ALPHAS],program_specs=branch.PROGRAM_SPECS)
  branch.main();b=json.loads((P/(branch.OUTPUT_STEM+'.json')).read_text());replays.extend([b['checks_max'],b['solve_error']]);records[domain]=dict(branch=b['summary'],modes={})
  for alpha in ALPHAS:
   mode.PANEL_PATH=None if domain=='fineweb' else P/'CODE_SHIFT_PANEL_V1.pt';mode.PROGRAM_FILE='SKIP_METRIC_BLEND_PROGRAM_V1.pt';mode.PROGRAM_KEY=alpha;mode.OUTPUT_STEM=f'SKIP_METRIC_BLEND_MODE_{domain.upper()}_A{int(alpha*100)}_V1';mode.PLAN=dict(mode.PLAN,documents=list(range(80,96)) if domain=='fineweb' else list(range(16)),program_file=mode.PROGRAM_FILE,program_key=alpha)
   mode.main();r=json.loads((P/(mode.OUTPUT_STEM+'.json')).read_text());replays.append(r['replay_max']);records[domain]['modes'][label(alpha)]=r['summary']
 pred=dict(pred_a_replay=fit['predictions']['pred_a_replay'] and max(replays)<1e-5,pred_b_calibration=fit['predictions']['pred_b_calibration'],pred_c_transfer=records['code']['modes']['blend50']['1']['effect_relative_error']<.4 and records['fineweb']['branch']['blend50']['ce_added']<.025)
 result=dict(primary_alpha=.5,records=records,predictions=pred,replay_max=max(replays),scope='Fixed-cost Gaussian/empirical calibration moment blend. All alpha outcomes on reused native diagnostic panels. Explicitly data-informed fit, no new OOD confirmation or semantic claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,primary_fineweb_ce=records['fineweb']['branch']['blend50']['ce_added'],primary_code_mode1_error=records['code']['modes']['blend50']['1']['effect_relative_error']),indent=2))
if __name__=='__main__':main()
