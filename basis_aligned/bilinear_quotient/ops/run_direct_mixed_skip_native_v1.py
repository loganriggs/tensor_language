#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact pred_b_gaussian pred_c_transfer
"""Frozen native DAG skip test, MIXED_SKIP_PLAN_V1.md.
pred_a_exact: previous solve<1e-8, all native replays<1e-5.
pred_b_gaussian: previous rank2 Gaussian error<=quadratic-only rank2.
pred_c_transfer: rank2 code mode1 effect error<.4, FineWeb branch CE<.025.
Null: exact Gaussian improvement does not transfer to native feature interventions.
Price ranks1/2/4/6:20802/21972/24312/26652coefficients,10products each.
All rank/law outcomes reported. No fitting or candidate selection on these panels.
"""
import os,json
import run_direct_native_quartic_branch_v1 as branch
import run_direct_native_mode_intervention_v1 as mode
RANKS=[1,2,4,6]
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(ranks=RANKS,domains=['fineweb','code'],native_forwards=160,primary=2)));return
 P=branch.P;out=P/'MIXED_SKIP_NATIVE_V1.json';assert not out.exists();fit=json.loads((P/'MIXED_SKIP_FIT_V1.json').read_text());records={};replays=[]
 for domain in ['fineweb','code']:
  branch.PANEL_PATH=None if domain=='fineweb' else P/'CODE_SHIFT_PANEL_V1.pt';branch.PROGRAM_SPECS={f'skip{r}':('MIXED_SKIP_PROGRAM_V1.pt',r) for r in RANKS};branch.PRIMARY='skip2';branch.OUTPUT_STEM='MIXED_SKIP_BRANCH_'+domain.upper()+'_V1';branch.PLAN=dict(branch.PLAN,documents=list(range(64,80)) if domain=='fineweb' else list(range(16)),arms=['exact','ablation']+[f'skip{r}' for r in RANKS],program_specs=branch.PROGRAM_SPECS)
  branch.main();b=json.loads((P/(branch.OUTPUT_STEM+'.json')).read_text());replays.extend([b['checks_max'],b['solve_error']]);records[domain]=dict(branch=b['summary'],modes={})
  for rank in RANKS:
   mode.PANEL_PATH=None if domain=='fineweb' else P/'CODE_SHIFT_PANEL_V1.pt';mode.PROGRAM_FILE='MIXED_SKIP_PROGRAM_V1.pt';mode.PROGRAM_KEY=rank;mode.OUTPUT_STEM=f'MIXED_SKIP_MODE_{domain.upper()}_R{rank}_V1';mode.PLAN=dict(mode.PLAN,documents=list(range(80,96)) if domain=='fineweb' else list(range(16)),program_file=mode.PROGRAM_FILE,program_key=rank)
   mode.main();r=json.loads((P/(mode.OUTPUT_STEM+'.json')).read_text());replays.append(r['replay_max']);records[domain]['modes'][rank]=r['summary']
 pred=dict(pred_a_exact=fit['predictions']['pred_a_exact'] and max(replays)<1e-5,pred_b_gaussian=fit['predictions']['pred_b_gaussian'],pred_c_transfer=records['code']['modes'][2]['1']['effect_relative_error']<.4 and records['fineweb']['branch']['skip2']['ce_added']<.025)
 result=dict(primary_rank=2,records=records,predictions=pred,replay_max=max(replays),scope='All frozen skip ranks on reused diagnostic panels; exact Gaussian weight fit, native branch and fixed canonical mode interventions. No new OOD confirmation or semantic claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,primary_fineweb_ce=records['fineweb']['branch']['skip2']['ce_added'],primary_code_mode1_error=records['code']['modes'][2]['1']['effect_relative_error']),indent=2))
if __name__=='__main__':main()
