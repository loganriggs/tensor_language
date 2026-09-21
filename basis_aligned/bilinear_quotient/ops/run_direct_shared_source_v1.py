#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_hybrid pred_c_change
"""Frozen shared quadratic dictionary:32 captures,16 freshFW+16 reusedstdlib.
Native source-interchange executor; pred_a exact replay. pred_b shared16 hybrid
error<=.15 AND <=1.10 unsharedcovariance16 every cohort/domain; pred_c shared16
change error<=.20 AND <=1.10 baseline every cohort/domain. Null: shared inputs
lose selective effect fidelity. Shared16 23588scalars versus41508 baseline;
32source squares and one finalproduct each, plus512small inner coefficients.
Native z/h inputs remain; no independent extraction or broad-OOD claim.
"""
import os,json
import run_direct_source_interchange_v1 as run
run.PROGRAM_FILE='MIDPOINT_SHARED_SOURCE_PROGRAMS_V1.pt'
run.PLAN_FILE='MIDPOINT_SHARED_SOURCE_PLAN_V1.json'
run.DONOR_FILE='MIDPOINT_SHARED_SOURCE_DONORS_V1.pt'
run.TOKEN_FILE='MIDPOINT_SHARED_SOURCE_TOKENS_V1.pt'
run.OUTPUT_FILE='MIDPOINT_SHARED_SOURCE_NATIVE_V1.json'
run.PRIMARY='shared16';run.ANCHOR='covariance_16';run.FORWARDS=32
if __name__=='__main__':
 run.main()
 if not (os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL')):
  path=run.P/run.OUTPUT_FILE;result=json.loads(path.read_text());s=result['summary']
  pred_a_instrument=result['predictions']['pred_a_instrument']
  pred_b_hybrid=all(s[d]['shared16']['hybrid'][c]['effect_relative_error']<=min(.15,1.10*s[d]['covariance_16']['hybrid'][c]['effect_relative_error']) for d in s for c in ['all','continuation','spaced_word'])
  pred_c_change=all(s[d]['shared16']['change'][c]['effect_relative_error']<=min(.20,1.10*s[d]['covariance_16']['change'][c]['effect_relative_error']) for d in s for c in ['all','continuation','spaced_word'])
  result['predictions']=dict(pred_a_instrument=pred_a_instrument,pred_b_hybrid=pred_b_hybrid,pred_c_change=pred_c_change);result['scope']=result['plan']['scope'];path.write_text(json.dumps(result,indent=2)+'\n');print('FINAL registered comparison',result['predictions'],flush=True)
