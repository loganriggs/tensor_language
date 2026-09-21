#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_exact pred_c_price
"""Exact original-weight source-pair program:32 reused balanced-donor captures.
pred_a native final/source replay. pred_b hybrid/change leading-feature removal
error<1e-4 and abs CE disagreement<1e-6 every cohort/domain. pred_c sourceproducts
1152<=4608/4 and storedfloats<10628354. Native z/h interfaces explicit.
Null: large block rewrite loses native effects numerically. No fit/newbehaviorclaim.
"""
import os,json
import run_direct_source_interchange_v1 as run
run.PROGRAM_FILE='MIDPOINT_ORIGINAL_SOURCE_REPLAY_PROGRAMS_V1.pt'
run.PLAN_FILE='MIDPOINT_ORIGINAL_SOURCE_REPLAY_PLAN_V1.json'
run.DONOR_FILE='MIDPOINT_BALANCED_SOURCE_DONORS_V1.pt'
run.TOKEN_FILE='MIDPOINT_SHARED_SOURCE_TOKENS_V1.pt'
run.OUTPUT_FILE='MIDPOINT_ORIGINAL_SOURCE_REPLAY_V1.json'
run.PRIMARY='original';run.ANCHOR='original';run.FORWARDS=32
if __name__=='__main__':
 run.main()
 if not (os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL')):
  path=run.P/run.OUTPUT_FILE;result=json.loads(path.read_text());s=result['summary'];price=json.loads((run.P/'MIDPOINT_ORIGINAL_SOURCE_BLOCK_V1.json').read_text())
  result['predictions']=dict(pred_a_instrument=result['predictions']['pred_a_instrument'],pred_b_exact=all(s[d]['original'][f][c]['effect_relative_error']<1e-4 and abs(s[d]['original'][f][c]['ce_disagreement'])<1e-6 for d in s for f in ['hybrid','change'] for c in ['all','continuation','spaced_word']),pred_c_price=price['source_products']<=4608/4 and price['stored_float_scalars']<10628354);result['scope']=result['plan']['scope'];path.write_text(json.dumps(result,indent=2)+'\n');print('FINAL original-source checks',result['predictions'],flush=True)
