#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_replay pred_c_simpler
"""Exact shared-product compiler replay on32 reused balanced-donor captures.
pred_a original native/source replay bars. pred_b compiled/original summary
logit error and CE difference<1e-6 both ranks/allcohorts/domains. pred_c variable
source products16/24<32 and smaller float count than23588/33060. Null: algebraic
rewrite fails numerically or saves no computation. Existing behavioral failures
must persist; this is compilation evidence, not adoption. Inputs nativez/h.
"""
import os,json
import run_direct_source_interchange_v1 as run
run.PROGRAM_FILE='MIDPOINT_BLOCK_SOURCE_REPLAY_PROGRAMS_V1.pt'
run.PLAN_FILE='MIDPOINT_BLOCK_SOURCE_REPLAY_PLAN_V1.json'
run.DONOR_FILE='MIDPOINT_BALANCED_SOURCE_DONORS_V1.pt'
run.TOKEN_FILE='MIDPOINT_SHARED_SOURCE_TOKENS_V1.pt'
run.OUTPUT_FILE='MIDPOINT_BLOCK_SOURCE_REPLAY_V1.json'
run.PRIMARY='block16';run.ANCHOR='covariance_16';run.FORWARDS=32
if __name__=='__main__':
 run.main()
 if not (os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL')):
  path=run.P/run.OUTPUT_FILE;result=json.loads(path.read_text());s=result['summary'];diff=max(abs(s[d]['block'+k][f][c][m]-s[d]['shared'+k][f][c][m]) for d in s for k in ['16','24'] for f in ['hybrid','change'] for c in ['all','continuation','spaced_word'] for m in ['effect_relative_error','ce_added'])
  prices=json.loads((run.P/'MIDPOINT_SOURCE_BLOCK_COMPILER_V1.json').read_text())['records']
  result['predictions']=dict(pred_a_instrument=result['predictions']['pred_a_instrument'],pred_b_replay=diff<1e-6,pred_c_simpler=all(r['source_products']<32 and r['stored_float_scalars']<({16:23588,24:33060}[r['shared_rank']]) for r in prices));result['compiled_summary_replay']=diff;result['scope']=result['plan']['scope'];path.write_text(json.dumps(result,indent=2)+'\n');print('FINAL registered compilation checks',result['predictions'],'max replay',diff,flush=True)
