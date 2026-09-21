#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_preserve pred_c_ce
"""Actual sharedinput and sharedoutput graph: ranks64/128/256,1024products.
pred_a_instrument native default checks and baseline energy/error replay<1e-5.
pred_b_preserve rank256 full effects<=1.05frozenrank8 bothdomains/families.
pred_c_ce rank256 replacement CEadded<.05bothdomains.48captures, reusedpanels,no fitting.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=48,shared_input_ranks=[64,128,256],products=1024)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_GRAPH_INPUT_MODE_GRAPHS_V1.pt';native.OUTPUT_NAME='MIDPOINT_SHARED_GRAPH_NATIVE_RAW_V1.json';native.main();r=json.loads((P/native.OUTPUT_NAME).read_text());s=r['summary'];old=json.loads((P/'MIDPOINT_CORRECTION_NATIVE_V1.json').read_text())['summary'];replay=max(abs(s[d]['program256'][f]['centered_effect_relative_error']-old[d]['program256'][f]['centered_effect_relative_error']) for d in s for f in ['removal','same_token']);pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'] and replay<1e-5,pred_b_preserve=all(s[d]['product_rank256'][f]['centered_effect_relative_error']<=1.05*old[d]['product_rank8'][f]['centered_effect_relative_error'] for d in s for f in ['removal','same_token']),pred_c_ce=all(s[d]['product_rank256']['replacement']['ce_added']<.05 for d in s));out=P/'MIDPOINT_SHARED_GRAPH_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,summary=s,baseline_replay=replay,scope='Actualsharedinput+sharedoutput execution. Marginalweighted mode subspaces fitoriginalcalibration; reuseddiagnostics, nofit. Nativeupstreamstates stillsupplied; nosemantic or wholemodelspeed claim.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
