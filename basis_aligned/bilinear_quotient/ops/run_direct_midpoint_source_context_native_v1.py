#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_context pred_c_source
"""Native normalized-interface source/context swaps with same-token donors.
pred_a_instrument prior-family results reproduce previous run <1e-8.
pred_b_context rank32 context-only effect error below baseline both domains.
pred_c_source rank32 source-only error <=1.05baseline both domains.
48 capture forwards, reused diagnostic panels, no fitting.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=48,families=['source_only','context_only'],fit=False)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_CENTERED_CORRECTION_GRAPHS_V1.pt';native.SOURCE_CONTEXT_FAMILIES=True;native.OUTPUT_NAME='MIDPOINT_SOURCE_CONTEXT_NATIVE_RAW_V1.json';native.main()
 r=json.loads((P/native.OUTPUT_NAME).read_text());s=r['summary'];old=json.loads((P/'MIDPOINT_CENTERED_CORRECTION_NATIVE_V1.json').read_text())['summary'];checks=[]
 for d in s:
  for name in ['baseline','rank8','rank32']:
   k='product_'+name
   for f in ['removal','same_token']:checks.append(abs(s[d][k][f]['centered_effect_relative_error']-old[d][k][f]['centered_effect_relative_error']))
 pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'] and max(checks)<1e-8,pred_b_context=all(s[d]['product_rank32']['context_only']['centered_effect_relative_error']<s[d]['product_baseline']['context_only']['centered_effect_relative_error'] for d in s),pred_c_source=all(s[d]['product_rank32']['source_only']['centered_effect_relative_error']<=1.05*s[d]['product_baseline']['source_only']['centered_effect_relative_error'] for d in s))
 out=P/'MIDPOINT_SOURCE_CONTEXT_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,summary=s,baseline_replay=max(checks),scope='Same-token donor m, fixed recipient n; context-only uses n minus fixed calibration mean. Cached normalized inputs, explicit residual writes, native final nonlinearities. Not upstream ablation or normalization recomputation. Reused panels; no semantic specificity claim.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
