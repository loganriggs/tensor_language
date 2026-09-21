#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_source pred_c_ce
"""Shared linear terms with fixed adaptive512 interaction;48native captures.
pred_a_instrument exact-linear baseline replay and all context deltas invariant<1e-8.
pred_b_source rank256 source-only error<=1.05exact-linear baseline bothdomains.
pred_c_ce rank256 replacement CEadded<.05bothdomains.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=48,linear_ranks=[64,128,256],products=512,fit=False)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_ADAPTIVE_SHARED_LINEAR_GRAPHS_V1.pt';native.SOURCE_CONTEXT_FAMILIES=True;native.OUTPUT_NAME='MIDPOINT_SHARED_LINEAR_NATIVE_RAW_V1.json';native.main()
 r=json.loads((P/native.OUTPUT_NAME).read_text());ss=r['summary'];old=json.loads((P/'MIDPOINT_ADAPTIVE_ALLOCATION_NATIVE_V1.json').read_text())['summary'];checks=[]
 for d in ss:
  for f in ['removal','same_token','source_only','context_only']:checks.append(abs(ss[d]['product_exact_linear'][f]['centered_effect_relative_error']-old[d]['product_adaptive_mixed512'][f]['centered_effect_relative_error']))
  for rank in [64,128,256]:checks.append(abs(ss[d][f'product_linear{rank}']['context_only']['centered_effect_relative_error']-ss[d]['product_exact_linear']['context_only']['centered_effect_relative_error']))
 pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'] and max(checks)<1e-8,pred_b_source=all(ss[d]['product_linear256']['source_only']['centered_effect_relative_error']<=1.05*ss[d]['product_exact_linear']['source_only']['centered_effect_relative_error'] for d in ss),pred_c_ce=all(ss[d]['product_linear256']['replacement']['ce_added']<.05 for d in ss))
 out=P/'MIDPOINT_SHARED_LINEAR_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,summary=ss,replay=max(checks),scope='Fixed centered interaction, shared first-order linear maps. Native context effects protected algebraically; other effects measured. Reused panels, no held fitting or broad adoption.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
