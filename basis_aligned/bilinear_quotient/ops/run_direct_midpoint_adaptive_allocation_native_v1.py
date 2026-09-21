#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_adaptive pred_c_preserve
"""Adaptive512 product allocation versus two uniform allocations;48captures.
pred_a_instrument existing baseline families replay<1e-8.
pred_b_adaptive context error beats both uniform512product models bothdomains.
pred_c_preserve source effect <=1.05better uniform model bothdomains.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=48,products=512,fit=False)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_ADAPTIVE_ALLOCATION_GRAPHS_V1.pt';native.SOURCE_CONTEXT_FAMILIES=True;native.OUTPUT_NAME='MIDPOINT_ADAPTIVE_ALLOCATION_NATIVE_RAW_V1.json';native.main()
 r=json.loads((P/native.OUTPUT_NAME).read_text());s=r['summary'];old=json.loads((P/'MIDPOINT_CENTERED_ALLOCATION_V1.json').read_text())['summary'];checks=[]
 for d in s:
  for k in ['w256r2','w512r1']:
   for f in ['source_only','context_only']:checks.append(abs(s[d]['product_'+k][f]['centered_effect_relative_error']-old[d]['product_'+k][f]['centered_effect_relative_error']))
 value=lambda d,k,f:s[d]['product_'+k][f]['centered_effect_relative_error']
 pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'] and max(checks)<1e-8,pred_b_adaptive=all(value(d,'adaptive512','context_only')<min(value(d,k,'context_only') for k in ['w256r2','w512r1']) for d in s),pred_c_preserve=all(value(d,'adaptive512','source_only')<=1.05*min(value(d,k,'source_only') for k in ['w256r2','w512r1']) for d in s))
 out=P/'MIDPOINT_ADAPTIVE_ALLOCATION_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,summary=s,replay=max(checks),scope='Fixed512 variableproducts, adaptive original-weight slice-rank allocation. Exact native first-order terms priced. Uniform baselines share outputbasis/metric. Reused native diagnostics; no held fitting or semantic identification.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
