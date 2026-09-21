#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_whole pred_c_source
"""Private versus shared linear output spaces at equal coefficient cost.
pred_a_instrument shared256 replay and context invariance<1e-8.
pred_b_whole private_adaptive whole-swap error<shared256 bothdomains.
pred_c_source private_adaptive source-only error<=1.05shared256 bothdomains.
48captures, fixed512products, no held fitting. Equal-private-rank control included.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=48,linear_coefficients=884736,total_coefficients=2654208,fit=False)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_PRIVATE_LINEAR_GRAPHS_V1.pt';native.SOURCE_CONTEXT_FAMILIES=True;native.OUTPUT_NAME='MIDPOINT_PRIVATE_LINEAR_NATIVE_RAW_V1.json';native.main()
 r=json.loads((P/native.OUTPUT_NAME).read_text());s=r['summary'];old=json.loads((P/'MIDPOINT_ROLE_LINEAR_NATIVE_V1.json').read_text())['summary'];checks=[]
 for d in s:
  for f in ['removal','same_token','source_only','context_only']:checks.append(abs(s[d]['product_shared256'][f]['centered_effect_relative_error']-old[d]['product_separate256'][f]['centered_effect_relative_error']))
  for k in ['private_equal','private_adaptive']:checks.append(abs(s[d]['product_'+k]['context_only']['centered_effect_relative_error']-s[d]['product_shared256']['context_only']['centered_effect_relative_error']))
 value=lambda d,k,f:s[d]['product_'+k][f]['centered_effect_relative_error']
 pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'] and max(checks)<1e-8,pred_b_whole=all(value(d,'private_adaptive','same_token')<value(d,'shared256','same_token') for d in s),pred_c_source=all(value(d,'private_adaptive','source_only')<=1.05*value(d,'shared256','source_only') for d in s))
 out=P/'MIDPOINT_PRIVATE_LINEAR_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,summary=s,replay=max(checks),scope='Same full parameter/product price; private branch output spaces with344/40adaptive ranks versus common256dictionary and private192/192control. Reused native panels. No claim of preservation against exact-linear program or semantic adoption.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
