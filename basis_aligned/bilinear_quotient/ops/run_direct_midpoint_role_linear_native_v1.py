#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_source pred_c_ce
"""Separate-role shared linear basis;48captures,no held fitting.
pred_a_instrument context invariance and exact baseline replay<1e-8.
pred_b_source separate256 source error<=1.05exact baseline bothdomains.
pred_c_ce separate256 CEadded<.05bothdomains.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=48,metrics=['paired','separate','source'],fit=False)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_ROLE_SHARED_LINEAR_GRAPHS_V1.pt';native.SOURCE_CONTEXT_FAMILIES=True;native.OUTPUT_NAME='MIDPOINT_ROLE_LINEAR_NATIVE_RAW_V1.json';native.main()
 r=json.loads((P/native.OUTPUT_NAME).read_text());s=r['summary'];old=json.loads((P/'MIDPOINT_SHARED_LINEAR_NATIVE_V1.json').read_text())['summary'];checks=[]
 for d in s:
  for f in ['removal','same_token','source_only','context_only']:checks.append(abs(s[d]['product_exact_linear'][f]['centered_effect_relative_error']-old[d]['product_exact_linear'][f]['centered_effect_relative_error']))
  for k in ['paired256','separate128','separate256','source256']:checks.append(abs(s[d]['product_'+k]['context_only']['centered_effect_relative_error']-s[d]['product_exact_linear']['context_only']['centered_effect_relative_error']))
 pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'] and max(checks)<1e-8,pred_b_source=all(s[d]['product_separate256']['source_only']['centered_effect_relative_error']<=1.05*s[d]['product_exact_linear']['source_only']['centered_effect_relative_error'] for d in s),pred_c_ce=all(s[d]['product_separate256']['replacement']['ce_added']<.05 for d in s))
 out=P/'MIDPOINT_ROLE_LINEAR_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,summary=s,replay=max(checks),scope='Separate first-order input-role metric, fixed adaptive512 products. Controls paired/shared and source-only. Reused native panels, no held fitting or semantic adoption.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
