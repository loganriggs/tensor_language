#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_improve pred_c_ce
"""Fixed1024learnedproducts: grouped vs zero-prior and weight-anchored writers.
pred_a_instrument native evaluator checks.
pred_b_improve anchored full effects <=original grouped both domains/families.
pred_c_ce anchored CEadded<.05 both domains.48 captures; no fitting.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=48,products=1024,refits=['ungrouped','anchored'])));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_PRODUCT_REFIT_COMPARE_V1.pt';native.OUTPUT_NAME='MIDPOINT_PRODUCT_REFIT_NATIVE_RAW_V1.json';native.main();r=json.loads((P/native.OUTPUT_NAME).read_text());s=r['summary'];pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'],pred_b_improve=all(s[d]['product_anchored'][f]['centered_effect_relative_error']<=s[d]['program256'][f]['centered_effect_relative_error'] for d in s for f in ['removal','same_token']),pred_c_ce=all(s[d]['product_anchored']['replacement']['ce_added']<.05 for d in s));out=P/'MIDPOINT_PRODUCT_REFIT_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,summary=s,scope='Full contribution; output regrouping of fixedlearnedproducts, reuseddiagnostics. No semantic identification or cost saving claimed.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
