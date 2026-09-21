#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_preserve pred_c_ce
"""Jointtensor productselection followedby coefficientmetric outputrefit.
pred_a_instrument native default checks.
pred_b_preserve 512products full effects<=1.05shared1024product baseline bothdomains/families.
pred_c_ce 512products CEadded<.05bothdomains.48captures,no fitting,reusedpanels.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=48,products=[256,512],fit=False)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_GRAPH_PRODUCT_PRUNE_V1.pt';native.OUTPUT_NAME='MIDPOINT_PRODUCT_PRUNE_NATIVE_RAW_V1.json';native.main();r=json.loads((P/native.OUTPUT_NAME).read_text());s=r['summary'];old=json.loads((P/'MIDPOINT_SHARED_GRAPH_NATIVE_V1.json').read_text())['summary'];pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'],pred_b_preserve=all(s[d]['product_products512'][f]['centered_effect_relative_error']<=1.05*old[d]['product_rank256'][f]['centered_effect_relative_error'] for d in s for f in ['removal','same_token']),pred_c_ce=all(s[d]['product_products512']['replacement']['ce_added']<.05 for d in s));out=P/'MIDPOINT_PRODUCT_PRUNE_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,summary=s,scope='Actual selectedproductgraph including compactoutputframe. Fullsource-dependent target, reusedpanels. Exactwriterfit metric differs from nativefunctionmetric; nosemantic orfreshconfirmation claim.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
