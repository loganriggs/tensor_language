#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_preserve pred_c_ce
"""Shared centered correction native screen; 48 captures, reused panels.
pred_a_instrument native replay checks and old baseline summary replay.
pred_b_preserve rank32 removal/swap errors <=1.05baseline on both domains.
pred_c_ce rank32 CE added <.05 on both domains.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=48,products=512,correction_ranks=[8,32],fit=False)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_CENTERED_CORRECTION_GRAPHS_V1.pt';native.OUTPUT_NAME='MIDPOINT_CENTERED_CORRECTION_NATIVE_RAW_V1.json';native.main()
 r=json.loads((P/native.OUTPUT_NAME).read_text());s=r['summary'];old=json.loads((P/'MIDPOINT_PRODUCT_PRUNE_NATIVE_V1.json').read_text())['summary'];checks=[]
 for d in s:
  for f in ['removal','same_token']:checks.append(abs(s[d]['product_baseline'][f]['centered_effect_relative_error']-old[d]['product_products512'][f]['centered_effect_relative_error']))
  checks.append(abs(s[d]['product_baseline']['replacement']['ce_added']-old[d]['product_products512']['replacement']['ce_added']))
 pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'] and max(checks)<1e-8,pred_b_preserve=all(s[d]['product_rank32'][f]['centered_effect_relative_error']<=1.05*s[d]['product_baseline'][f]['centered_effect_relative_error'] for d in s for f in ['removal','same_token']),pred_c_ce=all(s[d]['product_rank32']['replacement']['ce_added']<.05 for d in s))
 out=P/'MIDPOINT_CENTERED_CORRECTION_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,summary=s,baseline_replay=max(checks),scope='512 sharedproducts plus centered rank8/32 original-operator correction. Reused diagnostic panels; no fitting. Native final normalization/softcap active. Aggregate preservation not semantic identification.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
