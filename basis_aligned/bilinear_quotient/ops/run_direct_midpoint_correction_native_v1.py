#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_preserve pred_c_cegain
"""Rank8/32 linear graph corrections; mean-only rank0 control.48captures.
pred_a_instrument native evaluator checks.
pred_b_preserve rank8 effects<=1.02denseanchored both domains/families.
pred_c_cegain rank8 codeCE retains at least half original-to-anchored improvement.
No fitting; same1024products, rank8 adds17408weightcoefficients.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=48,correction_ranks=[0,8,32],products=1024)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_OUTPUT_CORRECTION_COMPARE_V1.pt';native.OUTPUT_NAME='MIDPOINT_CORRECTION_NATIVE_RAW_V1.json';native.main();r=json.loads((P/native.OUTPUT_NAME).read_text());s=r['summary'];old=json.loads((P/'MIDPOINT_PRODUCT_REFIT_NATIVE_V1.json').read_text())['summary'];pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'],pred_b_preserve=all(s[d]['product_rank8'][f]['centered_effect_relative_error']<=1.02*old[d]['product_anchored'][f]['centered_effect_relative_error'] for d in s for f in ['removal','same_token']),pred_c_cegain=s['code']['product_rank8']['replacement']['ce_added']<=(old['code']['program256']['replacement']['ce_added']+old['code']['product_anchored']['replacement']['ce_added'])/2);out=P/'MIDPOINT_CORRECTION_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,summary=s,scope='Lowrank output correction of learned products; reuseddiagnostics. Rank0 isolates mean correction. Sameproducts, extra linear reuse; not new semanticidentity.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
