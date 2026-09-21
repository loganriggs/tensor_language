#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_equal_budget pred_c_ce
"""Native512channel baseline,1024products vs256outputrank4.
pred_a_instrument executor exact readers/self-edit/coordinate checks.
pred_b_equal_budget correlation channel full effects <=program256 both domains/families.
pred_c_ce correlation channel CE added<.05 both domains.48 native captures, no fitting.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=48,channel_policies=['correlation','random0'],products=1024)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_CHANNEL_FILE='MIDPOINT_CHANNEL_REFIT_V1.pt';native.OUTPUT_NAME='MIDPOINT_CHANNEL_NATIVE_RAW_V1.json';native.main();r=json.loads((P/native.OUTPUT_NAME).read_text());s=r['summary'];pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'],pred_b_equal_budget=all(s[d]['channel_correlation'][f]['centered_effect_relative_error']<=s[d]['program256'][f]['centered_effect_relative_error'] for d in s for f in ['removal','same_token']),pred_c_ce=all(s[d]['channel_correlation']['replacement']['ce_added']<.05 for d in s));out=P/'MIDPOINT_CHANNEL_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,summary=s,scope='Reused native panels; equal1024product comparison, fixed native readers and calibration output regression. No semantic or individualchannel interventionclaim.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
