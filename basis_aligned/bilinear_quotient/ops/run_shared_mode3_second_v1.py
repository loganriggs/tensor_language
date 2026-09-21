#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_absolute pred_c_relative
"""Frozen mode3 source-circuit native confirmation,48captures.
Primary shared_plain256: natural/hybrid<=15%,change<=20%, <=1.10independent128
in everydomain/cohort. Secondnewdocument-indexedFW,16newcodefiles. No fitting.
"""
import os,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import run_direct_source_interchange_v1 as run
 run.PROGRAM_FILE='SHARED_MODE3_FRESH_PROGRAMS_V1.pt';run.PLAN_FILE='SHARED_MODE3_SECOND_PLAN_V1.json';run.DONOR_FILE='SHARED_MODE3_SECOND_DONORS_V1.pt';run.TOKEN_FILE='SHARED_MODE3_SECOND_TOKENS_V1.pt';run.OUTPUT_FILE='SHARED_MODE3_SECOND_NATIVE_V1.json'
 run.PRIMARY='shared_plain256';run.ANCHOR='independent128';run.FORWARDS=48;run.MODE_INDEX=2;run.FOLD_FILE='MODE3_ORIGINAL_SOURCE_FORMS_V1.pt';run.INCLUDE_NATURAL=True
 run.main()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):return
 path=P/run.OUTPUT_FILE;result=json.loads(path.read_text());summary=result['summary']
 primary=run.PRIMARY;baseline=run.ANCHOR
 absolute=all(summary[d][primary][family][cohort]['effect_relative_error']<=limit for d in summary for family,limit in [('natural',.15),('hybrid',.15),('change',.20)] for cohort in ['all','continuation','spaced_word'])
 relative=all(summary[d][primary][family][cohort]['effect_relative_error']<=1.10*summary[d][baseline][family][cohort]['effect_relative_error'] for d in summary for family in ['natural','hybrid','change'] for cohort in ['all','continuation','spaced_word'])
 result['predictions']=dict(pred_a_instrument=result['predictions']['pred_a_instrument'],pred_b_absolute=absolute,pred_c_relative=relative)
 path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['predictions']),flush=True)
if __name__=='__main__':main()
