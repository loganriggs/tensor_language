#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_frozen pred_b_individual pred_c_joint
"""Fresh midpoint confirmation; nofit, fixedprogramhash andpanels.
pred_a_frozen manifestprogramhash and allinstrumentchecks;
pred_b_individual error<.3/cos>.95 allindividual removals/same-token swaps;
pred_c_joint error<.2 forboth domains/interventions.
Null earlier results are diagnostic selection artifacts. Price96nativeforwards.
"""
import os,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=96,panels='MIDPOINT_CONFIRMATION',fit=False)));return
 import run_direct_midpoint_removal_v1 as removal
 import run_direct_midpoint_swap_v1 as swap
 meta=json.loads((P/'MIDPOINT_CONFIRMATION_PANELS_V1.json').read_text());assert hashlib.sha256((P/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt').read_bytes()).hexdigest()==meta['program_sha256'];out=P/'MIDPOINT_CONFIRMATION_V1.json';assert not out.exists()
 for module,kind in [(removal,'REMOVAL'),(swap,'SWAP')]:
  module.PANEL_PREFIX='MIDPOINT_CONFIRMATION';module.DONOR_PREFIX='MIDPOINT_CONFIRMATION_DONORS';module.OUTPUT_NAME=f'MIDPOINT_CONFIRMATION_{kind}_V1.json';module.main()
 rr=json.loads((P/removal.OUTPUT_NAME).read_text());ss=json.loads((P/swap.OUTPUT_NAME).read_text());pred=dict(pred_a_frozen=rr['predictions']['pred_a_export'] and ss['predictions']['pred_a_instrument'],pred_b_individual=rr['predictions']['pred_b_features'] and ss['predictions']['pred_b_features'],pred_c_joint=rr['predictions']['pred_c_joint'] and ss['predictions']['pred_c_joint'])
 result=dict(predictions=pred,removal=rr['summary'],swap=ss['summary'],program_sha256=meta['program_sha256'],scope='New documents for this midpoint program, frozen before evaluation. Related localcode, pretrained overlapunknown. Same native operational readers/writers; no semanticselectivity claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
