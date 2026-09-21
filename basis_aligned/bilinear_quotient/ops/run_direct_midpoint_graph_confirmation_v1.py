#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_frozen pred_b_absolute pred_c_gain
"""Fresh48documents for frozen rank8 output-correction graph.
pred_a_frozen program/panel hashes and native instrument.
pred_b_absolute full effects<.3 bothdomains/families and CEadded<.05both.
pred_c_gain codeCE at least10%lower than originalgrouped program.
48captures, no fitting. Fresh to this decompositionstudy; localcode not broad externalOOD.
"""
import os,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=48,fit=False,products=1024,correction_rank=8)));return
 import torch
 import run_direct_midpoint_full_replace_v1 as native
 meta=json.loads((P/'MIDPOINT_GRAPH_CONFIRMATION_PANELS_V1.json').read_text())
 for name,digest in meta['program_hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 for domain,info in meta['panels'].items():assert hashlib.sha256(torch.load(P/f'MIDPOINT_GRAPH_CONFIRMATION_{domain.upper()}_V1.pt',weights_only=True).numpy().tobytes()).hexdigest()==info['token_sha256']
 native.PANEL_PREFIX='MIDPOINT_GRAPH_CONFIRMATION';native.DONOR_PREFIX='MIDPOINT_GRAPH_CONFIRMATION_DONORS';native.EXTRA_PRODUCT_FILE='MIDPOINT_GRAPH_FROZEN_V1.pt';native.OUTPUT_NAME='MIDPOINT_GRAPH_CONFIRMATION_RAW_V1.json';native.main();r=json.loads((P/native.OUTPUT_NAME).read_text());s=r['summary'];pred=dict(pred_a_frozen=r['predictions']['pred_a_instrument'],pred_b_absolute=all(s[d]['product_rank8'][f]['centered_effect_relative_error']<.3 for d in s for f in ['removal','same_token']) and all(s[d]['product_rank8']['replacement']['ce_added']<.05 for d in s),pred_c_gain=s['code']['product_rank8']['replacement']['ce_added']<=.9*s['code']['program256']['replacement']['ce_added']);out=P/'MIDPOINT_GRAPH_CONFIRMATION_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,summary=s,program_hashes=meta['program_hashes'],scope='Frozenprograms, newdocuments for this decomposition study; nofit orpostselection. Relatedlocalcode, pretrainedoverlapunknown. Fullmidpointcontribution with upstream native states explicit.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
