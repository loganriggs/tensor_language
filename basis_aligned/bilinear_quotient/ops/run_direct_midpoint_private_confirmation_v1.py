#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_frozen pred_b_interventions pred_c_predictive
"""Fresh48-document frozen matched-storage comparison.
pred_a_frozen program/panel hashes and native checks.
pred_b_interventions private512 errors<=1.05original1024 all four intervention families bothdomains.
pred_c_predictive CEadded<.05 and <=original1024+.01bothdomains.
Full removal/swap30% bars also reported, not used to hide high context-only error.
"""
import os,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=48,products=[512,1024],fit=False,fresh_for_study=True)));return
 import torch
 import run_direct_midpoint_full_replace_v1 as native
 meta=json.loads((P/'MIDPOINT_PRIVATE_CONFIRMATION_PANELS_V1.json').read_text())
 for name,digest in meta['program_hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 frozen=json.loads((P/'MIDPOINT_PRIVATE_FROZEN_V1.json').read_text());assert hashlib.sha256((P/'MIDPOINT_PRIVATE_FROZEN_V1.pt').read_bytes()).hexdigest()==frozen['sha256']
 for domain,info in meta['panels'].items():assert hashlib.sha256(torch.load(P/f'MIDPOINT_PRIVATE_CONFIRMATION_{domain.upper()}_V1.pt',weights_only=True).numpy().tobytes()).hexdigest()==info['token_sha256']
 native.PANEL_PREFIX='MIDPOINT_PRIVATE_CONFIRMATION';native.DONOR_PREFIX='MIDPOINT_PRIVATE_CONFIRMATION_DONORS';native.EXTRA_PRODUCT_FILE='MIDPOINT_PRIVATE_FROZEN_V1.pt';native.SOURCE_CONTEXT_FAMILIES=True;native.OUTPUT_NAME='MIDPOINT_PRIVATE_CONFIRMATION_RAW_V1.json';native.main()
 raw=json.loads((P/native.OUTPUT_NAME).read_text());s=raw['summary'];families=['removal','same_token','source_only','context_only']
 pred=dict(pred_a_frozen=raw['predictions']['pred_a_instrument'],pred_b_interventions=all(s[d]['product_private512'][f]['centered_effect_relative_error']<=1.05*s[d]['product_original1024'][f]['centered_effect_relative_error'] for d in s for f in families),pred_c_predictive=all(s[d]['product_private512']['replacement']['ce_added']<.05 and s[d]['product_private512']['replacement']['ce_added']<=s[d]['product_original1024']['replacement']['ce_added']+.01 for d in s))
 secondary=dict(full_effects_below_30_percent=all(s[d]['product_private512'][f]['centered_effect_relative_error']<.3 for d in s for f in ['removal','same_token']))
 out=P/'MIDPOINT_PRIVATE_CONFIRMATION_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,secondary=secondary,summary=s,program_hashes=meta['program_hashes'],prices=frozen['prices'],scope='Frozen before panel construction. New documents in this decompositionstudy, relatedlocalcode not broad externalOOD; pretrainedoverlapunknown. All four intervention families retained and CEtradeoff explicit. No postfit.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
