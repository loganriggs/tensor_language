#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RESULT=ROOT/'basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_native_scalar_feature_discovery_v1_result.json';BINDING=HERE/'SUBJECT_NUMBER_NATIVE_SCALAR_FEATURE_DISCOVERY_V1_BINDING.json';OUT=HERE/'SUBJECT_NUMBER_NATIVE_SCALAR_FEATURE_DISCOVERY_V1_AUDIT.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads(RESULT.read_text());checks={'valid_null':r['terminal']=='native_scalar_feature_discovery_null','instrument':r['predictions']['pred_a_exact_activation_instrument'],'predictive_gate_failed':not r['predictions']['pred_b_native_singular_coordinates_predict_amplitude'],'compact_rank_selected':r['selected_rank']==2 and r['predictions']['pred_c_compact_rank_selected'],'complete_panel':r['instrument']['examples']==512,'exact_fit_count':r['instrument']['fits']==12,'zero_closure':max(r['instrument']['role_state_closure_max_absolute_error'],r['instrument']['role_normalized_closure_max_absolute_error'])==0,'outcome_blind':not any(r['outcome_access'].values())};payload={'schema':'subject_number_native_scalar_feature_discovery_v1_audit','result_sha256':sha(RESULT),'binding_sha256':sha(BINDING),'checks':checks,'all_checks_pass':all(checks.values())};OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
