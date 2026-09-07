#!/usr/bin/env python3
"""Zero-forward atomic adjudication of the rank16 hidden-weight compiler."""
# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_hashes_and_parent_dispositions_match pred_b_corrected_atomic_authority_passes pred_c_exact_compiler_and_gauge_pass pred_d_full_rank_maps_pass pred_e_participation_hypothesis_remains_false
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path
from circuit_fast_screen_managed_runner import atomic_create_json
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'circuits/prior_art/temporal_iswas_five_mlp_rank16_hidden_weight_compiler_v3_audit.json';V2=ROOT/'circuits/followups/temporal_iswas_five_mlp_rank16_hidden_weight_compiler_v2_result.json';POS=ROOT/'circuits/followups/temporal_iswas_five_mlp_position_svd_ladder_v1_result.json';GAIN=ROOT/'circuits/followups/temporal_iswas_five_mlp_rank16_gain_curve_v1_result.json';REV=ROOT/'circuits/followups/temporal_iswas_five_mlp_rank16_gain115_reverse_v1_result.json';OUT=ROOT/'circuits/followups/temporal_iswas_five_mlp_rank16_hidden_weight_compiler_v3_audit_result.json'
EXPECTED={'v2':'3476ed7c80854bd7e7282062601dff16a775393a8b44ee0879958050961b4198','position':'b6f4562bcc6e19dc808a0debaf35d32b0d6c1dbd6c6a6e7846a92253b89dc014','gain':'958f58cf6ac05973b65a38cd258093926999351016f1aa1bc2ec608c4844b75d','reverse':'c66584f4ee94a2fc7d390b4abcb296ffec933237de43b3924221e7482f465d52'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
def finite(x):
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return not isinstance(x,(int,float)) or math.isfinite(float(x))
def main():
 dry={'candidate_id':'temporal_auxiliary.iswas_five_mlp_rank16_hidden_weight_compiler_v3_audit','dryrun':True,'gpu_accessed':False,'model_loaded':False,'queue_touched':False,'model_forwards':0}
 if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 v=json.loads(V2.read_text());p=json.loads(POS.read_text());g=json.loads(GAIN.read_text());r=json.loads(REV.read_text());observed={'v2':sha(V2),'position':sha(POS),'gain':sha(GAIN),'reverse':sha(REV)}
 a=observed==EXPECTED and p['terminal']=='position_source_rank_boundary' and g['terminal']=='selective_gain_calibrated_source_program' and r['terminal']=='bidirectional_selective_five_mlp_rank16_program'
 atoms={'support_exact':sorted(v['manifest'])==['MLP0','MLP1','MLP2','MLP3','MLP6'],'raw_gram_within_cuda_tolerance':max(x['basis_gram_max_abs'] for x in v['manifest'].values())<=5e-4,'rank16_basis_exists':('rank16' in p['reports']),'gain_selected_1p15':g['selected_gain']==1.15,'reverse_bidirectional':r['terminal']=='bidirectional_selective_five_mlp_rank16_program','finite':finite(v),'price_exact':v['price']=={'model_forwards_max':17,'fit_updates':0,'model_updates':0,'transformer_backwards':0}}
 b=all(atoms.values());c=v['summary']['max_closure_rse']<=1e-10 and v['summary']['max_gauge_error']<=1e-10;d=v['summary']['full_rank_sites']==5;e=v['summary']['nonuniform_sites']<3
 preds={'pred_a_hashes_and_parent_dispositions_match':a,'pred_b_corrected_atomic_authority_passes':b,'pred_c_exact_compiler_and_gauge_pass':c,'pred_d_full_rank_maps_pass':d,'pred_e_participation_hypothesis_remains_false':e};terminal='invalid' if not(a and b and c) else 'distributed_weight_compiled_source_program' if all(preds.values()) else 'null'
 result={'schema':'temporal_iswas_five_mlp_rank16_hidden_weight_compiler_v3_audit_result_v1','created_utc':now(),'authority_sha256':EXPECTED,'atomic_authority':atoms,'v2_terminal_preserved':v['terminal'],'v2_scientific_summary':v['summary'],'predictions':preds,'terminal':terminal,'price':{'model_forwards':0,'example_evaluations':0,'fit_updates':0,'model_updates':0,'transformer_backwards':0}}
 atomic_create_json(OUT,result);print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
