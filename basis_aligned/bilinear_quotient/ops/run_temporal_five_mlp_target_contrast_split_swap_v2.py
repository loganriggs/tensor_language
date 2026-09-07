#!/usr/bin/env python3
"""Opposite-half confirmation of the fixed rank-8 target-contrast program."""
# BQGATE: EXPERIMENT pred_a_authority_disjointness_alignment_finiteness_and_price pred_b_swapped_target_fidelity pred_c_swapped_control_selectivity pred_d_crossfit_target_stability pred_e_crossfit_control_stability
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_five_mlp_target_contrast_response_basis_v1 as v1
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
ROOT=Path(__file__).resolve().parents[1];PRIOR=ROOT/"circuits/prior_art/temporal_five_mlp_target_contrast_split_swap_v2.json";OLD=ROOT/"circuits/followups/temporal_five_mlp_target_contrast_response_basis_v1_result.json";OUT=ROOT/"circuits/followups/temporal_five_mlp_target_contrast_split_swap_v2_result.json"
EXPECTED={"prior":"e9721b721d4e1a982e9ad5d0208d59b2128b9e2fb961f033d65dd367f10d0312","old":"db8d94bda707546878c78d62bde3c9bbb5b5510557f6fd9b3f042dbb05359f17","runner":"8843cbb584d87823aedc346361b3f62679e06b8f31901b057b80b4d6b403cad3"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def main():
 if {"prior":sha(PRIOR),"old":sha(OLD),"runner":sha(Path(v1.__file__))}!=EXPECTED:raise RuntimeError("authority changed")
 old=json.loads(OLD.read_text());dry={"candidate_id":"temporal_auxiliary.five_mlp_target_contrast_split_swap_v2","dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"rank":8,"model_forwards_max":10,"fit_updates":0,"model_updates":0,"transformer_backwards":0}
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dry,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 tcap,icap=json.loads(comp.TCAP.read_text()),json.loads(comp.ICAP.read_text());tr,ir,*_=greedy.rows_and_controls(tcap,icap);target_fit=tr[1::2]+ir[1::2];target_eval=tr[::2]+ir[::2];controls=matched.control_rows();control_fit,control_eval=controls["validation"],controls["discovery"]
 authority_ok=len(target_fit)==len(target_eval)==27 and len(control_fit)==20 and len(control_eval)==22 and not({r["row_id"] for r in target_fit}&{r["row_id"] for r in target_eval}) and all(r["base_semantic_position"]==r["donor_semantic_position"] for rows in (target_fit,target_eval,control_fit,control_eval) for r in rows)
 if not authority_ok:raise RuntimeError("split changed")
 tic=time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");t=backend.torch;reader,orientation,reader_ok=greedy.physical_reader(backend,json.loads(comp.WEIGHTS.read_text()))
 cfb,cfo,cf0,cfd,cf1=v1.cap(backend,control_fit);tfb,tfo,tf0,tfd,tf1=v1.cap(backend,target_fit);teb,teo,te0,ted,te1=v1.cap(backend,target_eval);ceb,ceo,ce0,ced,ce1=v1.cap(backend,control_eval)
 cb,_=comp.fit_bases(backend,cfb,cf0,cf1);bases,spectra=v1.fit_target(backend,tfb,tf0,tf1,cb);to=v1.run_project(backend,teb,te0,te1,bases,8);co=v1.run_project(backend,ceb,ce0,ce1,bases,8)
 tebs,teds,tes=[atlasrun.states(t,backend,x,target_eval) for x in (teo,ted,to)];cebs,ceds,ces=[atlasrun.states(t,backend,x,control_eval) for x in (ceo,ced,co)];target=v1.report(backend,target_eval,tebs,teds,tes,reader,len(tr[::2]));scales=json.loads(comp.GENERIC.read_text())["target_behavior_rms_scales"];cut=sum(r["task_id"]==matched.temporal.TASK_ID for r in control_eval);control=v1.report(backend,control_eval,cebs,ceds,ces,reader,cut,scales)
 finite=all(math.isfinite(x) for x in list(target["cells"].values())+list(target["behavior_signed_projection"].values())+list(control.values()));pa=authority_ok and reader_ok and orientation<=1e-6 and finite and all(spectra[s][7]>0 for s in spectra);pb=target["worst_target_residual"]<=.05 and all(target["behavior_signed_projection"][x]>=.8 for x in control);pc=all(control[x]<=.1 for x in control);old8=old["reports"]["8"];pd=all(abs(target["behavior_signed_projection"][x]-old8["target"]["behavior_signed_projection"][x])<=.15 for x in control) and abs(target["worst_target_residual"]-old8["target"]["worst_target_residual"])<=.03;pe=all(control[x]<=.03 and abs(control[x]-old8["control"][x])<=.02 for x in control)
 preds={"pred_a_authority_disjointness_alignment_finiteness_and_price":bool(pa),"pred_b_swapped_target_fidelity":bool(pb),"pred_c_swapped_control_selectivity":bool(pc),"pred_d_crossfit_target_stability":bool(pd),"pred_e_crossfit_control_stability":bool(pe)};terminal="invalid" if not pa else "crossfit_selective_response_program" if all(preds.values()) else "split_specific_null" if not(pb and pc) else "unstable_identification";result={"schema":"temporal_five_mlp_target_contrast_split_swap_result_v2","started_utc":now(),"finished_utc":now(),"serial_seconds":time.perf_counter()-tic,"authority_sha256":EXPECTED,"target_report":target,"control_report":control,"spectra":spectra,"predictions":preds,"terminal":terminal,"price":{"model_forwards":10,"fit_updates":0,"model_updates":0,"transformer_backwards":0}};atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("target_report","control_report","predictions","terminal","price")},sort_keys=True))
if __name__=="__main__":main()
