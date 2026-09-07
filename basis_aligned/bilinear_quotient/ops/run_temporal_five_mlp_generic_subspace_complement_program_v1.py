#!/usr/bin/env python3
"""Remove discovery-control response subspaces inside a fixed causal program."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_self_patch_finiteness_and_price pred_b_unprojected_replay_matches_prior pred_c_generic_projection_reduces_controls pred_d_complement_preserves_target pred_e_complement_is_selective
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_generic_subspace_complement_program_v1.json"
GENERIC = ROOT / "circuits/followups/temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1_result.json"
CONTROL = ROOT / "circuits/followups/temporal_five_mlp_matched_control_upstream_atlas_v2_result.json"
WEIGHTS = ROOT / "circuits/followups/temporal_iswas_two_mode_weight_pullback_v3_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_generic_subspace_complement_program_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.five_mlp_generic_subspace_complement_program_v1"
SITES = ("MLP1", "L9H1", "MLP3", "L9H4", "L8H1", "MLP6", "MLP4", "L11H3")
RANKS = (0, 1, 4, 8)
EXPECTED = {"prior":"4b48e7511e01de24e6ff6f80489a30aab0997d42526b150fdf3292c411fdbc7a",
            "generic":"57402478b86e88237bb745824e7aa8e6d56c17336753cee3d1b4e9359b5febe3",
            "control":"b75a4b81b8d2407ea3661f21be8b4a3bf967bc8a242fc08d562aa6579b97bb69"}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")

def fit_bases(backend, batch, base, donor):
    torch, width = backend.torch, backend.model.config.n_embd // backend.model.config.n_head
    bases, spectra = {}, {}
    for site in SITES:
        kind, layer, head = atlasrun.site_parts(site)
        family = "mlp" if kind == "mlp" else "attention"
        pieces = []
        for i, pos in enumerate(batch.semantic_positions):
            delta = donor[family][layer][i,:int(pos)+1] - base[family][layer][i,:int(pos)+1]
            if kind == "attn": delta = delta[:,head*width:(head+1)*width]
            pieces.append(delta)
        matrix = torch.cat(pieces).to(backend.device).float()
        _u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        bases[site] = vh[:8].T.contiguous()
        spectra[site] = [float(x) for x in s[:12]]
    return bases, spectra

def run_complement(backend, batch, base, donor, bases, rank):
    width, handles = backend.model.config.n_embd // backend.model.config.n_head, []
    for site in SITES:
        kind, layer, head = atlasrun.site_parts(site); q = bases[site][:,:rank]
        if kind == "attn":
            b, d = base["attention"][layer], donor["attention"][layer]
            def hook(_m,args,b=b,d=d,q=q,head=head):
                changed=args[0].clone(); start,stop=head*width,(head+1)*width
                for i,pos in enumerate(batch.semantic_positions):
                    n=int(pos)+1; delta=(d[i,:n,start:stop]-b[i,:n,start:stop]).to(changed)
                    if q.shape[1]: delta=delta-(delta@q)@q.T
                    changed[i,:n,start:stop]=b[i,:n,start:stop].to(changed)+delta
                return (changed,)+tuple(args[1:])
            handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(hook))
        else:
            b, d = base["mlp"][layer], donor["mlp"][layer]
            def hook(_m,_a,out,b=b,d=d,q=q):
                changed=out.clone()
                for i,pos in enumerate(batch.semantic_positions):
                    n=int(pos)+1; delta=(d[i,:n]-b[i,:n]).to(changed)
                    if q.shape[1]: delta=delta-(delta@q)@q.T
                    changed[i,:n]=b[i,:n].to(changed)+delta
                return changed
            handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(hook))
    try: return backend.native(batch,capture=True)
    finally:
        for h in handles: h.remove()

def margins(backend,state,rows):
    torch=backend.torch; logits=das.head_logits(backend,state); ix=torch.arange(len(rows),device=backend.device)
    a=torch.as_tensor([r["donor_answer_id"] for r in rows],device=backend.device)
    f=torch.as_tensor([r["donor_foil_id"] for r in rows],device=backend.device)
    return logits[ix,a]-logits[ix,f]

def main():
    if {"prior":sha(PRIOR),"generic":sha(GENERIC),"control":sha(CONTROL)} != EXPECTED: raise RuntimeError("authority changed")
    generic, control, weights = [json.loads(p.read_text()) for p in (GENERIC,CONTROL,WEIGHTS)]
    dry={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"ranks":RANKS,"sites":SITES,"model_forwards_max":16,"fit_updates":0,"model_updates":0,"transformer_backwards":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(dry,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    tcap,icap=json.loads(TCAP.read_text()),json.loads(ICAP.read_text())
    tr,ir,*_=greedy.rows_and_controls(tcap,icap); target_rows=tr+ir
    controls=matched.control_rows(); discovery,validation=controls["discovery"],controls["validation"]
    authority_ok=generic["terminal"]=="generic_transport_null" and control["terminal"]=="generic_transport_confirmed_no_selective_pool" and tuple(generic["selected"])==SITES and len(target_rows)==54 and len(discovery)==22 and len(validation)==20
    if not authority_ok: raise RuntimeError("population changed")
    tic=time.perf_counter(); started=now(); backend=producer.Bilin18TorchBackend.load("cuda"); torch=backend.torch
    reader,orientation,reader_ok=greedy.physical_reader(backend,weights)
    def captures(rows):
        bb,db=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor")
        bo,bc=atlasrun.capture_native(backend,bb); do,dc=atlasrun.capture_native(backend,db)
        return bb,bo,bc,do,dc
    disc_batch,_disc_out,disc_base,_disc_donor_out,disc_donor=captures(discovery)
    target_batch,target_base_out,target_base,target_donor_out,target_donor=captures(target_rows)
    val_batch,val_base_out,val_base,_val_donor_out,val_donor=captures(validation)
    bases,spectra=fit_bases(backend,disc_batch,disc_base,disc_donor)
    target_base_state=atlasrun.states(torch,backend,target_base_out,target_rows); target_donor_state=atlasrun.states(torch,backend,target_donor_out,target_rows)
    val_base_state=atlasrun.states(torch,backend,val_base_out,validation)
    ordinary,_=atlasrun.run_patch(backend,target_batch,target_donor,SITES); ordinary_state=atlasrun.states(torch,backend,ordinary,target_rows)
    self_out,_=atlasrun.run_patch(backend,target_batch,target_base,SITES); self_state=atlasrun.states(torch,backend,self_out,target_rows)
    full_b=margins(backend,target_donor_state,target_rows)-margins(backend,target_base_state,target_rows); full_m=(target_donor_state-target_base_state)@reader
    task_ids={"temporal":slice(0,len(tr)),"iswas":slice(len(tr),len(target_rows))}; val_n=sum(r["task_id"]==matched.temporal.TASK_ID for r in validation); val_ids={"temporal":slice(0,val_n),"iswas":slice(val_n,len(validation))}
    reports={}
    rank0_state=None
    for rank in RANKS:
        to=run_complement(backend,target_batch,target_base,target_donor,bases,rank); ts=atlasrun.states(torch,backend,to,target_rows)
        vo=run_complement(backend,val_batch,val_base,val_donor,bases,rank); vs=atlasrun.states(torch,backend,vo,validation)
        if rank==0: rank0_state=ts
        db=margins(backend,ts,target_rows)-margins(backend,target_base_state,target_rows); dm=(ts-target_base_state)@reader
        cells={}; projections={}
        for task,ids in task_ids.items():
            cells[f"{task}_behavior"]=float((db[ids]-full_b[ids]).square().sum()/full_b[ids].square().sum())
            projections[task]=float(db[ids]@full_b[ids]/full_b[ids].square().sum())
            for j in range(2): cells[f"{task}_mode{j+1}"]=float((dm[ids,j]-full_m[ids,j]).square().sum()/full_m[ids,j].square().sum())
        vd=margins(backend,vs,validation)-margins(backend,val_base_state,validation)
        ctl={task:float(vd[ids].square().mean().sqrt())/generic["target_behavior_rms_scales"][task] for task,ids in val_ids.items()}
        reports[str(rank)]={"worst_target_residual":max(cells.values()),"cells":cells,"behavior_signed_projection":projections,"validation_control_fraction":ctl}
    replay_error=float((rank0_state-ordinary_state).abs().max()); self_error=float((self_state-target_base_state).abs().max())
    candidates=[r for r in (4,8) if all(reports[str(r)]["validation_control_fraction"][t] <= .5*reports["0"]["validation_control_fraction"][t] for t in task_ids)]
    chosen=min(candidates,key=lambda r:reports[str(r)]["worst_target_residual"]) if candidates else None
    pa=authority_ok and reader_ok and orientation<=1e-6 and self_error<=1e-4 and all(math.isfinite(x) for z in reports.values() for x in z["cells"].values())
    pb=replay_error<=1e-4 and abs(reports["0"]["worst_target_residual"]-generic["selected_report"]["worst_residual"])<=.01
    pc=chosen is not None; pd=pc and reports[str(chosen)]["worst_target_residual"]<=.15 and all(reports[str(chosen)]["behavior_signed_projection"][t]>=.8 for t in task_ids); pe=pd and all(reports[str(chosen)]["validation_control_fraction"][t]<=.1 for t in task_ids)
    preds={"pred_a_authority_alignment_self_patch_finiteness_and_price":bool(pa),"pred_b_unprojected_replay_matches_prior":bool(pb),"pred_c_generic_projection_reduces_controls":bool(pc),"pred_d_complement_preserves_target":bool(pd),"pred_e_complement_is_selective":bool(pe)}
    terminal="invalid" if not(pa and pb) else "within_module_selective_program" if all(preds.values()) else "generic_task_entangled_null" if pc else "generic_subspace_miss_null"
    result={"schema":"temporal_five_mlp_generic_subspace_complement_program_result_v1","candidate_id":CANDIDATE_ID,"started_utc":started,"finished_utc":now(),"serial_seconds":time.perf_counter()-tic,"authority_sha256":EXPECTED,"instrument":{"authority_ok":authority_ok,"reader_hash_ok":reader_ok,"orientation_max_abs":orientation,"self_patch_max_abs":self_error,"rank0_ordinary_state_max_abs":replay_error},"spectra":spectra,"reports":reports,"chosen_rank":chosen,"predictions":preds,"terminal":terminal,"price":{"model_forwards":16,"fit_updates":0,"model_updates":0,"transformer_backwards":0}}
    atomic_create_json(OUT,result); print(json.dumps({k:result[k] for k in ("instrument","reports","chosen_rank","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__": main()
