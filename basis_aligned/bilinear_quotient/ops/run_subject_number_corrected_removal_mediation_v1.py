#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_corrected_instrument_and_capability pred_b_selective_coordinate_removal pred_c_small_full_head_mediation pred_d_rank1_reusable_edge
"""Corrected behavioral removal and L11H3 mediation with matched nulls."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


RUNNER = Path(__file__).resolve(); ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json"
RANK1 = POLY / "SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
SCALAR = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_CORRECTED_V1_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_CORRECTED_REMOVAL_MEDIATION_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_CORRECTED_REMOVAL_MEDIATION_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_CORRECTED_REMOVAL_MEDIATION_V1_RESULT.json"
LAYER, HEAD, HEAD_WIDTH = 11, 3, 128
NULLS, SEED = 16, 20260924
BARS = {"maximum_axis_damage_recovery": .18,
        "maximum_axis_to_full_relative_l2": .20,
        "maximum_control_fraction": .50,
        "maximum_full_damage_recovery": .20,
        "maximum_rank1_norm_ratio": 1.25,
        "maximum_state_error": 1e-5,
        "minimum_axis_damage_recovery": .04,
        "minimum_axis_full_cosine": .97,
        "minimum_cell_positive_fraction": .75,
        "minimum_damage_fraction": .05,
        "minimum_damage_rms": .5,
        "minimum_full_damage_cosine": .75,
        "minimum_full_damage_recovery": .05,
        "minimum_full_positive_fraction": .70,
        "minimum_native_accuracy": .75,
        "minimum_random_embedding_median_ratio": 2.,
        "minimum_random_head_median_ratio": 10.,
        "minimum_rank1_norm_ratio": .75}
PRICE = {"forwards": 72, "sequences": 4608, "embedding_nulls": 16,
         "head_nulls": 16, "fits": 0, "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_corrected_instrument_and_capability": None,
                       "pred_b_selective_coordinate_removal": None,
                       "pred_c_small_full_head_mediation": None,
                       "pred_d_rank1_reusable_edge": None}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    pn, an = np.linalg.norm(predicted), np.linalg.norm(actual)
    return {"cosine": float(predicted @ actual / max(pn*an, 1e-30)),
            "relative_l2_error": float(np.linalg.norm(predicted-actual)/max(an,1e-30)),
            "positive_fraction": float(np.mean(predicted > 0)),
            "predicted_to_actual_norm_ratio": float(pn/max(an,1e-30)),
            "least_squares_recovery": float(predicted @ actual/max(actual@actual,1e-30)),
            "effect_rms": float(np.sqrt(np.mean(predicted**2)))}


def groups(rows):
    out = {"overall": list(range(len(rows)))}
    for field, levels in (("number", ("singular","plural")),
                          ("template_id", ("near","behind","under","above")),
                          ("stratum", ("irregular","regular"))):
        for level in levels: out[f"{field}|{level}"] = [i for i,r in enumerate(rows) if r[field]==level]
    return out


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"authority": Path(authority.__file__), "decoder": DECODER,
             "rank1": RANK1, "corrected_scalar": SCALAR, "preregistration": PREREG}
    if binding["files"] != {name:sha(path) for name,path in paths.items()} \
            or binding["authority_sha256"] != authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"] != BARS or binding["nulls"] != NULLS \
            or binding["null_seed"] != SEED or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, rank, scalar = (json.loads(p.read_text()) for p in (DECODER,RANK1,SCALAR))
    if decoder["terminal"] != "embedding_number_decoder_frozen" \
            or rank["terminal"] != "rank1_frozen_weights_only" \
            or scalar["terminal"] != "corrected_context_rank1_interaction_candidate":
        raise ValueError("parent status changed")
    rows = authority.build_rows()
    return binding,decoder,rank,scalar,rows


def plan():
    _,decoder,rank,scalar,rows=load_bound()
    return {"schema":"subject_number_corrected_removal_mediation_v1_plan",
            "model_loaded":False,"gpu_accessed":False,"queue_touched":False,
            "rows":len(rows),"batch_sizes":[64,64],"embedding_nulls":NULLS,
            "head_nulls":NULLS,"rank":rank["rank"],
            "decoder_axis_sha256":decoder["frozen_decoder"]["axis_float32_sha256"],
            "scalar_terminal":scalar["terminal"],"authority_sha256":authority.canonical(rows),
            "bars":BARS,"price":PRICE,"binding_sha256":sha(BINDING)}


def main():
    planned=plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned,indent=2,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    signal.alarm(600)
    binding,decoder,rank,scalar,rows=load_bound()
    torch,F,facade=tangent.parent.factors._dependencies(); torch.set_num_threads(2)
    model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
    started=time.perf_counter(); device=next(model.parameters()).device
    decoder_axis=torch.tensor(decoder["frozen_decoder"]["axis"],dtype=torch.float64,device=device)
    threshold=float(decoder["frozen_decoder"]["threshold"]); unit_decoder=decoder_axis/decoder_axis.norm()
    writer_axis=torch.tensor(rank["axis"],dtype=torch.float32,device=device); writer_axis/=writer_axis.norm()
    rng=np.random.default_rng(SEED)
    embedding_axes=[torch.tensor(rng.standard_normal(model.config.n_embd),dtype=torch.float64,device=device)
                    for _ in range(NULLS)]
    head_axes=[]
    for _ in range(NULLS):
        vector=torch.tensor(rng.standard_normal(model.config.n_embd),dtype=torch.float32,device=device)
        vector-=(vector@writer_axis)*writer_axis; head_axes.append(vector/vector.norm())
    answer_ids={"singular":authority.task14.ENCODING.encode(" is")[0],
                "plural":authority.task14.ENCODING.encode(" are")[0]}
    control_ids=[authority.task14.ENCODING.encode(" can")[0],authority.task14.ENCODING.encode(" will")[0]]
    counts={"forwards":0,"sequences":0,"embedding_nulls":NULLS,"head_nulls":NULLS,
            "fits":0,"backwards":0,"parameter_updates":0}
    geometry=[]; capture_error=0.; all_outputs=[]
    head_slice=slice(HEAD*HEAD_WIDTH,(HEAD+1)*HEAD_WIDTH)

    def process(batch_rows):
        nonlocal capture_error
        n=len(batch_rows); batch=torch.arange(n,device=device)
        tokens=torch.tensor([r["token_ids"] for r in batch_rows],dtype=torch.long,device=device)
        positions=torch.tensor([r["subject_position"] for r in batch_rows],dtype=torch.long,device=device)
        selected=torch.tensor([[r["native_answer_id"],answer_ids["plural" if r["number"]=="singular" else "singular"],*control_ids]
                               for r in batch_rows],dtype=torch.long,device=device)
        with torch.no_grad(): base=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float()
        subject=base[batch,positions].double(); projection=subject@unit_decoder
        target_projection=threshold/float(decoder_axis.norm()); orthogonal=subject-projection[:,None]*unit_decoder
        scale=torch.sqrt((subject.square().sum(1)-target_projection**2)/orthogonal.square().sum(1))
        removed_subject=target_projection*unit_decoder+scale[:,None]*orthogonal
        removed=base.clone(); removed[batch,positions]=removed_subject.float()
        distance=(removed_subject-subject).norm(dim=1); radius2=subject.square().sum(1)
        geometry.extend([float((removed_subject@decoder_axis-threshold).abs().max()),
                         float(((removed_subject.norm(dim=1)-subject.norm(dim=1)).abs()/subject.norm(dim=1)).max())])
        base_head=None; removed_head=None

        def run(initial,rescue=None,role=None):
            nonlocal base_head,removed_head,capture_error
            counts["forwards"]+=1; counts["sequences"]+=n; found={}
            with torch.no_grad():
                x=initial; x0=initial; first_value=None
                for layer,block in enumerate(model.transformer.h):
                    x=block.lambdas[0]*x+block.lambdas[1]*x0; handle=None
                    if layer==LAYER:
                        def hook(_module,arguments): found["head"]=arguments[0][batch,positions,head_slice].detach().clone()
                        handle=block.attn.c_proj.register_forward_pre_hook(hook)
                    try: attention,first_value=block.attn(F.rms_norm(x,(x.shape[-1],)),first_value)
                    finally:
                        if handle is not None: handle.remove()
                    if layer==LAYER:
                        current=found["head"]
                        if role=="base": base_head=current
                        elif role=="removed": removed_head=current
                        if rescue is not None:
                            if base_head is None or removed_head is None: raise RuntimeError("donor heads missing")
                            capture_error=max(capture_error,float((current-removed_head).abs().max()))
                            full=F.linear(base_head-current,block.attn.c_proj.weight[:,head_slice])
                            axis=(full@writer_axis)[:,None]*writer_axis
                            delta=full if rescue=="full" else axis if rescue=="axis" else axis.norm(dim=1)[:,None]*head_axes[rescue]
                            attention=attention.clone(); attention[batch,positions]+=delta.to(attention.dtype)
                    x=x+attention; x=x+block.mlp(F.rms_norm(x,(x.shape[-1],)))
                logits=model.lm_head(F.rms_norm(x,(x.shape[-1],))); logits=(30.*torch.tanh(logits/30.)).float()
            return logits[batch[:,None],positions[:,None],selected].detach().cpu().numpy()

        base_values=run(base,role="base"); removed_values=run(removed,role="removed")
        full_values=run(removed,rescue="full"); axis_values=run(removed,rescue="axis")
        embedding_values=[]; random_geometry=0.
        for random_axis in embedding_axes:
            tangent_direction=random_axis[None,:]-((subject@random_axis)/radius2)[:,None]*subject
            tangent_direction/=tangent_direction.norm(dim=1,keepdim=True)
            cosine=1.-distance.square()/(2.*radius2); sine=torch.sqrt(torch.clamp(1.-cosine.square(),min=0.))
            random_subject=cosine[:,None]*subject+sine[:,None]*subject.norm(dim=1)[:,None]*tangent_direction
            random_geometry=max(random_geometry,float((((random_subject-subject).norm(dim=1)-distance).abs()/distance).max()),
                                float(((random_subject.norm(dim=1)-subject.norm(dim=1)).abs()/subject.norm(dim=1)).max()))
            random_input=base.clone(); random_input[batch,positions]=random_subject.float()
            embedding_values.append(run(random_input))
        head_values=[run(removed,rescue=i) for i in range(NULLS)]
        return {"base":base_values,"removed":removed_values,"full":full_values,"axis":axis_values,
                "embedding":np.asarray(embedding_values),"head":np.asarray(head_values),
                "random_geometry":random_geometry,
                "decoder_scores":(subject@decoder_axis-threshold).cpu().numpy()}

    group_ids=[[i for i,r in enumerate(rows) if r["template_id"] in names]
               for names in (("near","behind"),("under","above"))]
    outputs=[process([rows[i] for i in ids]) for ids in group_ids]
    geometry.extend([o["random_geometry"] for o in outputs])
    def combine(key):
        sample=outputs[0][key]; shape=(len(rows),)+sample.shape[1:] if sample.ndim<3 else (sample.shape[0],len(rows),sample.shape[2])
        result=np.empty(shape,dtype=sample.dtype)
        for ids,out in zip(group_ids,outputs):
            if sample.ndim<3: result[ids]=out[key]
            else: result[:,ids]=out[key]
        return result
    base,removed,full,axis=(combine(k) for k in ("base","removed","full","axis"))
    embedding,head=combine("embedding"),combine("head"); decoder_scores=combine("decoder_scores")
    base_margin=base[:,0]-base[:,1]; removed_margin=removed[:,0]-removed[:,1]
    damage=base_margin-removed_margin
    full_rescue=(full[:,0]-full[:,1])-removed_margin; axis_rescue=(axis[:,0]-axis[:,1])-removed_margin
    embedding_damage=base_margin[None,:]-(embedding[:,:,0]-embedding[:,:,1])
    head_rescues=(head[:,:,0]-head[:,:,1])-removed_margin[None,:]
    control_change=(removed[:,2]-removed[:,3])-(base[:,2]-base[:,3])
    cell_ids=groups(rows); damage_cells={name:{"count":len(ids),"positive_fraction":float(np.mean(damage[ids]>0)),
        "rms":float(np.sqrt(np.mean(damage[ids]**2)))} for name,ids in cell_ids.items()}
    capability={name:{"count":len(ids),"accuracy":float(np.mean(base_margin[ids]>0))} for name,ids in cell_ids.items()}
    full_report=metrics(full_rescue,damage); axis_damage=metrics(axis_rescue,damage); axis_full=metrics(axis_rescue,full_rescue)
    embedding_rms=np.sqrt(np.mean(embedding_damage**2,axis=1)); head_reports=[metrics(x,full_rescue) for x in head_rescues]
    head_rms=np.asarray([x["effect_rms"] for x in head_reports]); head_errors=np.asarray([x["relative_l2_error"] for x in head_reports])
    damage_rms=damage_cells["overall"]["rms"]; native_rms=float(np.sqrt(np.mean(base_margin**2)))
    control_rms=float(np.sqrt(np.mean(control_change**2)))
    labels=np.asarray([1 if r["number"]=="plural" else -1 for r in rows])
    finite=bool(np.isfinite(np.asarray([*damage,*full_rescue,*axis_rescue,*embedding_damage.ravel(),*head_rescues.ravel()])).all())
    pred_a=bool(finite and np.all(decoder_scores*labels>0) and max(geometry)<=BARS["maximum_state_error"]
                and capture_error<=BARS["maximum_state_error"] and all(v["accuracy"]>=BARS["minimum_native_accuracy"] for v in capability.values())
                and counts==PRICE and checkpoint.weights_sha256==decoder["checkpoint_weights_sha256"])
    pred_b=bool(pred_a and damage_rms>=BARS["minimum_damage_rms"] and damage_rms/max(native_rms,1e-30)>=BARS["minimum_damage_fraction"]
                and all(v["positive_fraction"]>=BARS["minimum_cell_positive_fraction"] for v in damage_cells.values())
                and damage_rms>float(embedding_rms.max()) and damage_rms/max(float(np.median(embedding_rms)),1e-30)>=BARS["minimum_random_embedding_median_ratio"]
                and control_rms/max(damage_rms,1e-30)<=BARS["maximum_control_fraction"])
    pred_c=bool(pred_b and full_report["cosine"]>=BARS["minimum_full_damage_cosine"] and full_report["positive_fraction"]>=BARS["minimum_full_positive_fraction"]
                and BARS["minimum_full_damage_recovery"]<=full_report["least_squares_recovery"]<=BARS["maximum_full_damage_recovery"])
    pred_d=bool(pred_c and axis_full["cosine"]>=BARS["minimum_axis_full_cosine"] and axis_full["relative_l2_error"]<=BARS["maximum_axis_to_full_relative_l2"]
                and BARS["minimum_rank1_norm_ratio"]<=axis_full["predicted_to_actual_norm_ratio"]<=BARS["maximum_rank1_norm_ratio"]
                and BARS["minimum_axis_damage_recovery"]<=axis_damage["least_squares_recovery"]<=BARS["maximum_axis_damage_recovery"]
                and axis_full["relative_l2_error"]<float(head_errors.min()) and axis_full["effect_rms"]>=BARS["minimum_random_head_median_ratio"]*float(np.median(head_rms)))
    predictions=dict(zip(PREDICTION_REGISTRY,(pred_a,pred_b,pred_c,pred_d)))
    terminal="corrected_removal_and_rank1_mediation_held" if pred_d else "invalid" if not pred_a else "corrected_removal_mediation_null"
    result={"schema":"subject_number_corrected_removal_mediation_v1_result","terminal":terminal,"predictions":predictions,
            "instrument":{"finite":finite,"decoder_accuracy":float(np.mean(decoder_scores*labels>0)),"maximum_geometry_error":max(geometry),
                          "repeated_removed_head_capture_max_absolute_error":capture_error,"native_capability":capability,"counts":counts},
            "coordinate_removal":{"damage_rms":damage_rms,"native_margin_rms":native_rms,"damage_fraction":damage_rms/max(native_rms,1e-30),
                                  "cells":damage_cells,"control_change_rms":control_rms,"control_fraction":control_rms/max(damage_rms,1e-30)},
            "equal_distance_embedding_controls":{"count":NULLS,"seed":SEED,"damage_rms":embedding_rms.tolist(),"median_damage_rms":float(np.median(embedding_rms)),
                                                "maximum_damage_rms":float(embedding_rms.max()),"target_to_median_ratio":damage_rms/max(float(np.median(embedding_rms)),1e-30)},
            "full_head_rescue_to_damage":full_report,"rank1_rescue_to_damage":axis_damage,"rank1_rescue_to_full_head":axis_full,
            "matched_random_head_rescues":{"count":NULLS,"seed":SEED,"median_effect_rms":float(np.median(head_rms)),"minimum_relative_l2_error":float(head_errors.min()),"reports":head_reports},
            "records":[{"row_id":r["row_id"],"number":r["number"],"template":r["template_id"],"attractor_number":r["attractor_number"],
                       "native_margin":float(base_margin[i]),"damage":float(damage[i]),"full_head_rescue":float(full_rescue[i]),"rank1_rescue":float(axis_rescue[i])} for i,r in enumerate(rows)],
            "bars":BARS,"price":PRICE,"authority_sha256":authority.canonical(rows),"binding_sha256":sha(BINDING),"runner_sha256":sha(RUNNER),
            "checkpoint_weights_sha256":checkpoint.weights_sha256,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "wall_seconds":time.perf_counter()-started,"scope":"Prospective corrected-attractor behavioral removal and donor-restored L11H3 mediation replication."}
    managed.atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ("terminal","predictions","instrument","coordinate_removal","equal_distance_embedding_controls","full_head_rescue_to_damage","rank1_rescue_to_damage","rank1_rescue_to_full_head","matched_random_head_rescues")},indent=2,sort_keys=True))
    assert pred_a


if __name__=="__main__": main()
