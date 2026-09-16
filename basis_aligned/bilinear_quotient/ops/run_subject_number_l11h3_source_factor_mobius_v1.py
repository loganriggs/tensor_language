#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_source_factor_instrument pred_b_compact_factor_masks pred_c_compact_source_roles pred_d_sparse_source_factor_atoms pred_e_writer_specificity
"""Exact source-position × Q/K/Q2/K2/U Möbius decomposition of the L11H3 scalar."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

import attention_source_factor_primitive as source_factor
import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


RUNNER=Path(__file__).resolve(); ROOT=RUNNER.parents[3]; POLY=ROOT/"basis_aligned/polynomial_causal"
DECODER=POLY/"SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json"
RANK1=POLY/"SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
SCALAR=POLY/"SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_CORRECTED_V1_RESULT.json"
PREREG=POLY/"SUBJECT_NUMBER_L11H3_SOURCE_FACTOR_MOBIUS_V1_PREREGISTRATION.md"
BINDING=POLY/"SUBJECT_NUMBER_L11H3_SOURCE_FACTOR_MOBIUS_V1_BINDING.json"
OUT=POLY/"SUBJECT_NUMBER_L11H3_SOURCE_FACTOR_MOBIUS_V1_RESULT.json"
LAYER,HEAD=11,3; NULLS,SEED=16,20260925
ROLES=("context","determiner_1","attractor","punctuation","determiner_2","subject")
BARS={"maximum_absolute_closure_error":1e-4,"maximum_head_identity_error":1e-4,
      "maximum_relative_closure_error":1e-5,"maximum_scalar_replay_error":1e-4,
      "maximum_synthetic_error":1e-12,"maximum_atom8_relative_l2":.25,
      "maximum_factor4_relative_l2":.35,"maximum_source2_relative_l2":.35,
      "minimum_random_median_advantage":.10}
PRICE={"partial_forwards":4,"sequences":256,"mobius_terms":31,"readouts":17,
       "greedy_atom_steps":8,"greedy_factor_steps":4,"greedy_source_steps":2,
       "behavior_logits":0,"fits":0,"backwards":0,"parameter_updates":0}
PREDICTION_REGISTRY={"pred_a_exact_source_factor_instrument":None,
                     "pred_b_compact_factor_masks":None,"pred_c_compact_source_roles":None,
                     "pred_d_sparse_source_factor_atoms":None,"pred_e_writer_specificity":None}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_fixture():
    rng=np.random.default_rng(194); base=rng.normal(size=5); donor=rng.normal(size=5)
    values={}; dividends={}
    for mask in range(32):
        chosen=np.where([(mask>>i)&1 for i in range(5)],donor,base)
        values[mask]=float(np.prod(chosen)); dividend=values[mask]
        sub=(mask-1)&mask
        while sub: dividend-=dividends[sub]; sub=(sub-1)&mask
        if mask: dividend-=dividends[0]
        dividends[mask]=dividend
    return abs(sum(dividends[m] for m in range(1,32))-(values[31]-values[0]))


def greedy(target,atoms,steps,names):
    residual=target.copy(); selected=[]; curve=[]; available=list(range(len(atoms)))
    denom=max(float(np.linalg.norm(target)),1e-30)
    for _ in range(min(steps,len(available))):
        choice=min(available,key=lambda i:float(np.linalg.norm(residual-atoms[i])))
        residual=residual-atoms[choice]; available.remove(choice); selected.append(names[choice])
        curve.append(float(np.linalg.norm(residual)/denom))
    return {"selected":selected,"relative_l2_curve":curve,
            "final_relative_l2":curve[-1] if curve else 1.}


def load_bound():
    binding=json.loads(BINDING.read_text())
    paths={"authority":Path(authority.__file__),"source_primitive":Path(source_factor.__file__),
           "decoder":DECODER,"rank1":RANK1,"corrected_scalar":SCALAR,"preregistration":PREREG}
    if binding["files"]!={k:sha(v) for k,v in paths.items()} or binding["authority_sha256"]!=authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"]!=BARS or binding["nulls"]!=NULLS or binding["null_seed"]!=SEED or binding["price"]!=PRICE:
        raise ValueError("binding changed")
    decoder,rank,scalar=(json.loads(p.read_text()) for p in (DECODER,RANK1,SCALAR))
    if decoder["terminal"]!="embedding_number_decoder_frozen" or rank["terminal"]!="rank1_frozen_weights_only" \
            or scalar["terminal"]!="corrected_context_rank1_interaction_candidate": raise ValueError("parent status changed")
    rows=authority.build_rows()
    if any(len(r["token_ids"])!=6 or r["subject_position"]!=5 for r in rows): raise ValueError("source roles changed")
    return binding,decoder,rank,scalar,rows


def plan():
    _,decoder,rank,scalar,rows=load_bound()
    return {"schema":"subject_number_l11h3_source_factor_mobius_v1_plan","model_loaded":False,
            "gpu_accessed":False,"queue_touched":False,"rows":len(rows),"batch_sizes":[64,64],
            "source_roles":list(ROLES),"factor_names":list(source_factor.SOURCE_FACTORS),
            "mobius_terms":31,"readouts":17,"rank":rank["rank"],
            "decoder_axis_sha256":decoder["frozen_decoder"]["axis_float32_sha256"],
            "scalar_terminal":scalar["terminal"],"authority_sha256":authority.canonical(rows),
            "bars":BARS,"price":PRICE,"binding_sha256":sha(BINDING)}


def main():
    planned=plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned,indent=2,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    signal.alarm(600); binding,decoder,rank,scalar,rows=load_bound()
    torch,F,facade=tangent.parent.factors._dependencies(); torch.set_num_threads(2)
    model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
    started=time.perf_counter(); device=next(model.parameters()).device
    decoder_axis=torch.tensor(decoder["frozen_decoder"]["axis"],dtype=torch.float64,device=device)
    threshold=float(decoder["frozen_decoder"]["threshold"]); unit=decoder_axis/decoder_axis.norm()
    writer=torch.tensor(rank["axis"],dtype=torch.float32,device=device); writer/=writer.norm()
    rng=np.random.default_rng(SEED); axes=[writer]
    for _ in range(NULLS):
        v=torch.tensor(rng.standard_normal(model.config.n_embd),dtype=torch.float32,device=device)
        v-=(v@writer)*writer; axes.append(v/v.norm())
    axes=torch.stack(axes); counts={"partial_forwards":0,"sequences":0,"mobius_terms":31,"readouts":17,
        "greedy_atom_steps":8,"greedy_factor_steps":4,"greedy_source_steps":2,"behavior_logits":0,
        "fits":0,"backwards":0,"parameter_updates":0}
    all_atoms=[[] for _ in range(17)]; all_alpha=[[] for _ in range(17)]
    identity_error=closure_abs=closure_rel=0.; replay_errors=[]

    def capture(initial,positions):
        counts["partial_forwards"]+=1; counts["sequences"]+=len(initial); first=None; x=initial; x0=initial
        with torch.no_grad():
            for layer,block in enumerate(model.transformer.h):
                x=block.lambdas[0]*x+block.lambdas[1]*x0; state=F.rms_norm(x,(x.shape[-1],))
                if layer==LAYER:
                    _write,factors=source_factor.replay_attention_with_source_factors(
                        state,first,block.attn,positions,HEAD,torch,F,include_qk_factors=True)
                    return factors
                attention,first=block.attn(state,first); x=x+attention; x=x+block.mlp(F.rms_norm(x,(x.shape[-1],)))
        raise RuntimeError("layer not reached")

    group_ids=[[i for i,r in enumerate(rows) if r["template_id"] in names]
               for names in (("near","behind"),("under","above"))]
    recorded=np.asarray([r["alpha"] for r in scalar["records"]])
    for ids in group_ids:
        batch_rows=[rows[i] for i in ids]; tokens=torch.tensor([r["token_ids"] for r in batch_rows],dtype=torch.long,device=device)
        positions=torch.tensor([r["subject_position"] for r in batch_rows],dtype=torch.long,device=device)
        batch=torch.arange(len(ids),device=device)
        with torch.no_grad(): base_input=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float()
        subject=base_input[batch,positions].double(); projection=subject@unit; target=threshold/float(decoder_axis.norm())
        orthogonal=subject-projection[:,None]*unit; scale=torch.sqrt((subject.square().sum(1)-target**2)/orthogonal.square().sum(1))
        removed_subject=target*unit+scale[:,None]*orthogonal; removed_input=base_input.clone(); removed_input[batch,positions]=removed_subject.float()
        base,removed=capture(base_input,positions),capture(removed_input,positions)
        for factors in (base,removed):
            identity_error=max(identity_error,float((torch.einsum("bk,bkd->bd",factors["p"],factors["u"])-factors["head"]).abs().max()))
        dividends=source_factor.source_factor_mobius(base,removed,torch)
        vector_delta=base["head"]-removed["head"]
        closure=-sum(dividends[m].sum(1) for m in range(1,32))
        closure_abs=max(closure_abs,float((closure-vector_delta).abs().max()))
        closure_rel=max(closure_rel,float((closure-vector_delta).norm()/vector_delta.norm().clamp_min(1e-30)))
        direct=vector_delta@axes.T
        replay_errors.append(float(np.max(np.abs(direct[:,0].cpu().numpy()-recorded[ids]))))
        for ai in range(17):
            projected=torch.stack([-(dividends[m]@axes[ai]).float() for m in range(1,32)])
            all_atoms[ai].append(projected.cpu().numpy()); all_alpha[ai].append(direct[:,ai].cpu().numpy())

    mask_names=["*".join(source_factor.factor_names(m)) for m in range(1,32)]
    readout_reports=[]
    for ai in range(17):
        atoms=np.concatenate(all_atoms[ai],axis=1) # mask,row,source
        alpha=np.concatenate(all_alpha[ai])
        atom_vectors=np.asarray([atoms[m,:,s] for m in range(31) for s in range(6)])
        atom_names=[f"{ROLES[s]}:{mask_names[m]}" for m in range(31) for s in range(6)]
        factor_vectors=atoms.sum(2); source_vectors=atoms.sum(0).T
        report={"alpha_rms":float(np.sqrt(np.mean(alpha**2))),
                "atom_greedy":greedy(alpha,atom_vectors,8,atom_names),
                "factor_greedy":greedy(alpha,factor_vectors,4,mask_names),
                "source_greedy":greedy(alpha,source_vectors,2,list(ROLES))}
        if ai==0:
            report["factor_reports"]=[{"factor_mask":m+1,"factors":list(source_factor.factor_names(m+1)),
                "rms":float(np.sqrt(np.mean(factor_vectors[m]**2))),
                "aligned_recovery":float(factor_vectors[m]@alpha/max(alpha@alpha,1e-30))} for m in range(31)]
            report["source_reports"]=[{"source_position":s,"role":ROLES[s],
                "rms":float(np.sqrt(np.mean(source_vectors[s]**2))),
                "aligned_recovery":float(source_vectors[s]@alpha/max(alpha@alpha,1e-30))} for s in range(6)]
        readout_reports.append(report)
    target_report=readout_reports[0]; random_atom_errors=np.asarray([r["atom_greedy"]["final_relative_l2"] for r in readout_reports[1:]])
    synthetic=synthetic_fixture(); finite=bool(np.isfinite(np.asarray([identity_error,closure_abs,closure_rel,*replay_errors,*random_atom_errors])).all())
    pred_a=bool(finite and identity_error<=BARS["maximum_head_identity_error"] and closure_abs<=BARS["maximum_absolute_closure_error"]
                and closure_rel<=BARS["maximum_relative_closure_error"] and max(replay_errors)<=BARS["maximum_scalar_replay_error"]
                and synthetic<=BARS["maximum_synthetic_error"] and counts==PRICE and checkpoint.weights_sha256==decoder["checkpoint_weights_sha256"])
    pred_b=bool(pred_a and target_report["factor_greedy"]["final_relative_l2"]<=BARS["maximum_factor4_relative_l2"])
    pred_c=bool(pred_a and target_report["source_greedy"]["final_relative_l2"]<=BARS["maximum_source2_relative_l2"])
    pred_d=bool(pred_a and target_report["atom_greedy"]["final_relative_l2"]<=BARS["maximum_atom8_relative_l2"])
    advantage=float(np.median(random_atom_errors)-target_report["atom_greedy"]["final_relative_l2"])
    pred_e=bool(pred_d and advantage>=BARS["minimum_random_median_advantage"])
    predictions=dict(zip(PREDICTION_REGISTRY,(pred_a,pred_b,pred_c,pred_d,pred_e)))
    terminal="sparse_l11h3_source_factor_candidate" if pred_e else "valid_source_factor_decomposition" if pred_a else "invalid"
    result={"schema":"subject_number_l11h3_source_factor_mobius_v1_result","terminal":terminal,"predictions":predictions,
            "instrument":{"finite":finite,"head_identity_max_abs_error":identity_error,"mobius_closure_max_abs_error":closure_abs,
                          "mobius_closure_max_relative_l2":closure_rel,"scalar_replay_max_abs_error":max(replay_errors),
                          "synthetic_fixture_error":synthetic,"counts":counts},
            "writer_axis":target_report,"random_readout_specificity":{"count":NULLS,"seed":SEED,"atom8_relative_l2":random_atom_errors.tolist(),
                "median_atom8_relative_l2":float(np.median(random_atom_errors)),"target_atom8_relative_l2":target_report["atom_greedy"]["final_relative_l2"],"median_advantage":advantage},
            "bars":BARS,"price":PRICE,"authority_sha256":authority.canonical(rows),"binding_sha256":sha(BINDING),"runner_sha256":sha(RUNNER),
            "checkpoint_weights_sha256":checkpoint.weights_sha256,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "wall_seconds":time.perf_counter()-started,"scope":"Opened-panel exact native source-factor decomposition; selected atoms require fresh causal confirmation."}
    managed.atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ("terminal","predictions","instrument","writer_axis","random_readout_specificity")},indent=2,sort_keys=True))
    assert pred_a


if __name__=="__main__": main()
