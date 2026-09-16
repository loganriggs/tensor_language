#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_upstream_instrument pred_b_compact_upstream_terms pred_c_writer_specificity
"""Correct integer-keyed rerun of the V1 upstream subject-value decomposition."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib,json,os,signal,time
from pathlib import Path
import numpy as np

import attention_source_factor_primitive as source_factor
import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as authority
import circuit_fast_screen_managed_runner as managed
import run_subject_number_l11h3_subject_value_upstream_mobius_v1 as failed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent

RUNNER=Path(__file__).resolve(); ROOT=RUNNER.parents[3]; POLY=ROOT/"basis_aligned/polynomial_causal"
DECODER=failed.DECODER; RANK1=failed.RANK1; PARENT=failed.PARENT
FAILED_RESULT=POLY/"SUBJECT_NUMBER_L11H3_SUBJECT_VALUE_UPSTREAM_MOBIUS_V1_RESULT.json"
PREREG=POLY/"SUBJECT_NUMBER_L11H3_SUBJECT_VALUE_UPSTREAM_MOBIUS_V2_PREREGISTRATION.md"
BINDING=POLY/"SUBJECT_NUMBER_L11H3_SUBJECT_VALUE_UPSTREAM_MOBIUS_V2_BINDING.json"
OUT=POLY/"SUBJECT_NUMBER_L11H3_SUBJECT_VALUE_UPSTREAM_MOBIUS_V2_RESULT.json"
PORTS=failed.PORTS; BARS=failed.BARS; PRICE=failed.PRICE; NULLS=failed.NULLS; SEED=failed.SEED
LAYER,HEAD,HEAD_WIDTH=failed.LAYER,failed.HEAD,failed.HEAD_WIDTH
PREDICTION_REGISTRY={"pred_a_exact_upstream_instrument":None,
                     "pred_b_compact_upstream_terms":None,
                     "pred_c_writer_specificity":None}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_bound():
    binding=json.loads(BINDING.read_text()); paths={"authority":Path(authority.__file__),"source_primitive":Path(source_factor.__file__),
        "failed_runner":Path(failed.__file__),"failed_result":FAILED_RESULT,"decoder":DECODER,"rank1":RANK1,
        "parent_result":PARENT,"preregistration":PREREG}
    if binding["files"]!={k:sha(v) for k,v in paths.items()} or binding["authority_sha256"]!=authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"]!=BARS or binding["nulls"]!=NULLS or binding["null_seed"]!=SEED or binding["price"]!=PRICE: raise ValueError("binding changed")
    decoder,rank,parent_result,failed_result=(json.loads(p.read_text()) for p in (DECODER,RANK1,PARENT,FAILED_RESULT))
    if decoder["terminal"]!="embedding_number_decoder_frozen" or rank["terminal"]!="rank1_frozen_weights_only" \
            or parent_result["terminal"]!="valid_source_factor_decomposition" or failed_result["terminal"]!="invalid": raise ValueError("parent status changed")
    return binding,decoder,rank,parent_result,failed_result,authority.build_rows()

def plan():
    _,decoder,rank,parent_result,failed_result,rows=load_bound()
    return {"schema":"subject_number_l11h3_subject_value_upstream_mobius_v2_plan","model_loaded":False,"gpu_accessed":False,"queue_touched":False,
            "rows":len(rows),"batch_sizes":[64,64],"ports":list(PORTS),"correction":"integer_layer_keys","failed_v1_terminal":failed_result["terminal"],
            "parent_terminal":parent_result["terminal"],"rank":rank["rank"],"authority_sha256":authority.canonical(rows),
            "bars":BARS,"price":PRICE,"binding_sha256":sha(BINDING),"decoder_axis_sha256":decoder["frozen_decoder"]["axis_float32_sha256"]}

def main():
    planned=plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned,indent=2,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    signal.alarm(600); binding,decoder,rank,parent_result,failed_result,rows=load_bound()
    torch,F,facade=tangent.parent.factors._dependencies(); torch.set_num_threads(2)
    model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True); started=time.perf_counter(); device=next(model.parameters()).device
    decoder_axis=torch.tensor(decoder["frozen_decoder"]["axis"],dtype=torch.float64,device=device); threshold=float(decoder["frozen_decoder"]["threshold"]); unit=decoder_axis/decoder_axis.norm()
    writer=torch.tensor(rank["axis"],dtype=torch.float32,device=device); writer/=writer.norm(); rng=np.random.default_rng(SEED); axes=[writer]
    for _ in range(NULLS):
        v=torch.tensor(rng.standard_normal(model.config.n_embd),dtype=torch.float32,device=device); v-=(v@writer)*writer; axes.append(v/v.norm())
    axes=torch.stack(axes); counts={"partial_forwards":0,"sequences":0,"corners":32,"mobius_terms":31,"readouts":17,"greedy_steps":4,
        "behavior_logits":0,"fits":0,"backwards":0,"parameter_updates":0}
    component_error=value_replay=closure_abs=closure_rel=0.; all_atoms=[[] for _ in range(17)]; all_target=[[] for _ in range(17)]

    def capture(initial,positions):
        counts["partial_forwards"]+=1; counts["sequences"]+=len(initial); x=initial; x0=initial; first=None
        components={("embedding",-1):initial.clone()}
        with torch.no_grad():
            for layer,block in enumerate(model.transformer.h):
                x=block.lambdas[0]*x+block.lambdas[1]*x0
                for key in components: components[key]=components[key]*block.lambdas[0]
                components[("embedding",-1)]+=block.lambdas[1]*x0; state=F.rms_norm(x,(x.shape[-1],))
                if layer==LAYER:
                    _write,factors=source_factor.replay_attention_with_source_factors(state,first,block.attn,positions,HEAD,torch,F,include_qk_factors=True)
                    zero=torch.zeros_like(x)
                    grouped={"embedding_recurrence":components[("embedding",-1)],
                        "early_writes_0_3":sum((value for (kind,index),value in components.items() if kind!="embedding" and index in range(0,4)),zero),
                        "middle_writes_4_7":sum((value for (kind,index),value in components.items() if kind!="embedding" and index in range(4,8)),zero),
                        "late_writes_8_10":sum((value for (kind,index),value in components.items() if kind!="embedding" and index in range(8,11)),zero)}
                    return x,grouped,first,factors
                attention,first=block.attn(state,first); x=x+attention; components[("attn",layer)]=attention
                mlp=block.mlp(F.rms_norm(x,(x.shape[-1],))); x=x+mlp; components[("mlp",layer)]=mlp
        raise RuntimeError("layer not reached")

    group_ids=[[i for i,r in enumerate(rows) if r["template_id"] in names] for names in (("near","behind"),("under","above"))]
    for ids in group_ids:
        batch_rows=[rows[i] for i in ids]; tokens=torch.tensor([r["token_ids"] for r in batch_rows],dtype=torch.long,device=device); pos=torch.full((len(ids),),5,device=device,dtype=torch.long); b=torch.arange(len(ids),device=device)
        with torch.no_grad(): base_input=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float()
        subject=base_input[b,pos].double(); projection=subject@unit; target_projection=threshold/float(decoder_axis.norm()); orthogonal=subject-projection[:,None]*unit
        scale=torch.sqrt((subject.square().sum(1)-target_projection**2)/orthogonal.square().sum(1)); removed_subject=target_projection*unit+scale[:,None]*orthogonal
        removed_input=base_input.clone(); removed_input[b,pos]=removed_subject.float()
        raw_b,groups_b,first_b,factors_b=capture(base_input,pos); raw_r,groups_r,first_r,factors_r=capture(removed_input,pos)
        component_error=max(component_error,float((sum(groups_b.values())-raw_b).abs().max()),float((sum(groups_r.values())-raw_r).abs().max()))
        attention=model.transformer.h[LAYER].attn; head_slice=attention.c_proj.weight[:,HEAD*HEAD_WIDTH:(HEAD+1)*HEAD_WIDTH]; p_subject=factors_b["p"][:,5]
        def corner(mask):
            raw=sum((groups_r if mask&(1<<i) else groups_b)[PORTS[i]] for i in range(4)); state=F.rms_norm(raw,(raw.shape[-1],))
            current=attention.c_v(state).view(len(ids),6,9,128)[:,:,HEAD]; first=(first_r if mask&(1<<4) else first_b).view(len(ids),6,9,128)[:,:,HEAD]
            effective=(1-attention.lamb)*current+attention.lamb*first; u=F.linear(effective[:,5],head_slice); return p_subject[:,None]*(u@axes.T)
        values={mask:corner(mask) for mask in range(32)}; dividends={}
        for mask in range(32):
            div=values[mask].clone(); sub=(mask-1)&mask
            while sub: div-=dividends[sub]; sub=(sub-1)&mask
            if mask: div-=dividends[0]
            dividends[mask]=div
        target=values[0]-values[31]; closure=-sum(dividends[m] for m in range(1,32))
        closure_abs=max(closure_abs,float((closure-target).abs().max())); closure_rel=max(closure_rel,float((closure-target).norm()/target.norm().clamp_min(1e-30)))
        direct=(factors_b["p"][:,5,None]*(factors_b["u"][:,5]-factors_r["u"][:,5]))@axes.T; value_replay=max(value_replay,float((direct-target).abs().max()))
        for ai in range(17):
            all_atoms[ai].append(torch.stack([-dividends[m][:,ai] for m in range(1,32)]).cpu().numpy()); all_target[ai].append(target[:,ai].cpu().numpy())
    names=["*".join(PORTS[i] for i in range(5) if m&(1<<i)) for m in range(1,32)]; reports=[]
    for ai in range(17):
        atoms=np.concatenate(all_atoms[ai],axis=1); target=np.concatenate(all_target[ai]); report={"target_rms":float(np.sqrt(np.mean(target**2))),"greedy":failed.greedy(target,atoms,names)}
        if ai==0: report["terms"]=[{"mask":m+1,"ports":[PORTS[i] for i in range(5) if (m+1)&(1<<i)],"order":(m+1).bit_count(),
            "rms":float(np.sqrt(np.mean(atoms[m]**2))),"aligned_recovery":float(atoms[m]@target/max(target@target,1e-30))} for m in range(31)]
        reports.append(report)
    target_report=reports[0]; random_errors=np.asarray([r["greedy"]["final_relative_l2"] for r in reports[1:]]); advantage=float(np.median(random_errors)-target_report["greedy"]["final_relative_l2"])
    synthetic=failed.parent.synthetic_fixture(); finite=bool(np.isfinite(np.asarray([component_error,value_replay,closure_abs,closure_rel,synthetic,*random_errors])).all())
    pred_a=bool(finite and component_error<=BARS["maximum_component_reconstruction_error"] and value_replay<=BARS["maximum_native_value_replay_error"]
        and closure_abs<=BARS["maximum_absolute_closure_error"] and closure_rel<=BARS["maximum_relative_closure_error"] and synthetic<=BARS["maximum_synthetic_error"]
        and counts==PRICE and checkpoint.weights_sha256==decoder["checkpoint_weights_sha256"])
    pred_b=bool(pred_a and target_report["greedy"]["final_relative_l2"]<=BARS["maximum_term4_relative_l2"]); pred_c=bool(pred_b and advantage>=BARS["minimum_random_median_advantage"])
    predictions=dict(zip(PREDICTION_REGISTRY,(pred_a,pred_b,pred_c))); terminal="compact_subject_value_upstream_candidate" if pred_c else "valid_subject_value_upstream_decomposition" if pred_a else "invalid"
    result={"schema":"subject_number_l11h3_subject_value_upstream_mobius_v2_result","terminal":terminal,"predictions":predictions,
        "instrument":{"finite":finite,"component_reconstruction_max_abs_error":component_error,"native_subject_value_replay_max_abs_error":value_replay,
            "mobius_closure_max_abs_error":closure_abs,"mobius_closure_max_relative_l2":closure_rel,"synthetic_fixture_error":synthetic,"counts":counts},
        "writer_axis":target_report,"random_readout_specificity":{"count":NULLS,"seed":SEED,"term4_relative_l2":random_errors.tolist(),
            "median_term4_relative_l2":float(np.median(random_errors)),"target_term4_relative_l2":target_report["greedy"]["final_relative_l2"],"median_advantage":advantage},
        "correction":{"failed_v1_terminal":failed_result["terminal"],"bug":"string prefix made layer 10 match layer 1","fix":"integer layer keys"},
        "bars":BARS,"price":PRICE,"authority_sha256":authority.canonical(rows),"binding_sha256":sha(BINDING),"runner_sha256":sha(RUNNER),
        "checkpoint_weights_sha256":checkpoint.weights_sha256,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "wall_seconds":time.perf_counter()-started,"scope":"Corrected opened-panel exact five-port upstream decomposition; V1 remains invalid."}
    managed.atomic_create_json(OUT,result); print(json.dumps({k:result[k] for k in ("terminal","predictions","instrument","writer_axis","random_readout_specificity","correction")},indent=2,sort_keys=True)); assert pred_a

if __name__=="__main__": main()
