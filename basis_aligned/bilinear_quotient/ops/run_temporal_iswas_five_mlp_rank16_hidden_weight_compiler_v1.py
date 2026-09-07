#!/usr/bin/env python3
"""Compile the bidirectional five-MLP rank-16 source interface into exact hidden weight factors."""
# BQGATE: EXPERIMENT pred_a_authority_basis_support_finiteness_and_price pred_b_exact_weight_compiler_closes pred_c_all_native_maps_have_full_task_rank pred_d_hidden_participation_is_nonuniform pred_e_manifest_is_gauge_invariant
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import pooled_response_projector as pooled
import run_temporal_iswas_five_mlp_position_svd_ladder_v1 as svd
import run_temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v1 as edge
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlas

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_iswas_five_mlp_rank16_hidden_weight_compiler_v1.json"
POS=ROOT/"circuits/followups/temporal_iswas_five_mlp_position_svd_ladder_v1_result.json"
GAIN=ROOT/"circuits/followups/temporal_iswas_five_mlp_rank16_gain_curve_v1_result.json"
REV=ROOT/"circuits/followups/temporal_iswas_five_mlp_rank16_gain115_reverse_v1_result.json"
BASIS_RUNNER=ROOT/"ops/run_temporal_iswas_five_mlp_position_svd_ladder_v1.py"
OUT=ROOT/"circuits/followups/temporal_iswas_five_mlp_rank16_hidden_weight_compiler_v1_result.json"
EXPECTED={"prior":None,"position":"b6f4562bcc6e19dc808a0debaf35d32b0d6c1dbd6c6a6e7846a92253b89dc014","gain":"958f58cf6ac05973b65a38cd258093926999351016f1aa1bc2ec608c4844b75d","reverse":"c66584f4ee94a2fc7d390b4abcb296ffec933237de43b3924221e7482f465d52","basis_runner":"cc2733a518e015b3cccde4d109e46125d69e90fa60555a593c921f6f0e860236"}
SUPPORT=("MLP0","MLP1","MLP2","MLP3","MLP6"); RANK=16; MAX_FORWARDS=8
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def valid(batch,x): return x[batch.valid_mask].float()
def capture(backend,batch):
    hidden={}; out={}; handles=[]
    for site in SUPPORT:
        layer=int(site[3:]); module=backend.model.transformer.h[layer].mlp
        handles.append(module.Down.register_forward_pre_hook(lambda _m,a,s=site:hidden.__setitem__(s,a[0].detach().clone())))
        handles.append(module.register_forward_hook(lambda _m,_a,y,s=site:out.__setitem__(s,y.detach().clone())))
    try: backend.native(batch,capture=False)
    finally:
        for h in handles:h.remove()
    return hidden,out
def population_report(backend,batch,b0,b1,o0,o1,bases,maps):
    records={}; worst=0.0
    for site in SUPPORT:
        dh=valid(batch,b1[site]-b0[site]); dy=valid(batch,o1[site]-o0[site]); q=bases[site]["rank16"]
        generic=(dy@q)@q.T; compiled=(dh@maps[site])@q.T
        rse=float((generic-compiled).square().sum()/generic.square().sum().clamp_min(1e-30)); worst=max(worst,rse)
        records[site]={"closure_rse":rse,"generic_norm":float(generic.norm()),"compiled_norm":float(compiled.norm())}
    return records,worst
def main():
    expected=dict(EXPECTED); expected["prior"]=sha(PRIOR)
    observed={"prior":sha(PRIOR),"position":sha(POS),"gain":sha(GAIN),"reverse":sha(REV),"basis_runner":sha(BASIS_RUNNER)}
    dry={"candidate_id":"temporal_auxiliary.iswas_five_mlp_rank16_hidden_weight_compiler_v1","dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"support":list(SUPPORT),"rank":RANK,"model_forwards_max":MAX_FORWARDS,"fit_updates":0,"model_updates":0,"transformer_backwards":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(dry,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    tic=time.perf_counter(); started=now(); backend=producer.Bilin18TorchBackend.load("cuda"); torch=backend.torch
    fitted=pooled.fit(backend); bases,_,_,_=svd.fit_position_bases(backend,fitted["target_rows"])
    fb,_,f0,_,f1,_,_=__import__('run_temporal_five_mlp_crossfit_response_weight_interface_v1').cap_inputs(backend,fitted["target_rows"])
    hb0,ho0=capture(backend,fb); hb1,ho1=capture(backend,__import__('circuit_das_subspace')._batch(backend,fitted["target_rows"],side="donor"))
    fresh=oodctx.capture(backend,__import__('run_temporal_iswas_rank46_task_typed_mode_causal_factorial_v1').TCAP,__import__('run_temporal_iswas_rank46_task_typed_mode_causal_factorial_v1').ICAP)
    fh0,fo0=capture(backend,fresh["batch"]); fh1,fo1=capture(backend,fresh["donor_batch"])
    maps={}; manifest={}; finite=[]; nonuniform=0; ranks=[]; gauge=[]
    g=torch.Generator(device="cpu");g.manual_seed(160115);rot=torch.linalg.qr(torch.randn(RANK,RANK,generator=g))[0].to(backend.device)
    for site in SUPPORT:
        layer=int(site[3:]);q=bases[site]["rank16"];down=backend.model.transformer.h[layer].mlp.Down.weight.detach().float();a=down.T@q;maps[site]=a
        sing=torch.linalg.svdvals(a);rank=int((sing>=sing[0]*1e-6).sum());ranks.append(rank);energy=a.square().sum(1).sort(descending=True).values;top=float(energy[:math.ceil(.1*len(energy))].sum()/energy.sum());nonuniform+=top>=.25
        eff=float(energy.sum().square()/energy.square().sum());qr=q@rot;ar=a@rot;probe=valid(fb,hb1[site]-hb0[site]);x=(probe@a)@q.T;y=(probe@ar)@qr.T;ge=float((x-y).norm()/x.norm().clamp_min(1e-30));gauge.append(ge)
        gram=float((q.T@q-torch.eye(RANK,device=q.device)).abs().max());finite += [*sing.tolist(),top,eff,ge,gram]
        manifest[site]={"basis_sha256":thash(q),"hidden_factor_sha256":thash(a),"basis_gram_max_abs":gram,"hidden_factor_shape":list(a.shape),"numerical_rank":rank,"singular_values":[float(v) for v in sing],"top_10pct_hidden_energy_fraction":top,"effective_hidden_participation":eff,"gauge_output_relative_error":ge}
    fit_records,fit_worst=population_report(backend,fb,hb0,hb1,ho0,ho1,bases,maps);fresh_records,fresh_worst=population_report(backend,fresh["batch"],fh0,fh1,fo0,fo1,bases,maps)
    pa=observed==expected and json.loads(POS.read_text())["selected_rank"]==16 and json.loads(REV.read_text())["terminal"]=="bidirectional_selective_five_mlp_rank16_program" and max(x["basis_gram_max_abs"] for x in manifest.values())<=1e-4 and all(math.isfinite(x) for x in finite)
    pb=max(fit_worst,fresh_worst)<=1e-10;pc=all(r==RANK for r in ranks);pd=nonuniform>=3;pe=max(gauge)<=1e-10
    preds={"pred_a_authority_basis_support_finiteness_and_price":bool(pa),"pred_b_exact_weight_compiler_closes":bool(pb),"pred_c_all_native_maps_have_full_task_rank":bool(pc),"pred_d_hidden_participation_is_nonuniform":bool(pd),"pred_e_manifest_is_gauge_invariant":bool(pe)}
    terminal="invalid" if not(pa and pb) else "weight_compiled_source_program" if all(preds.values()) else "distributed_weight_compiled_source_program" if pa and pb and pc and pe else "null"
    result={"schema":"temporal_iswas_five_mlp_rank16_hidden_weight_compiler_result_v1","started_utc":started,"finished_utc":now(),"serial_seconds":time.perf_counter()-tic,"authority_sha256":expected,"manifest":manifest,"fit":fit_records,"fresh":fresh_records,"summary":{"max_closure_rse":max(fit_worst,fresh_worst),"full_rank_sites":sum(r==RANK for r in ranks),"nonuniform_sites":nonuniform,"max_gauge_error":max(gauge)},"predictions":preds,"terminal":terminal,"price":{"model_forwards_max":MAX_FORWARDS,"fit_updates":0,"model_updates":0,"transformer_backwards":0}}
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("summary","predictions","terminal","price")},sort_keys=True))
if __name__=="__main__":main()
