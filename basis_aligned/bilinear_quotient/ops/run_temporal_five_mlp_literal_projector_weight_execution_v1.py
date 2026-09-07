#!/usr/bin/env python3
"""Execute frozen response projectors through literal c_proj and bilinear-Down contractions."""
# BQGATE: EXPERIMENT pred_a_authority_hash_replay_instrument_finiteness_and_price pred_b_attention_weight_contractions_close_locally pred_c_mlp_weight_contractions_close_locally pred_d_literal_execution_matches_generic_projector pred_e_literal_program_preserves_ood_behavior_and_controls
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population
import run_temporal_five_mlp_target_contrast_response_basis_v1 as v1
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_frozen_response_ood_regularization_v2 as ood
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_literal_projector_weight_execution_v1.json"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
KLFIT = ROOT / "circuits/followups/temporal_five_mlp_family_feasible_kl_refinement_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_literal_projector_weight_execution_v1_result.json"
EXPECTED = {"prior": "d424aea01f9eaceb3aaf0e3b43a851d4832c98bedaef3e9e6e52ed714b1d2ae1",
            "interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
            "klfit": "347d9ba45de6c5fdd54743c55f119eb3da529fcb8e25a1658b0549fe3b9c1853"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def compiled_run(backend, batch, base, donor, base_inputs, donor_inputs, bases):
    """Same intervention as generic projection, written only as weight contractions."""
    torch = backend.torch; width = backend.model.config.n_embd // backend.model.config.n_head
    handles = []; attention = {}; mlps = {}
    for site, q in bases.items():
        kind, layer, head = atlasrun.site_parts(site)
        (attention if kind == "attn" else mlps).setdefault(layer, []).append((site, head, q))
    hidden_delta = {}
    with torch.no_grad():
        for layer, entries in mlps.items():
            module = backend.model.transformer.h[layer].mlp
            for site, _head, _q in entries:
                xb, xd = base_inputs[site].to(backend.device), donor_inputs[site].to(backend.device)
                hidden_delta[site] = (module.Left(xd).float() * module.Right(xd).float()
                                      - module.Left(xb).float() * module.Right(xb).float())
    for layer, entries in attention.items():
        module = backend.model.transformer.h[layer].attn.c_proj
        def ahook(mod, args, output, layer=layer, entries=tuple(entries)):
            live = args[0]; changed = output.clone()
            for _site, head, q in entries:
                a, z = head * width, (head + 1) * width
                w = mod.weight.detach().float()[:, a:z]; weight_map = w @ q
                for i, pos in enumerate(batch.semantic_positions):
                    n = int(pos) + 1
                    b = base["attention"][layer][i, :n, a:z].to(live).float()
                    d = donor["attention"][layer][i, :n, a:z].to(live).float() - b
                    correction = (b - live[i, :n, a:z].float()) @ w.T + (d @ q) @ weight_map.T
                    changed[i, :n] = changed[i, :n] + correction.to(changed)
            return changed
        handles.append(module.register_forward_hook(ahook))
    for layer, entries in mlps.items():
        module = backend.model.transformer.h[layer].mlp
        def mhook(mod, _args, output, layer=layer, entries=tuple(entries)):
            changed = output.clone(); w = mod.Down.weight.detach().float()
            for site, _head, q in entries:
                weight_map = w.T @ q; dh = hidden_delta[site]
                for i, pos in enumerate(batch.semantic_positions):
                    n = int(pos) + 1
                    b = base["mlp"][layer][i, :n].to(output).float()
                    projected = (dh[i, :n] @ weight_map) @ q.T
                    changed[i, :n] = b.to(changed) + projected.to(changed)
            return changed
        handles.append(module.register_forward_hook(mhook))
    try: return backend.native(batch, capture=True)
    finally:
        for handle in handles: handle.remove()


def local_closure(backend, batch, base, donor, base_inputs, donor_inputs, bases):
    torch = backend.torch; width = backend.model.config.n_embd // backend.model.config.n_head
    records = {}
    for site, q in bases.items():
        kind, layer, head = atlasrun.site_parts(site); direct, literal = [], []
        if kind == "attn":
            a, z = head * width, (head + 1) * width
            w = backend.model.transformer.h[layer].attn.c_proj.weight.detach().float()[:, a:z]
            wm = w @ q
            for i, pos in enumerate(batch.semantic_positions):
                d = (donor["attention"][layer][i, :int(pos)+1, a:z] - base["attention"][layer][i, :int(pos)+1, a:z]).to(backend.device).float()
                direct.append(((d @ q) @ q.T) @ w.T); literal.append((d @ q) @ wm.T)
        else:
            module = backend.model.transformer.h[layer].mlp; w = module.Down.weight.detach().float(); wm = w.T @ q
            xb, xd = base_inputs[site].to(backend.device), donor_inputs[site].to(backend.device)
            hb = module.Left(xb).float() * module.Right(xb).float(); hd = module.Left(xd).float() * module.Right(xd).float()
            for i, pos in enumerate(batch.semantic_positions):
                dh = hd[i, :int(pos)+1] - hb[i, :int(pos)+1]
                direct.append(((dh @ w.T @ q) @ q.T)); literal.append((dh @ wm) @ q.T)
        a, b = torch.cat(direct), torch.cat(literal)
        records[site] = {"kind": kind, "rse": float((a-b).square().sum()/a.square().sum().clamp_min(1e-30)),
                         "weight_map_sha256": interface.thash((w @ q) if kind == "attn" else (w.T @ q))}
    return records


def comparison(backend, rows, generic, literal):
    torch = backend.torch
    gs, ls = [atlasrun.states(torch, backend, x, rows) for x in (generic, literal)]
    gl, ll = das.head_logits(backend, gs).float(), das.head_logits(backend, ls).float()
    gc, lc = gl-gl.mean(-1, keepdim=True), ll-ll.mean(-1, keepdim=True)
    return {"state_rse": float((ls-gs).square().sum()/gs.square().sum().clamp_min(1e-30)),
            "centered_logits_rse": float((lc-gc).square().sum()/gc.square().sum().clamp_min(1e-30)),
            "state_max_abs": float((ls-gs).abs().max()), "logits_max_abs": float((ll-gl).abs().max())}, gs, ls


def main():
    if {"prior": sha(PRIOR), "interface": sha(INTERFACE), "klfit": sha(KLFIT)} != EXPECTED: raise RuntimeError("literal projector authority changed")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_literal_projector_weight_execution_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "sites": comp.SITES,
           "projectors": ("even_fit", "odd_fit"), "model_forwards_max": 21, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    iface, kl = json.loads(INTERFACE.read_text()), json.loads(KLFIT.read_text())
    if iface["terminal"] != "stable_weight_readable_response_program" or kl["terminal"] != "kl_refinement_inconclusive": raise RuntimeError("upstream terminal changed")
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    reader, orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))
    old_tcap, old_icap = json.loads(comp.TCAP.read_text()), json.loads(comp.ICAP.read_text()); old_tr, old_ir, *_ = greedy.rows_and_controls(old_tcap, old_icap)
    controls = matched.control_rows(); te, to = old_tr[::2]+old_ir[::2], old_tr[1::2]+old_ir[1::2]
    ceb,_,ce0,_,ce1,_,_=interface.cap_inputs(backend,controls["discovery"]); cob,_,co0,_,co1,_,_=interface.cap_inputs(backend,controls["validation"])
    teb,_,te0,_,te1,_,_=interface.cap_inputs(backend,te); tob,_,to0,_,to1,_,_=interface.cap_inputs(backend,to)
    cbe,_=comp.fit_bases(backend,ceb,ce0,ce1); cbo,_=comp.fit_bases(backend,cob,co0,co1); qe,_=v1.fit_target(backend,teb,te0,te1,cbe); qo,_=v1.fit_target(backend,tob,to0,to1,cbo)
    projectors={"even_fit":qe,"odd_fit":qo}; hashes_ok=all([interface.thash(qe[s]),interface.thash(qo[s])]==iface["records"][s]["fit_basis_sha256"] for s in comp.SITES)
    stcap,sicap=json.loads(klfit.STC.read_text()),json.loads(klfit.SIC.read_text())
    tr=sum((population.capable_rows(klfit.sealed_t,stcap,p,12) for p in ("A1","A2")),[]); ir=sum((population.capable_rows(klfit.sealed_i,sicap,p,12) for p in ("A1","A2")),[]); rows=tr+ir
    crows=[r for r in klfit.sealed_t.build_rows() if r["transform_id"]=="P"][:16]
    tb,tbo,t0,_tdo,t1,tbi,tdi=interface.cap_inputs(backend,rows); cb,cbo,c0,_cdo,c1,cbi,cdi=interface.cap_inputs(backend,crows)
    base_target=atlasrun.states(torch,backend,tbo,rows); full_out,_=atlasrun.run_patch(backend,tb,t1,comp.SITES); full_target=atlasrun.states(torch,backend,full_out,rows)
    base_control=atlasrun.states(torch,backend,cbo,crows); scales=json.loads(comp.GENERIC.read_text())["target_behavior_rms_scales"]
    reports={}; attention_rse=[]; mlp_rse=[]; execution=[]
    for label,bases in projectors.items():
        tg=ood.run_project(backend,tb,t0,t1,bases); tl=compiled_run(backend,tb,t0,t1,tbi,tdi,bases)
        cg=ood.run_project(backend,cb,c0,c1,bases); cl=compiled_run(backend,cb,c0,c1,cbi,cdi,bases)
        tc,tgs,tls=comparison(backend,rows,tg,tl); cc,cgs,cls=comparison(backend,crows,cg,cl); execution += [tc["state_rse"],tc["centered_logits_rse"],cc["state_rse"],cc["centered_logits_rse"]]
        local=local_closure(backend,tb,t0,t1,tbi,tdi,bases)
        attention_rse += [x["rse"] for x in local.values() if x["kind"]=="attn"]; mlp_rse += [x["rse"] for x in local.values() if x["kind"]=="mlp"]
        control_ctx = {"rows": crows, "base_logits": das.head_logits(backend, base_control).float()}
        control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, cls).float())
        reports[label]={"local":local,"target":ood.target_report(backend,rows,base_target,full_target,tls,reader,len(tr)),"control":control,"generic_vs_literal_target":tc,"generic_vs_literal_control":cc}
    finite=[x for r in reports.values() for x in list(r["target"]["cells"].values())+list(r["target"]["behavior_signed_projection"].values())+list(r["control"]["margin_rms_fraction"].values())+list(r["generic_vs_literal_target"].values())+list(r["generic_vs_literal_control"].values())]
    pa=hashes_ok and reader_ok and orientation<=1e-6 and all(math.isfinite(x) for x in finite); pb=max(attention_rse)<=1e-10; pc=max(mlp_rse)<=1e-10; pd=max(execution)<=1e-5
    pe=all(r["target"]["worst_target_residual"]<=.05 and min(r["target"]["behavior_signed_projection"].values())>=.8 and max(r["control"]["margin_rms_fraction"].values())<=.1 for r in reports.values())
    predictions={"pred_a_authority_hash_replay_instrument_finiteness_and_price":bool(pa),"pred_b_attention_weight_contractions_close_locally":bool(pb),"pred_c_mlp_weight_contractions_close_locally":bool(pc),"pred_d_literal_execution_matches_generic_projector":bool(pd),"pred_e_literal_program_preserves_ood_behavior_and_controls":bool(pe)}
    terminal="invalid" if not(pa and pb and pc) else "literal_weight_executable_response_program" if all(predictions.values()) else "weight_execution_mismatch"
    summary={"max_attention_local_rse":max(attention_rse),"max_mlp_local_rse":max(mlp_rse),"max_execution_rse":max(execution),"target_worst_max":max(r["target"]["worst_target_residual"] for r in reports.values()),"target_projection_min":min(min(r["target"]["behavior_signed_projection"].values()) for r in reports.values()),"control_margin_max":max(max(r["control"]["margin_rms_fraction"].values()) for r in reports.values())}
    result={"schema":"temporal_five_mlp_literal_projector_weight_execution_result_v1","started_utc":started,"finished_utc":now(),"serial_seconds":time.perf_counter()-tic,"authority_sha256":EXPECTED,"reports":reports,"summary":summary,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":21,"fit_updates":0,"model_updates":0,"transformer_backwards":0}}
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("summary","predictions","terminal","price")},sort_keys=True))


if __name__=="__main__":main()
